/**
 * Kernel 01: Naive Flash Attention — AMD HIP Port
 *
 * Ported from kernel_01_naive.cu (CUDA) to HIP for AMD GPU compatibility.
 * Algorithm is identical; only the GPU API calls differ.
 *
 * Key CUDA → HIP differences applied here:
 *   1. Header:  <cuda_runtime.h>        → <hip/hip_runtime.h>
 *   2. Warp shuffle:
 *              __shfl_xor_sync(mask, v, off) → __shfl_xor(v, off)
 *      (HIP wavefront is 64 threads on RDNA/CDNA; for correctness on 64-wide
 *       wavefronts the kernel uses WAVEFRONT_SIZE=64 when compiled for AMD.)
 *   3. API prefix:  cuda*               → hip*
 *   4. cudaError_t / cudaSuccess / cudaGetLastError → hip equivalents
 *   5. NVCC-specific pragmas are still valid with hipcc.
 *
 * IMPORTANT — Wavefront size note:
 *   NVIDIA: warp  = 32 threads (always).
 *   AMD GCN/RDNA2+: wavefront = 64 threads by default;
 *                   RDNA3 can also run 32-thread wave32 mode.
 *   This file compiles in WAVE64 mode (default hipcc behavior).
 *   Each query row is now handled by 2 warps-worth of NVIDIA threads,
 *   but by a single 64-thread wavefront on AMD.
 *   We keep WARP_SIZE = 64 when WAVEFRONT_SIZE is defined by hipcc;
 *   fall back to 32 otherwise (e.g. wave32 devices or CUDA compat mode).
 *
 * Algorithm (online softmax, per query row i):
 *   Init: m_i = -inf, l_i = 0, acc_i = 0
 *   For j in [0, seq_len):
 *     s_ij  = Q_i . K_j * scale
 *     m_new = max(m_i, s_ij)
 *     alpha = exp(m_i - m_new)
 *     beta  = exp(s_ij - m_new)
 *     l_new = alpha * l_i + beta
 *     acc_i = alpha * acc_i + beta * V_j
 *     m_i = m_new, l_i = l_new
 *   O_i = acc_i / l_i
 *
 * Reference: https://lubits.ch/flash/  |  arXiv:2307.08691
 *
 * Grid:  (seq_len, num_heads, batch_size)
 * Block: (WF_SIZE, 1, 1)   where WF_SIZE = wavefront width (64 or 32)
 *
 * Compile (AMD GPU):
 *   hipcc -O3 -o flash_attn_hip kernels/kernel_01_naive.hip
 *
 * Compile (verify syntax only, no AMD hardware):
 *   hipcc --offload-arch=gfx1100 --no-gpu-bundle-output -c kernels/kernel_01_naive.hip
 */

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cmath>

// ---------------------------------------------------------------------------
// Wavefront width — AMD default is 64; wave32 mode or CUDA compat gives 32.
// hipcc defines __HIP_PLATFORM_HCC__ (or __HIP_PLATFORM_AMD__) so we can
// detect the target at compile time.
// ---------------------------------------------------------------------------
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
  #define WF_SIZE 64
#else
  #define WF_SIZE 32   // CUDA compat / testing path
#endif

#define NEG_INF (-1e30f)

// ---------------------------------------------------------------------------
// Warp / wavefront-level reduction primitives
//
// HIP note: __shfl_xor() is the un-masked variant; on AMD hardware the full
// wavefront participates implicitly.  __shfl_xor_sync() is also supported in
// newer ROCm (≥ 5.3) for forward compatibility.
// ---------------------------------------------------------------------------

__device__ __forceinline__ float wf_reduce_sum(float val) {
#if WF_SIZE == 64
    // Two rounds of butterfly: first across the upper 32 lanes, then lower 16…1
    val += __shfl_xor(val, 32);
#endif
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor(val, offset);
    }
    return val;
}

__device__ __forceinline__ float wf_reduce_max(float val) {
#if WF_SIZE == 64
    val = fmaxf(val, __shfl_xor(val, 32));
#endif
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor(val, offset));
    }
    return val;
}

// ---------------------------------------------------------------------------
// Template kernel
//
// Template parameter ELEMS_PER_THREAD accounts for the wider wavefront:
//   AMD WF=64, HEAD_DIM=64 → each thread handles 1 element.
//   AMD WF=64, HEAD_DIM=128 → each thread handles 2 elements.
//   NVIDIA WF=32, HEAD_DIM=64 → each thread handles 2 elements. (compat mode)
// ---------------------------------------------------------------------------

template <int HEAD_DIM, int ELEMS_PER_THREAD>
__global__ void flash_attn_hip_v1_impl(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int seq_len,
    int head_dim,
    int64_t batch_stride,
    int64_t head_stride,
    float softmax_scale
) {
    const int query_row = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int batch_idx = blockIdx.z;
    const int tid       = threadIdx.x;   // 0 .. WF_SIZE-1

    const int64_t base = batch_idx * batch_stride + head_idx * head_stride;
    const float* Q_base = Q + base;
    const float* K_base = K + base;
    const float* V_base = V + base;
    float*       O_base = O + base;

    const float* Q_i = Q_base + query_row * head_dim;

    // -----------------------------------------------------------------------
    // Load Q_i into registers.
    // Thread tid handles elements: tid, tid+WF_SIZE, tid+2*WF_SIZE, ...
    // -----------------------------------------------------------------------
    float q_local[ELEMS_PER_THREAD];
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) {
        int idx = tid + e * WF_SIZE;
        q_local[e] = (idx < HEAD_DIM) ? Q_i[idx] : 0.0f;
    }

    // -----------------------------------------------------------------------
    // Online softmax accumulators
    // -----------------------------------------------------------------------
    float m_i = NEG_INF;   // running max
    float l_i = 0.0f;      // running sum of exp(S - m)

    float acc_local[ELEMS_PER_THREAD];
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) {
        acc_local[e] = 0.0f;
    }

    // -----------------------------------------------------------------------
    // Main KV loop
    // -----------------------------------------------------------------------
    for (int kv_row = 0; kv_row < seq_len; kv_row++) {
        const float* K_j = K_base + kv_row * head_dim;
        const float* V_j = V_base + kv_row * head_dim;

        // Step 1 — partial dot product per thread, wavefront-reduce to scalar
        float dot = 0.0f;
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_THREAD; e++) {
            int idx = tid + e * WF_SIZE;
            if (idx < HEAD_DIM) {
                dot += q_local[e] * K_j[idx];
            }
        }
        float s_ij = wf_reduce_sum(dot) * softmax_scale;

        // Step 2 — online softmax update
        float m_new  = fmaxf(m_i, s_ij);
        float alpha  = expf(m_i   - m_new);
        float beta   = expf(s_ij  - m_new);
        float l_new  = alpha * l_i + beta;

        #pragma unroll
        for (int e = 0; e < ELEMS_PER_THREAD; e++) {
            int idx = tid + e * WF_SIZE;
            float v_val = (idx < HEAD_DIM) ? V_j[idx] : 0.0f;
            acc_local[e] = alpha * acc_local[e] + beta * v_val;
        }

        m_i = m_new;
        l_i = l_new;
    }

    // -----------------------------------------------------------------------
    // Final normalisation — write output
    // -----------------------------------------------------------------------
    float inv_l = 1.0f / l_i;
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) {
        int idx = tid + e * WF_SIZE;
        if (idx < HEAD_DIM) {
            O_base[query_row * head_dim + idx] = acc_local[e] * inv_l;
        }
    }
}

// ---------------------------------------------------------------------------
// Host-side launch helper
// ---------------------------------------------------------------------------

hipError_t launch_flash_attn_hip_v1(
    const float* Q, const float* K, const float* V, float* O,
    int batch_size, int num_heads, int seq_len, int head_dim,
    hipStream_t stream = 0
) {
    int64_t head_stride  = static_cast<int64_t>(seq_len) * head_dim;
    int64_t batch_stride = static_cast<int64_t>(num_heads) * head_stride;
    float softmax_scale  = 1.0f / sqrtf(static_cast<float>(head_dim));

    dim3 grid(seq_len, num_heads, batch_size);
    dim3 block(WF_SIZE, 1, 1);

    // Dispatch based on head_dim.
    // ELEMS_PER_THREAD = HEAD_DIM / WF_SIZE  (round up to 1 minimum)
    if (head_dim <= 32) {
        // WF=64: 32/64 < 1 → each thread covers 0 or 1 elem; use EPT=1 with guard
        flash_attn_hip_v1_impl<32, 1><<<grid, block, 0, stream>>>(
            Q, K, V, O, seq_len, head_dim, batch_stride, head_stride, softmax_scale);
    } else if (head_dim <= 64) {
#if WF_SIZE == 64
        flash_attn_hip_v1_impl<64, 1><<<grid, block, 0, stream>>>(
            Q, K, V, O, seq_len, head_dim, batch_stride, head_stride, softmax_scale);
#else
        flash_attn_hip_v1_impl<64, 2><<<grid, block, 0, stream>>>(
            Q, K, V, O, seq_len, head_dim, batch_stride, head_stride, softmax_scale);
#endif
    } else if (head_dim <= 128) {
#if WF_SIZE == 64
        flash_attn_hip_v1_impl<128, 2><<<grid, block, 0, stream>>>(
            Q, K, V, O, seq_len, head_dim, batch_stride, head_stride, softmax_scale);
#else
        flash_attn_hip_v1_impl<128, 4><<<grid, block, 0, stream>>>(
            Q, K, V, O, seq_len, head_dim, batch_stride, head_stride, softmax_scale);
#endif
    } else {
        fprintf(stderr, "HIP kernel v1 supports head_dim <= 128, got %d\n", head_dim);
        return hipErrorInvalidValue;
    }

    return hipGetLastError();
}

// ---------------------------------------------------------------------------
// Minimal self-contained test (compiled when HIP_SELFTEST is defined)
// hipcc -DHIP_SELFTEST -O2 kernels/kernel_01_naive.hip -o flash_hip_test
// ---------------------------------------------------------------------------
#ifdef HIP_SELFTEST

#include <cstdlib>
#include <cstring>
#include <cassert>

// CPU reference: naive attention (float32)
static void cpu_attention_ref(
    const float* Q, const float* K, const float* V, float* O,
    int B, int H, int S, int D
) {
    float scale = 1.0f / sqrtf((float)D);
    for (int b = 0; b < B; b++)
    for (int h = 0; h < H; h++)
    for (int i = 0; i < S; i++) {
        // Compute attention scores and running softmax
        float m = -1e30f, l = 0.0f;
        float* acc = new float[D]();

        for (int j = 0; j < S; j++) {
            float dot = 0.0f;
            for (int d = 0; d < D; d++) {
                int qi = ((b*H + h)*S + i)*D + d;
                int kj = ((b*H + h)*S + j)*D + d;
                dot += Q[qi] * K[kj] * scale;
            }
            float m_new = fmaxf(m, dot);
            float alpha = expf(m - m_new);
            float beta  = expf(dot - m_new);
            l = alpha * l + beta;
            for (int d = 0; d < D; d++) {
                int vj = ((b*H + h)*S + j)*D + d;
                acc[d] = alpha * acc[d] + beta * V[vj];
            }
            m = m_new;
        }
        for (int d = 0; d < D; d++) {
            int oi = ((b*H + h)*S + i)*D + d;
            O[oi] = acc[d] / l;
        }
        delete[] acc;
    }
}

int main() {
    const int B = 1, H = 1, S = 128, D = 64;
    size_t sz = (size_t)B * H * S * D * sizeof(float);

    float* hQ = new float[B*H*S*D];
    float* hK = new float[B*H*S*D];
    float* hV = new float[B*H*S*D];
    float* hO = new float[B*H*S*D]();
    float* hRef = new float[B*H*S*D]();

    // Random init
    srand(42);
    for (int i = 0; i < B*H*S*D; i++) {
        hQ[i] = (float)rand()/RAND_MAX - 0.5f;
        hK[i] = (float)rand()/RAND_MAX - 0.5f;
        hV[i] = (float)rand()/RAND_MAX - 0.5f;
    }

    // CPU reference
    cpu_attention_ref(hQ, hK, hV, hRef, B, H, S, D);

    // GPU path
    float *dQ, *dK, *dV, *dO;
    hipMalloc(&dQ, sz); hipMalloc(&dK, sz);
    hipMalloc(&dV, sz); hipMalloc(&dO, sz);
    hipMemcpy(dQ, hQ, sz, hipMemcpyHostToDevice);
    hipMemcpy(dK, hK, sz, hipMemcpyHostToDevice);
    hipMemcpy(dV, hV, sz, hipMemcpyHostToDevice);
    hipMemset(dO, 0, sz);

    hipError_t err = launch_flash_attn_hip_v1(dQ, dK, dV, dO, B, H, S, D);
    hipDeviceSynchronize();

    if (err != hipSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", hipGetErrorString(err));
        return 1;
    }

    hipMemcpy(hO, dO, sz, hipMemcpyDeviceToHost);

    // Verify
    float max_diff = 0.0f, mean_diff = 0.0f;
    int n = B*H*S*D;
    for (int i = 0; i < n; i++) {
        float d = fabsf(hO[i] - hRef[i]);
        max_diff  = fmaxf(max_diff, d);
        mean_diff += d;
    }
    mean_diff /= n;

    printf("HIP Flash Attention v1 self-test (B=%d H=%d S=%d D=%d)\n", B, H, S, D);
    printf("  max_diff  = %.2e  (target < 1e-4)\n", max_diff);
    printf("  mean_diff = %.2e\n", mean_diff);
    if (max_diff < 1e-4f) {
        printf("  PASS ✓\n");
    } else {
        printf("  FAIL ✗\n");
        return 1;
    }

    hipFree(dQ); hipFree(dK); hipFree(dV); hipFree(dO);
    delete[] hQ; delete[] hK; delete[] hV; delete[] hO; delete[] hRef;
    return 0;
}

#endif // HIP_SELFTEST
