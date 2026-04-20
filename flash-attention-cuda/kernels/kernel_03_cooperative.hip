/**
 * Kernel 03: Cooperative Loading Flash Attention — HIP Port
 *
 * Ported from kernels/kernel_03_cooperative.cu (CUDA → HIP)
 *
 * Key innovations preserved from CUDA version:
 * - Multiple queries (QUERIES_PER_BLOCK) share the same K/V tile
 * - All warps cooperate to load K/V tiles into shared memory
 * - Online softmax with running max (m_i) and normalization (l_i)
 *
 * HIP-specific adaptations:
 * - __shfl_xor_sync(mask, val, offset) → __shfl_xor(val, offset)
 *   HIP shuffle instructions operate at the wavefront level without a mask
 * - WF_SIZE macro: AMD wavefront = 64 threads (vs NVIDIA warp = 32)
 *   This affects ELEMS_PER_THREAD: ceil(HEAD_DIM / WF_SIZE)
 * - cuda* → hip* API prefixes
 * - HIP_SELFTEST mode: compile with -DHIP_SELFTEST for standalone verification
 *
 * Build:
 *   hipcc -O3 --offload-arch=gfx1100 -o kernel03_hip kernel_03_cooperative.hip
 *   (Use gfx1100 for RX 7900, gfx90a for MI250)
 *
 * Reference: FlashAttention-2 (arXiv:2307.08691)
 * Contribution: WorkBuddy collab agent — 2026-04-21
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <hip/hip_runtime.h>

#define NEG_INF (-1e30f)

// ============================================================================
// Platform-specific wavefront size
// ============================================================================

#ifdef __HIP_PLATFORM_AMD__
    #define WF_SIZE 64  // AMD wavefront = 64 threads
#else
    #define WF_SIZE 32  // NVIDIA warp = 32 threads (fallback)
#endif

#define QUERIES_PER_BLOCK 8

// ============================================================================
// Warp/Wavefront-level reduction primitives
// ============================================================================

__device__ __forceinline__ float wf_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WF_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor(val, offset);
    }
    return val;
}

__device__ __forceinline__ float wf_reduce_max(float val) {
    #pragma unroll
    for (int offset = WF_SIZE / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor(val, offset));
    }
    return val;
}

// ============================================================================
// Cooperative Loading Kernel (HIP)
// ============================================================================

/**
 * Core kernel: Multiple queries per block share K/V tiles.
 *
 * Grid:  (ceil(seq_len / QUERIES_PER_BLOCK), num_heads, batch_size)
 * Block: (WF_SIZE, QUERIES_PER_BLOCK, 1) = 256 or 512 threads
 *
 * On AMD (WF_SIZE=64): Block = 64 * 8 = 512 threads
 * On NVIDIA (WF_SIZE=32): Block = 32 * 8 = 256 threads
 *
 * Each "warp" (wavefront on AMD) processes one query row.
 * All wavefronts cooperate to load K/V tiles into shared memory.
 */
template <int HEAD_DIM, int ELEMS_PER_THREAD>
__global__ void flash_attn_kernel_v3_hip_impl(
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
    // Thread indexing
    const int tid = threadIdx.x;           // 0..WF_SIZE-1 (lane within wavefront)
    const int wf_id = threadIdx.y;         // 0..QUERIES_PER_BLOCK-1 (which query)
    const int query_row = blockIdx.x * QUERIES_PER_BLOCK + wf_id;
    const int head_idx = blockIdx.y;
    const int batch_idx = blockIdx.z;

    // Bounds check
    if (query_row >= seq_len) return;

    const int64_t base_offset = batch_idx * batch_stride + head_idx * head_stride;
    const float* Q_base = Q + base_offset;
    const float* K_base = K + base_offset;
    const float* V_base = V + base_offset;
    float* O_base = O + base_offset;

    // Each wavefront loads its own Q row into registers
    const float* Q_row = Q_base + query_row * head_dim;
    float q_reg[ELEMS_PER_THREAD];
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) {
        int idx = tid + e * WF_SIZE;
        q_reg[e] = (idx < HEAD_DIM) ? Q_row[idx] : 0.0f;
    }

    // Online softmax accumulators
    float m_i = NEG_INF;
    float l_i = 0.0f;
    float acc[ELEMS_PER_THREAD];
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) {
        acc[e] = 0.0f;
    }

    // Shared memory for cooperative K/V tile loading
    // Layout: [QUERIES_PER_BLOCK][HEAD_DIM] for K and V
    extern __shared__ float smem_[];
    float* s_K = smem_;
    float* s_V = smem_ + QUERIES_PER_BLOCK * HEAD_DIM;

    // Iterate over KV rows in tiles
    for (int tile_start = 0; tile_start < seq_len; tile_start += QUERIES_PER_BLOCK) {
        const int tile_end = min(tile_start + QUERIES_PER_BLOCK, seq_len);
        const int tile_rows = tile_end - tile_start;

        // =================================================================
        // COOPERATIVE LOADING: All wavefronts load K/V tile together
        // =================================================================
        for (int r = 0; r < tile_rows; r++) {
            int kv_row = tile_start + r;
            const float* K_row = K_base + kv_row * head_dim;
            float* K_tile_row = s_K + r * HEAD_DIM;

            #pragma unroll
            for (int e = 0; e < ELEMS_PER_THREAD; e++) {
                int idx = tid + e * WF_SIZE;
                if (idx < HEAD_DIM) {
                    K_tile_row[idx] = K_row[idx];
                }
            }
        }

        for (int r = 0; r < tile_rows; r++) {
            int kv_row = tile_start + r;
            const float* V_row = V_base + kv_row * head_dim;
            float* V_tile_row = s_V + r * HEAD_DIM;

            #pragma unroll
            for (int e = 0; e < ELEMS_PER_THREAD; e++) {
                int idx = tid + e * WF_SIZE;
                if (idx < HEAD_DIM) {
                    V_tile_row[idx] = V_row[idx];
                }
            }
        }

        __syncthreads();

        // =================================================================
        // COMPUTE: Each wavefront processes its query using shared tile
        // =================================================================
        for (int r = 0; r < tile_rows; r++) {
            const float* K_tile_row = s_K + r * HEAD_DIM;
            const float* V_tile_row = s_V + r * HEAD_DIM;

            // Dot product: Q[query_row] . K[kv_row]
            float dot = 0.0f;
            #pragma unroll
            for (int e = 0; e < ELEMS_PER_THREAD; e++) {
                int idx = tid + e * WF_SIZE;
                if (idx < HEAD_DIM) {
                    dot += q_reg[e] * K_tile_row[idx];
                }
            }
            float s_ij = wf_reduce_sum(dot);
            s_ij *= softmax_scale;

            // Online softmax update
            float m_new = fmaxf(m_i, s_ij);
            float alpha = expf(m_i - m_new);
            float beta = expf(s_ij - m_new);
            float l_new = alpha * l_i + beta;

            // Update output accumulator
            #pragma unroll
            for (int e = 0; e < ELEMS_PER_THREAD; e++) {
                int idx = tid + e * WF_SIZE;
                float v_val = (idx < HEAD_DIM) ? V_tile_row[idx] : 0.0f;
                acc[e] = alpha * acc[e] + beta * v_val;
            }

            m_i = m_new;
            l_i = l_new;
        }

        __syncthreads();
    }

    // Final normalization and write output
    float inv_l = 1.0f / l_i;
    float* O_row = O_base + query_row * head_dim;
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) {
        int idx = tid + e * WF_SIZE;
        if (idx < HEAD_DIM) {
            O_row[idx] = acc[e] * inv_l;
        }
    }
}

// ============================================================================
// Dispatch stub (empty — use launch function)
// ============================================================================

__global__ void flash_attn_kernel_v3_hip(
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
    // Use launch_flash_attn_v3_hip() instead of calling this directly.
}

// ============================================================================
// Host launch helper
// ============================================================================

hipError_t launch_flash_attn_v3_hip(
    const float* Q, const float* K, const float* V, float* O,
    int batch_size, int num_heads, int seq_len, int head_dim,
    hipStream_t stream = 0
) {
    int64_t head_stride = static_cast<int64_t>(seq_len) * head_dim;
    int64_t batch_stride = static_cast<int64_t>(num_heads) * head_stride;
    float softmax_scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    int num_query_blocks = (seq_len + QUERIES_PER_BLOCK - 1) / QUERIES_PER_BLOCK;
    dim3 grid(num_query_blocks, num_heads, batch_size);
    dim3 block(WF_SIZE, QUERIES_PER_BLOCK, 1);

    size_t smem_size = 2 * QUERIES_PER_BLOCK * head_dim * sizeof(float);

    // Check shared memory limit
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    if (smem_size > prop.sharedMemPerBlock) {
        fprintf(stderr, "Kernel v3 HIP requires %zu bytes shared memory, device has %zu\n",
                smem_size, prop.sharedMemPerBlock);
        return hipErrorInvalidValue;
    }

    // Dispatch by head_dim — ELEMS_PER_THREAD = ceil(HEAD_DIM / WF_SIZE)
    // On AMD (WF=64): dim=64 → EPT=1, dim=128 → EPT=2
    // On NVIDIA (WF=32): dim=64 → EPT=2, dim=128 → EPT=4
    if (head_dim <= 32) {
        hipLaunchKernelGGL(
            flash_attn_kernel_v3_hip_impl<32, 1>,
            grid, block, smem_size, stream,
            Q, K, V, O, seq_len, head_dim, batch_stride, head_stride, softmax_scale
        );
    } else if (head_dim <= WF_SIZE) {
        hipLaunchKernelGGL(
            flash_attn_kernel_v3_hip_impl<WF_SIZE, 1>,
            grid, block, smem_size, stream,
            Q, K, V, O, seq_len, head_dim, batch_stride, head_stride, softmax_scale
        );
    } else if (head_dim <= 2 * WF_SIZE) {
        hipLaunchKernelGGL(
            flash_attn_kernel_v3_hip_impl<2 * WF_SIZE, 2>,
            grid, block, smem_size, stream,
            Q, K, V, O, seq_len, head_dim, batch_stride, head_stride, softmax_scale
        );
    } else {
        fprintf(stderr, "Kernel v3 HIP supports head_dim <= %d, got %d\n",
                2 * WF_SIZE, head_dim);
        return hipErrorInvalidValue;
    }

    return hipGetLastError();
}

// ============================================================================
// Self-test mode: verify correctness with CPU reference
// ============================================================================

#ifdef HIP_SELFTEST

#include <vector>
#include <algorithm>
#include <random>

// CPU reference: naive attention
void cpu_reference_attn(
    const float* Q, const float* K, const float* V,
    float* O_ref, int seq_len, int head_dim, float scale
) {
    for (int q = 0; q < seq_len; q++) {
        for (int d = 0; d < head_dim; d++) {
            float sum_val = 0.0f;
            float sum_exp = 0.0f;
            float max_score = -1e30f;

            // First pass: find max
            for (int k = 0; k < seq_len; k++) {
                float dot = 0.0f;
                for (int dd = 0; dd < head_dim; dd++) {
                    dot += Q[q * head_dim + dd] * K[k * head_dim + dd];
                }
                float score = dot * scale;
                if (score > max_score) max_score = score;
            }

            // Second pass: accumulate
            for (int k = 0; k < seq_len; k++) {
                float dot = 0.0f;
                for (int dd = 0; dd < head_dim; dd++) {
                    dot += Q[q * head_dim + dd] * K[k * head_dim + dd];
                }
                float score = dot * scale;
                float e = expf(score - max_score);
                sum_val += e * V[k * head_dim + d];
                sum_exp += e;
            }

            O_ref[q * head_dim + d] = sum_val / sum_exp;
        }
    }
}

int run_selftest(int seq_len, int head_dim) {
    printf("  Selftest: seq=%d, head_dim=%d (WF_SIZE=%d)\n", seq_len, head_dim, WF_SIZE);

    size_t bytes = (size_t)seq_len * head_dim * sizeof(float);

    // Allocate host memory
    std::vector<float> h_Q(seq_len * head_dim);
    std::vector<float> h_K(seq_len * head_dim);
    std::vector<float> h_V(seq_len * head_dim);
    std::vector<float> h_O(seq_len * head_dim);
    std::vector<float> h_O_ref(seq_len * head_dim);

    // Initialize with random data
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : h_Q) v = dist(rng);
    for (auto& v : h_K) v = dist(rng);
    for (auto& v : h_V) v = dist(rng);

    // Device allocations
    float *d_Q, *d_K, *d_V, *d_O;
    hipMalloc(&d_Q, bytes);
    hipMalloc(&d_K, bytes);
    hipMalloc(&d_V, bytes);
    hipMalloc(&d_O, bytes);

    hipMemcpy(d_Q, h_Q.data(), bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_K, h_K.data(), bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_V, h_V.data(), bytes, hipMemcpyHostToDevice);
    hipMemset(d_O, 0, bytes);

    // Launch kernel
    hipError_t err = launch_flash_attn_v3_hip(
        d_Q, d_K, d_V, d_O,
        /*batch=*/1, /*heads=*/1, seq_len, head_dim
    );
    if (err != hipSuccess) {
        printf("  FAIL: hip kernel launch error: %s\n", hipGetErrorString(err));
        return 1;
    }
    hipDeviceSynchronize();

    // Copy results back
    hipMemcpy(h_O.data(), d_O, bytes, hipMemcpyDeviceToHost);

    // CPU reference
    float scale = 1.0f / sqrtf((float)head_dim);
    cpu_reference_attn(h_Q.data(), h_K.data(), h_V.data(), h_O_ref.data(),
                       seq_len, head_dim, scale);

    // Compare
    float max_diff = 0.0f;
    float total_diff = 0.0f;
    int count = seq_len * head_dim;
    for (int i = 0; i < count; i++) {
        float diff = fabsf(h_O[i] - h_O_ref[i]);
        max_diff = fmaxf(max_diff, diff);
        total_diff += diff;
    }
    float mean_diff = total_diff / count;

    printf("    Max diff: %.2e, Mean diff: %.2e\n", max_diff, mean_diff);

    // Cleanup
    hipFree(d_Q); hipFree(d_K); hipFree(d_V); hipFree(d_O);

    float tolerance = 1e-3f;  // FP32 tolerance
    if (max_diff > tolerance) {
        printf("  FAIL: max_diff %.2e exceeds tolerance %.2e\n", max_diff, tolerance);
        return 1;
    }
    printf("  PASS\n");
    return 0;
}

int main() {
    printf("Kernel 03 Cooperative Loading — HIP Self-Test\n");
    printf("Wavefront size: %d\n", WF_SIZE);
    printf("Queries per block: %d\n\n", QUERIES_PER_BLOCK);

    int passed = 0, total = 0;

    // Test cases matching CUDA test_correctness.cu
    struct TestCase { int seq; int dim; };
    TestCase tests[] = {
        {64, 64}, {128, 64}, {256, 64}, {512, 64}, {1024, 64},
        {128, 32}, {64, 128},
    };

    for (auto& t : tests) {
        // Skip tests that exceed supported head_dim
        if (t.dim > 2 * WF_SIZE) {
            printf("  SKIP: seq=%d dim=%d (head_dim > 2*WF_SIZE=%d)\n",
                   t.seq, t.dim, 2 * WF_SIZE);
            continue;
        }
        total++;
        passed += (run_selftest(t.seq, t.dim) == 0) ? 1 : 0;
    }

    printf("\nResults: %d/%d tests passed\n", passed, total);
    return (passed == total) ? 0 : 1;
}

#endif // HIP_SELFTEST
