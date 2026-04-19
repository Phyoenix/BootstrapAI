/**
 * Kernel 01: Naive Flash Attention
 *
 * First implementation -- correctness over performance.
 * Uses global memory only (no shared memory optimization).
 * Implements the online softmax algorithm from the Flash Attention paper.
 *
 * Algorithm (per query row i, iterating over all KV rows j):
 *   Initialize: m_i = -inf, l_i = 0, acc_i = 0
 *   For each KV row j:
 *     s_ij = Q_i . K_j * scale
 *     m_new = max(m_i, s_ij)
 *     alpha = exp(m_i - m_new)
 *     beta  = exp(s_ij - m_new)
 *     l_new = alpha * l_i + beta
 *     acc_i = alpha * acc_i + beta * V_j
 *     m_i = m_new, l_i = l_new
 *   O_i = acc_i / l_i
 *
 * Reference: https://lubits.ch/flash/Part-3
 * Paper: FlashAttention-2 (arXiv:2307.08691), Algorithm 1
 *
 * Grid:  (seq_len, num_heads, batch_size)
 * Block: (WARP_SIZE, 1, 1) = (32, 1, 1)
 *
 * Strategy: Each thread block = 1 warp = 32 threads.
 * For HEAD_DIM=64: each thread handles 2 elements of Q/K/V.
 * For HEAD_DIM=128: each thread handles 4 elements of Q/K/V.
 * Dot product is computed per-thread (partial), then warp-reduced.
 */

#include "flash_attention.h"
#include <cstdio>

#define NEG_INF (-1e30f)
#define WARP_SIZE 32

// ============================================================================
// Warp-level reduction primitives
// ============================================================================

__device__ __forceinline__ float warp_reduce_sum_v(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max_v(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// ============================================================================
// Template kernel implementation
// ============================================================================

template <int HEAD_DIM, int ELEMS_PER_THREAD>
__global__ void flash_attn_kernel_v1_impl(
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
    const int tid = threadIdx.x;  // 0..31 for a single warp

    const int64_t base_offset = batch_idx * batch_stride + head_idx * head_stride;
    const float* Q_base = Q + base_offset;
    const float* K_base = K + base_offset;
    const float* V_base = V + base_offset;
    float* O_base = O + base_offset;

    const float* Q_i = Q_base + query_row * head_dim;

    // ========================================================================
    // Load Q_i into registers
    // Thread t handles elements: t, t+32, t+64, t+96, ...
    // For HEAD_DIM=64, ELEMS_PER_THREAD=2: thread t loads Q[t], Q[t+32]
    // For HEAD_DIM=128, ELEMS_PER_THREAD=4: thread t loads Q[t], Q[t+32], Q[t+64], Q[t+96]
    // ========================================================================
    float q_local[ELEMS_PER_THREAD];

    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) {
        int idx = tid + e * WARP_SIZE;
        q_local[e] = (idx < HEAD_DIM) ? Q_i[idx] : 0.0f;
    }

    // ========================================================================
    // Online softmax accumulation
    // ========================================================================
    float m_i = NEG_INF;   // running max
    float l_i = 0.0f;      // running sum of exp(S - m)

    // Output accumulator (unnormalized)
    float acc_local[ELEMS_PER_THREAD];
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) {
        acc_local[e] = 0.0f;
    }

    // ========================================================================
    // Main loop: iterate over all key/value rows
    // ========================================================================
    for (int kv_row = 0; kv_row < seq_len; kv_row++) {
        const float* K_j = K_base + kv_row * head_dim;
        const float* V_j = V_base + kv_row * head_dim;

        // ------------------------------------------------------------------
        // Step 1: Compute S_ij = Q_i . K_j * scale
        // Each thread computes a partial dot product over its elements,
        // then we warp-reduce to get the full dot product.
        // ------------------------------------------------------------------
        float dot = 0.0f;
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_THREAD; e++) {
            int idx = tid + e * WARP_SIZE;
            if (idx < HEAD_DIM) {
                dot += q_local[e] * K_j[idx];
            }
        }

        // Warp-level sum reduction
        float s_ij = warp_reduce_sum_v(dot);
        s_ij *= softmax_scale;

        // ------------------------------------------------------------------
        // Step 2: Online softmax update
        // ------------------------------------------------------------------
        float m_new = fmaxf(m_i, s_ij);

        float alpha = expf(m_i - m_new);    // rescale old accumulators
        float beta  = expf(s_ij - m_new);   // weight for new V contribution

        // Update l
        float l_new = alpha * l_i + beta;

        // Update output accumulator: rescale old + add new V contribution
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_THREAD; e++) {
            int idx = tid + e * WARP_SIZE;
            float v_val = (idx < HEAD_DIM) ? V_j[idx] : 0.0f;
            acc_local[e] = alpha * acc_local[e] + beta * v_val;
        }

        m_i = m_new;
        l_i = l_new;
    }

    // ========================================================================
    // Final normalization: O_i = acc_i / l_i
    // ========================================================================
    float inv_l = 1.0f / l_i;
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) {
        int idx = tid + e * WARP_SIZE;
        if (idx < HEAD_DIM) {
            O_base[query_row * head_dim + idx] = acc_local[e] * inv_l;
        }
    }
}

// ============================================================================
// Non-template wrapper (do not call directly -- use launch_flash_attn_v1)
// ============================================================================

__global__ void flash_attn_kernel_v1(
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
    // Dispatch stub -- use launch_flash_attn_v1() instead.
}

// ============================================================================
// Host-side launch helper
// ============================================================================

cudaError_t launch_flash_attn_v1(
    const float* Q, const float* K, const float* V, float* O,
    int batch_size, int num_heads, int seq_len, int head_dim,
    cudaStream_t stream
) {
    int64_t head_stride  = static_cast<int64_t>(seq_len) * head_dim;
    int64_t batch_stride = static_cast<int64_t>(num_heads) * head_stride;

    float softmax_scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    dim3 grid(seq_len, num_heads, batch_size);
    dim3 block(WARP_SIZE, 1, 1);  // Single warp per query row

    if (head_dim <= 32) {
        flash_attn_kernel_v1_impl<32, 1><<<grid, block, 0, stream>>>(
            Q, K, V, O, seq_len, head_dim, batch_stride, head_stride, softmax_scale
        );
    } else if (head_dim <= 64) {
        flash_attn_kernel_v1_impl<64, 2><<<grid, block, 0, stream>>>(
            Q, K, V, O, seq_len, head_dim, batch_stride, head_stride, softmax_scale
        );
    } else if (head_dim <= 128) {
        flash_attn_kernel_v1_impl<128, 4><<<grid, block, 0, stream>>>(
            Q, K, V, O, seq_len, head_dim, batch_stride, head_stride, softmax_scale
        );
    } else {
        fprintf(stderr, "Kernel v1 only supports head_dim <= 128, got %d\n", head_dim);
        return cudaErrorInvalidValue;
    }

    return cudaGetLastError();
}
