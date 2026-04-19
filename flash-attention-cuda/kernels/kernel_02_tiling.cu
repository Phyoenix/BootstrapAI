/**
 * Kernel 02: Tiled Flash Attention (shared memory optimization)
 *
 * Architecture: 32 threads per block (one warp)
 * Grid: (seq_len, num_heads, batch_size)
 * Block: (32, 1, 1)
 *
 * Changes from kernel v1:
 * - K and V are loaded into shared memory tiles (TILE=32 rows each)
 * - Dot products and softmax use shared memory reads instead of global memory
 * - Online softmax remains identical to kernel v1
 *
 * Reference: https://lubits.ch/flash/Part-4
 * Paper: FlashAttention-2 (arXiv:2307.08691)
 */

#include "flash_attention.h"
#include "utils.cuh"

#define NEG_INF (-1e30f)
#define TILE 32

// ============================================================================
// Warp-level sum reduction (XOR shuffle, 32 threads)
__device__ __forceinline__ float warp_reduce_sum_s(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// ============================================================================
// Kernel implementation
// Each thread t handles elements d = tid + e * WARP_SIZE (same as kernel v1)
//
// Shared memory layout:
//   s_K: [TILE * HEAD_DIM] — K tile, row-major: [row * HEAD_DIM + d]
//   s_V: [TILE * HEAD_DIM] — V tile, row-major: [row * HEAD_DIM + d]
//
// For HEAD_DIM=64: s_K[32*64] = 8192 floats = 32KB per tile
// For HEAD_DIM=128: s_K[32*128] = 16384 floats = 64KB per tile (exceeds limit)
// ============================================================================

template <int HEAD_DIM, int ELEMS_PER_THREAD>
__global__ void flash_attn_kernel_v2_impl(
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
    const int tid = threadIdx.x;  // 0..31

    const int64_t base_offset = batch_idx * batch_stride + head_idx * head_stride;
    const float* Q_base = Q + base_offset;
    const float* K_base = K + base_offset;
    const float* V_base = V + base_offset;
    float* O_base = O + base_offset;

    // Shared memory: K tile + V tile
    // Each tile: TILE rows x HEAD_DIM cols
    // For HEAD_DIM=64:  2 x 32 x 64 x 4 = 16 KB  OK
    // For HEAD_DIM=128: 2 x 32 x 128 x 4 = 32 KB  OK (within 48 KB)
    extern __shared__ float smem_[];
    float* s_K = smem_;
    float* s_V = smem_ + TILE * HEAD_DIM;

    // ---- Load Q_i into registers (same as kernel v1) ----
    // Thread t handles d = tid, tid+32, tid+64, ...
    const float* Q_i = Q_base + query_row * head_dim;
    float q_reg[ELEMS_PER_THREAD];
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) {
        int idx = tid + e * 32;
        q_reg[e] = (idx < HEAD_DIM) ? Q_i[idx] : 0.0f;
    }

    // ---- Online softmax accumulators (same as kernel v1) ----
    float m_i = NEG_INF;
    float l_i = 0.0f;
    float acc[ELEMS_PER_THREAD];
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) {
        acc[e] = 0.0f;
    }

    // ---- Iterate over KV rows in tiles ----
    for (int tile = 0; tile < seq_len; tile += TILE) {
        const int kvs = tile;
        const int kve = min(tile + TILE, seq_len);
        const int tr  = kve - kvs;

        // Load K tile: each thread t loads all elements of K[kvs+r, tid+e*32]
        for (int r = 0; r < tr; r++) {
            int kv_row = kvs + r;
            const float* K_j = K_base + kv_row * head_dim;
            float* K_tile_row = s_K + r * HEAD_DIM;
            #pragma unroll
            for (int e = 0; e < ELEMS_PER_THREAD; e++) {
                int idx = tid + e * 32;
                if (idx < HEAD_DIM) {
                    K_tile_row[idx] = K_j[idx];
                }
            }
        }

        // Load V tile: same pattern as K
        for (int r = 0; r < tr; r++) {
            int kv_row = kvs + r;
            const float* V_j = V_base + kv_row * head_dim;
            float* V_tile_row = s_V + r * HEAD_DIM;
            #pragma unroll
            for (int e = 0; e < ELEMS_PER_THREAD; e++) {
                int idx = tid + e * 32;
                if (idx < HEAD_DIM) {
                    V_tile_row[idx] = V_j[idx];
                }
            }
        }

        __syncthreads();

        // Compute attention for rows in this tile (same as kernel v1, from smem)
        for (int r = 0; r < tr; r++) {
            float* K_tile_row = s_K + r * HEAD_DIM;
            float* V_tile_row = s_V + r * HEAD_DIM;

            float dot = 0.0f;
            #pragma unroll
            for (int e = 0; e < ELEMS_PER_THREAD; e++) {
                int idx = tid + e * 32;
                if (idx < HEAD_DIM) {
                    dot += q_reg[e] * K_tile_row[idx];
                }
            }
            float s_ij = warp_reduce_sum_s(dot);
            s_ij *= softmax_scale;

            float m_new = fmaxf(m_i, s_ij);
            float alpha = expf(m_i - m_new);
            float beta  = expf(s_ij - m_new);
            float l_new = alpha * l_i + beta;

            #pragma unroll
            for (int e = 0; e < ELEMS_PER_THREAD; e++) {
                int idx = tid + e * 32;
                float v_val = (idx < HEAD_DIM) ? V_tile_row[idx] : 0.0f;
                acc[e] = alpha * acc[e] + beta * v_val;
            }

            m_i = m_new;
            l_i = l_new;
        }

        __syncthreads();
    }

    // ---- Final normalization (same as kernel v1) ----
    float inv_l = 1.0f / l_i;
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) {
        int idx = tid + e * 32;
        if (idx < HEAD_DIM) {
            O_base[query_row * head_dim + idx] = acc[e] * inv_l;
        }
    }
}

// ============================================================================
// Dispatch stub
// ============================================================================

__global__ void flash_attn_kernel_v2(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int seq_len,
    int head_dim,
    int64_t batch_stride,
    int64_t head_stride,
    float softmax_scale
) {}

// ============================================================================
// Host launch helper
// ============================================================================

cudaError_t launch_flash_attn_v2(
    const float* Q, const float* K, const float* V, float* O,
    int batch_size, int num_heads, int seq_len, int head_dim,
    cudaStream_t stream
) {
    int64_t head_stride  = static_cast<int64_t>(seq_len) * head_dim;
    int64_t batch_stride = static_cast<int64_t>(num_heads) * head_stride;
    float softmax_scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    dim3 grid(seq_len, num_heads, batch_size);
    dim3 block(32, 1, 1);

    size_t smem = 2 * TILE * head_dim * sizeof(float);

    if (head_dim <= 32) {
        flash_attn_kernel_v2_impl<32, 1><<<grid, block, smem, stream>>>(
            Q, K, V, O, seq_len, head_dim, batch_stride, head_stride, softmax_scale
        );
    } else if (head_dim <= 64) {
        flash_attn_kernel_v2_impl<64, 2><<<grid, block, smem, stream>>>(
            Q, K, V, O, seq_len, head_dim, batch_stride, head_stride, softmax_scale
        );
    } else if (head_dim <= 128) {
        flash_attn_kernel_v2_impl<128, 4><<<grid, block, smem, stream>>>(
            Q, K, V, O, seq_len, head_dim, batch_stride, head_stride, softmax_scale
        );
    } else {
        return cudaErrorInvalidValue;
    }

    return cudaGetLastError();
}
