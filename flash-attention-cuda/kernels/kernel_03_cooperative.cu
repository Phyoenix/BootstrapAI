/**
 * Kernel 03: Cooperative Loading Flash Attention
 *
 * Key innovation: Multiple queries share the same K/V tile.
 * This is the fix for Kernel 2's performance regression.
 *
 * Architecture:
 * - Grid:  (seq_len/QUERIES_PER_BLOCK, num_heads, batch_size)
 * - Block: (WARP_SIZE, QUERIES_PER_BLOCK, 1) = (32, 8, 1) = 256 threads
 *
 * Each block processes QUERIES_PER_BLOCK query rows (e.g., 8 queries).
 * All warps in the block cooperate to load K/V tiles into shared memory.
 * Each warp then computes attention for its assigned query using the shared tile.
 *
 * Memory traffic reduction:
 * - Kernel 1/2: Each query loads all K/V rows from HBM
 * - Kernel 3:   8 queries share each K/V tile load
 *               HBM traffic reduced by 8x for K/V reads
 *
 * Expected performance: 2x+ speedup over Kernel 1 for large seq_len
 *
 * Reference: FlashAttention-2 paper, Section 3.2 "Tiling and Memory-Efficient Attention"
 * Paper: FlashAttention-2 (arXiv:2307.08691)
 */

#include "flash_attention.h"
#include "utils.cuh"

#define NEG_INF (-1e30f)
#define WARP_SIZE 32
#define QUERIES_PER_BLOCK 8  // Number of queries processed per block

// ============================================================================
// Warp-level reduction primitives
// ============================================================================

__device__ __forceinline__ float warp_reduce_sum_c(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max_c(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// ============================================================================
// Cooperative Loading Kernel
// ============================================================================

template <int HEAD_DIM, int ELEMS_PER_THREAD>
__global__ void flash_attn_kernel_v3_impl(
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
    const int tid = threadIdx.x;           // 0..31 (lane within warp)
    const int warp_id = threadIdx.y;       // 0..QUERIES_PER_BLOCK-1 (which query in block)
    const int query_row = blockIdx.x * QUERIES_PER_BLOCK + warp_id;
    const int head_idx = blockIdx.y;
    const int batch_idx = blockIdx.z;

    // Bounds check
    if (query_row >= seq_len) return;

    const int64_t base_offset = batch_idx * batch_stride + head_idx * head_stride;
    const float* Q_base = Q + base_offset;
    const float* K_base = K + base_offset;
    const float* V_base = V + base_offset;
    float* O_base = O + base_offset;

    // Each warp loads its own Q row into registers
    const float* Q_row = Q_base + query_row * head_dim;
    float q_reg[ELEMS_PER_THREAD];
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) {
        int idx = tid + e * WARP_SIZE;
        q_reg[e] = (idx < HEAD_DIM) ? Q_row[idx] : 0.0f;
    }

    // Online softmax accumulators (per query)
    float m_i = NEG_INF;   // running max
    float l_i = 0.0f;      // running sum of exp(S - m)
    float acc[ELEMS_PER_THREAD];
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) {
        acc[e] = 0.0f;
    }

    // Shared memory for cooperative K/V tile loading
    // Layout: [QUERIES_PER_BLOCK][HEAD_DIM] for both K and V
    // Each warp contributes to loading part of the tile
    extern __shared__ float smem_[];
    float* s_K = smem_;
    float* s_V = smem_ + QUERIES_PER_BLOCK * HEAD_DIM;

    // Iterate over KV rows in tiles of QUERIES_PER_BLOCK
    for (int tile_start = 0; tile_start < seq_len; tile_start += QUERIES_PER_BLOCK) {
        const int tile_end = min(tile_start + QUERIES_PER_BLOCK, seq_len);
        const int tile_rows = tile_end - tile_start;

        // =====================================================================
        // COOPERATIVE LOADING: All warps load K/V tile together
        // =====================================================================
        // Each thread loads elements for all rows in the tile
        // Thread (tid, warp_id) loads: K[tile_start + row][tid + e*32] for all rows
        
        // Load K tile
        for (int r = 0; r < tile_rows; r++) {
            int kv_row = tile_start + r;
            const float* K_row = K_base + kv_row * head_dim;
            float* K_tile_row = s_K + r * HEAD_DIM;
            
            #pragma unroll
            for (int e = 0; e < ELEMS_PER_THREAD; e++) {
                int idx = tid + e * WARP_SIZE;
                if (idx < HEAD_DIM) {
                    K_tile_row[idx] = K_row[idx];
                }
            }
        }

        // Load V tile
        for (int r = 0; r < tile_rows; r++) {
            int kv_row = tile_start + r;
            const float* V_row = V_base + kv_row * head_dim;
            float* V_tile_row = s_V + r * HEAD_DIM;
            
            #pragma unroll
            for (int e = 0; e < ELEMS_PER_THREAD; e++) {
                int idx = tid + e * WARP_SIZE;
                if (idx < HEAD_DIM) {
                    V_tile_row[idx] = V_row[idx];
                }
            }
        }

        __syncthreads();

        // =====================================================================
        // COMPUTE: Each warp processes its assigned query using shared tile
        // =====================================================================
        for (int r = 0; r < tile_rows; r++) {
            const float* K_tile_row = s_K + r * HEAD_DIM;
            const float* V_tile_row = s_V + r * HEAD_DIM;

            // Compute dot product Q[query_row] . K[kv_row]
            float dot = 0.0f;
            #pragma unroll
            for (int e = 0; e < ELEMS_PER_THREAD; e++) {
                int idx = tid + e * WARP_SIZE;
                if (idx < HEAD_DIM) {
                    dot += q_reg[e] * K_tile_row[idx];
                }
            }
            float s_ij = warp_reduce_sum_c(dot);
            s_ij *= softmax_scale;

            // Online softmax update
            float m_new = fmaxf(m_i, s_ij);
            float alpha = expf(m_i - m_new);
            float beta = expf(s_ij - m_new);
            float l_new = alpha * l_i + beta;

            // Update output accumulator
            #pragma unroll
            for (int e = 0; e < ELEMS_PER_THREAD; e++) {
                int idx = tid + e * WARP_SIZE;
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
        int idx = tid + e * WARP_SIZE;
        if (idx < HEAD_DIM) {
            O_row[idx] = acc[e] * inv_l;
        }
    }
}

// ============================================================================
// Dispatch stub
// ============================================================================

__global__ void flash_attn_kernel_v3(
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
    // Dispatch stub -- use launch_flash_attn_v3() instead.
}

// ============================================================================
// Host launch helper
// ============================================================================

cudaError_t launch_flash_attn_v3(
    const float* Q, const float* K, const float* V, float* O,
    int batch_size, int num_heads, int seq_len, int head_dim,
    cudaStream_t stream
) {
    int64_t head_stride = static_cast<int64_t>(seq_len) * head_dim;
    int64_t batch_stride = static_cast<int64_t>(num_heads) * head_stride;
    float softmax_scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    // Grid: each block processes QUERIES_PER_BLOCK query rows
    int num_query_blocks = (seq_len + QUERIES_PER_BLOCK - 1) / QUERIES_PER_BLOCK;
    dim3 grid(num_query_blocks, num_heads, batch_size);
    dim3 block(WARP_SIZE, QUERIES_PER_BLOCK, 1);  // 32 * 8 = 256 threads

    // Shared memory: K tile + V tile, each [QUERIES_PER_BLOCK][HEAD_DIM]
    size_t smem_size = 2 * QUERIES_PER_BLOCK * head_dim * sizeof(float);

    // Check shared memory limit
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (smem_size > prop.sharedMemPerBlock) {
        fprintf(stderr, "Kernel v3 requires %zu bytes shared memory, but device only has %zu\n",
                smem_size, prop.sharedMemPerBlock);
        return cudaErrorInvalidValue;
    }

    if (head_dim <= 32) {
        flash_attn_kernel_v3_impl<32, 1><<<grid, block, smem_size, stream>>>(
            Q, K, V, O, seq_len, head_dim, batch_stride, head_stride, softmax_scale
        );
    } else if (head_dim <= 64) {
        flash_attn_kernel_v3_impl<64, 2><<<grid, block, smem_size, stream>>>(
            Q, K, V, O, seq_len, head_dim, batch_stride, head_stride, softmax_scale
        );
    } else if (head_dim <= 128) {
        flash_attn_kernel_v3_impl<128, 4><<<grid, block, smem_size, stream>>>(
            Q, K, V, O, seq_len, head_dim, batch_stride, head_stride, softmax_scale
        );
    } else {
        fprintf(stderr, "Kernel v3 only supports head_dim <= 128, got %d\n", head_dim);
        return cudaErrorInvalidValue;
    }

    return cudaGetLastError();
}
