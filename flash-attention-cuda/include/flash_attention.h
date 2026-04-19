/**
 * Flash Attention CUDA Implementation
 * Header file for kernel declarations and shared utilities.
 *
 * Reference: https://lubits.ch/flash/ (Flash Attention from Scratch)
 * Paper: FlashAttention-2 (arXiv:2307.08691)
 */

#ifndef FLASH_ATTENTION_H
#define FLASH_ATTENTION_H

#include <cuda_runtime.h>
#include <cstdint>

// ============================================================================
// Kernel version declarations
// ============================================================================

/**
 * Kernel 01: Naive Flash Attention (correctness-first, no optimizations)
 *
 * Implements the online softmax algorithm with global memory only.
 * Each thread block processes one (batch, head, query_block) combination.
 *
 * Grid: (num_q_blocks, num_heads, batch_size)
 * Block: (BLOCK_SIZE, 1, 1)
 *
 * @param Q      Query matrix  [batch * heads * seq_len * head_dim]
 * @param K      Key matrix    [batch * heads * seq_len * head_dim]
 * @param V      Value matrix  [batch * heads * seq_len * head_dim]
 * @param O      Output matrix [batch * heads * seq_len * head_dim]
 * @param seq_len   Sequence length (must be divisible by BLOCK_SIZE)
 * @param head_dim  Head dimension
 * @param batch_stride  Stride between batches in Q/K/V/O
 * @param head_stride   Stride between heads
 * @param softmax_scale 1.0 / sqrt(head_dim)
 */
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
);

// ============================================================================
// Utility functions (host-side)
// ============================================================================

/**
 * Launch kernel v1 with appropriate grid/block configuration.
 */
cudaError_t launch_flash_attn_v1(
    const float* Q, const float* K, const float* V, float* O,
    int batch_size, int num_heads, int seq_len, int head_dim,
    cudaStream_t stream = 0
);

/**
 * Compute FLOPs for attention: 2 * batch * heads * seq^2 * head_dim
 */
inline double compute_attention_flops(
    int batch_size, int num_heads, int seq_len, int head_dim
) {
    return 2.0 * batch_size * num_heads * seq_len * seq_len * head_dim;
}

/**
 * Kernel 02: Tiled Flash Attention (shared memory optimization)
 *
 * Reduces HBM bandwidth by caching K/V tiles in shared memory.
 * See kernels/kernel_02_tiling.cu for full description.
 *
 * Grid: (seq_len, num_heads, batch_size)  — one block per query row
 * Block: (64, 1, 1)
 *
 * @param Q, K, V, O  Same layout as kernel v1
 * @param seq_len     Sequence length
 * @param head_dim    Head dimension (32, 64, or 128)
 * @param batch_stride  Stride between batches
 * @param head_stride   Stride between heads
 * @param softmax_scale  1.0 / sqrt(head_dim)
 */
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
);

/**
 * Launch kernel v2 with appropriate grid/block configuration.
 * Automatically falls back to kernel v1 if shared memory exceeds limit.
 */
cudaError_t launch_flash_attn_v2(
    const float* Q, const float* K, const float* V, float* O,
    int batch_size, int num_heads, int seq_len, int head_dim,
    cudaStream_t stream = 0
);

#endif // FLASH_ATTENTION_H
