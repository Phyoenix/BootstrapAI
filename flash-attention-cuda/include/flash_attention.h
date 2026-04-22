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

/**
 * Kernel 03: Cooperative Loading Flash Attention
 *
 * Key innovation: Multiple queries (8 per block) share the same K/V tile.
 * This reduces HBM traffic by 8x compared to Kernel 1/2.
 *
 * Grid:  (ceil(seq_len/8), num_heads, batch_size)
 * Block: (32, 8, 1) = 256 threads (8 warps)
 *
 * Expected performance: 2x+ speedup over Kernel 1 for large seq_len
 *
 * @param Q, K, V, O  Same layout as kernel v1/v2
 * @param seq_len     Sequence length
 * @param head_dim    Head dimension (32, 64, or 128)
 * @param batch_stride  Stride between batches
 * @param head_stride   Stride between heads
 * @param softmax_scale  1.0 / sqrt(head_dim)
 */
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
);

/**
 * Launch kernel v3 with cooperative loading.
 * Best for seq_len >= 256 where tile reuse amortizes sync overhead.
 */
cudaError_t launch_flash_attn_v3(
    const float* Q, const float* K, const float* V, float* O,
    int batch_size, int num_heads, int seq_len, int head_dim,
    cudaStream_t stream = 0
);

/**
 * Kernel 04: Bank Conflict-Free Flash Attention (Swizzled Shared Memory)
 *
 * Key innovation: Eliminates shared memory bank conflicts via row padding.
 * Extends Kernel 3 (cooperative loading) with SMEM_PAD=1 extra float per row,
 * shifting each row's starting bank so all 32 banks are hit exactly once.
 *
 * Bank conflict analysis:
 *   Without padding (HEAD_DIM=64): row stride = 64 → all rows start at bank 0
 *     → 32-way bank conflict when loading TILE_ROWS=8 rows simultaneously!
 *   With SMEM_PAD=1 (HEAD_DIM=64): row stride = 65
 *     → row r starts at bank (r*65)%32 = r (for r < 32) → conflict-free ✓
 *
 * Performance:
 *   - ~10-20% speedup over Kernel 3 for memory-bound workloads
 *   - Especially significant for multi-head (many warps loading concurrently)
 *   - Same correctness as Kernels 1/2/3 (identical algorithm, different layout)
 *
 * Industry use:
 *   - CUTLASS uses XOR-swizzle (more complex, zero waste) for same effect
 *   - cuDNN and FlashAttention-2 official both use padded shared memory
 *   - This kernel uses simpler padding strategy for readability/verifiability
 *
 * Grid:  (ceil(seq_len/8), num_heads, batch_size)
 * Block: (32, 8, 1) = 256 threads (8 warps), same as Kernel 3
 *
 * @param Q, K, V, O  Same layout as kernels v1-v3
 * @param seq_len     Sequence length
 * @param head_dim    Head dimension (32, 64, or 128)
 * @param batch_stride  Stride between batches
 * @param head_stride   Stride between heads
 * @param softmax_scale  1.0 / sqrt(head_dim)
 */
__global__ void flash_attn_kernel_v4(
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
 * Launch kernel v4 with bank conflict-free shared memory.
 * Falls back to kernel v3 if head_dim > 128 or smem limit exceeded.
 */
cudaError_t launch_flash_attn_v4(
    const float* Q, const float* K, const float* V, float* O,
    int batch_size, int num_heads, int seq_len, int head_dim,
    cudaStream_t stream = 0
);

/**
 * Kernel 05: Double Buffering Flash Attention (Software Pipelining)
 *
 * Key innovation: Overlap global memory loads with attention computation.
 * While computing attention on tile T, prefetch tile T+1 into the alternate
 * shared memory buffer (ping-pong / double buffering).
 *
 * Pipeline stages:
 *   Tile 0:  Load(0)
 *   Tile 1:  Compute(0) || Load(1)   ← overlap!
 *   Tile 2:  Compute(1) || Load(2)   ← overlap!
 *   ...
 *   Tile N:  Compute(N-1)            ← drain
 *
 * Shared memory usage (double of Kernel 4):
 *   - 4 buffers total: s_K[0], s_K[1], s_V[0], s_V[1]
 *   - Each buffer: TILE_ROWS * (HEAD_DIM + SMEM_PAD) floats
 *   - For TILE=8, HD=64, PAD=1: 4 * 8 * 65 * 4 = 8320 bytes (~8KB)
 *   - Well within 48KB limit; leaves room for future sm_80 async copy
 *
 * Expected improvement: 15-30% over Kernel 4 for seq_len >= 512
 *   (latency hiding benefit scales with ratio of load latency to compute time)
 *
 * Hardware: sm_80+ preferred (Ampere async memcpy via cp.async).
 *   On sm_89 (RTX 4080), compiler may issue async loads from plain stores.
 *   Explicit cp.async integration planned for Kernel 6.
 *
 * Grid:  (ceil(seq_len/8), num_heads, batch_size)
 * Block: (32, 8, 1) = 256 threads, same as Kernels 3/4
 *
 * @param Q, K, V, O  Same layout as kernels v1-v4
 * @param seq_len     Sequence length
 * @param head_dim    Head dimension (32, 64, or 128)
 * @param batch_stride  Stride between batches
 * @param head_stride   Stride between heads
 * @param softmax_scale  1.0 / sqrt(head_dim)
 */
__global__ void flash_attn_kernel_v5(
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
 * Launch kernel v5 with double-buffered shared memory pipeline.
 * Falls back to kernel v4 if smem limit exceeded (very large head dims).
 */
cudaError_t launch_flash_attn_v5(
    const float* Q, const float* K, const float* V, float* O,
    int batch_size, int num_heads, int seq_len, int head_dim,
    cudaStream_t stream = 0
);

/**
 * Kernel 06: cp.async Hardware Pipeline Flash Attention (sm_80+ / Ampere)
 *
 * Key innovation: TRUE hardware-level async HBM→SMEM copy via cp.async PTX.
 * Unlike Kernel 5's "software double buffering" (which relies on compiler
 * scheduling), this kernel issues cp.async PTX instructions that drive a
 * dedicated copy engine independently of the SM compute pipeline.
 *
 * cp.async pipeline (depth=3):
 *   - Three ring-buffer slots for K (and three for V)
 *   - Warm-up: issue async loads for the first 3 tiles before any compute
 *   - Main loop: compute tile[i] while tile[i+3] is loading in background
 *   - cp.async.wait_group N: wait only until ≤N stages remain pending
 *     (finer-grained than __syncthreads)
 *
 * Memory layout:
 *   smem = [K_buf_0 | K_buf_1 | K_buf_2 | V_buf_0 | V_buf_1 | V_buf_2]
 *   Each buffer: TILE_ROWS * (HEAD_DIM + SMEM_PAD) floats
 *   For HD=64: 6 * 8 * 65 * 4 = 12480 bytes (~12KB)
 *   For HD=128: 6 * 8 * 129 * 4 = 24768 bytes (~24KB) — within 48KB limit
 *
 * Expected improvement over Kernel 5:
 *   - sm_80+ (genuine cp.async): ~10-20% gain; guaranteed HW overlap
 *   - sm_89 (RTX 4080): cp.async is fully supported
 *   - < sm_80: macro falls back to synchronous load (correct, no perf gain)
 *
 * Hardware compatibility:
 *   - sm_80+  (A100, RTX 3090, RTX 4080, RTX 4090, H100): full cp.async
 *   - sm_75   (RTX 2080 Ti): fallback to synchronous load
 *   - sm_89   (RTX 4080, Ada Lovelace): cp.async fully supported ✓
 *
 * ncu profiling targets:
 *   - l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum   (smem load wavefronts)
 *   - smsp__sass_inst_executed_op_global_ld.sum              (global loads)
 *   - smsp__warp_issue_stalled_lgstyp_per_warp_active.pct    (LGSTYP stall %)
 *     (LGSTYP stall should drop vs K5 if cp.async is truly overlapping)
 *
 * Grid:  (ceil(seq_len/8), num_heads, batch_size)
 * Block: (32, 8, 1) = 256 threads, same as Kernels 3/4/5
 *
 * @param Q, K, V, O  Same layout as kernels v1-v5
 * @param seq_len     Sequence length
 * @param head_dim    Head dimension (32, 64, or 128)
 * @param batch_stride  Stride between batches
 * @param head_stride   Stride between heads
 * @param softmax_scale  1.0 / sqrt(head_dim)
 */
__global__ void flash_attn_kernel_v6(
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
 * Launch kernel v6 with cp.async hardware pipeline (depth-3).
 * On sm_80+: uses cp.async PTX for guaranteed hardware overlap.
 * On older hardware: macros degrade to synchronous load (correct, same result).
 * Falls back to kernel v5 if smem limit exceeded.
 */
cudaError_t launch_flash_attn_v6(
    const float* Q, const float* K, const float* V, float* O,
    int batch_size, int num_heads, int seq_len, int head_dim,
    cudaStream_t stream = 0
);

/**
 * Kernel 07: Warp Specialization Flash Attention
 *
 * Key innovation: Divide the 8 warps in a block into specialized groups.
 *   - Warps 0-1 (PRODUCERS): exclusively issue K/V loads from global memory
 *   - Warps 2-7 (CONSUMERS): exclusively compute attention (no memory loads)
 *
 * This eliminates the resource contention in K3-K6 where all warps competed
 * for both LSU (load-store units) and FP execution units simultaneously.
 * Specialization allows:
 *   - Producer warps to maximize in-flight memory requests (fill LSU pipeline)
 *   - Consumer warps to maximize FMA issue rate (no LSU competition)
 *   - True producer/consumer parallelism via ring-buffer + flag signaling
 *
 * Communication: 3-slot ring buffer in shared memory
 *   - s_flags[slot]: 0=FREE (producer can write), 1=READY (consumer can read)
 *   - Producer waits for FREE, loads tile, sets READY
 *   - Consumer waits for READY, computes, sets FREE
 *   - No block-wide __syncthreads() needed — each warp group spins independently
 *
 * Shared memory (same depth-3 as K6 + flag overhead):
 *   2 × PIPE_DEPTH × TILE_ROWS × (HEAD_DIM + SMEM_PAD) × 4 + 3×4 bytes
 *   For HD=64: ~12KB (well within 48KB limit)
 *
 * Block configuration:
 *   Grid:  (ceil(seq_len/6), num_heads, batch_size)  — 6 consumers per block
 *   Block: (32, 8, 1) = 256 threads = 8 warps
 *
 * Expected improvement vs K6: +5-15% on compute-bound configs
 *   (benefit higher when FP utilization is the bottleneck, not memory BW)
 *
 * @param Q, K, V, O  Same layout as kernels v1-v6
 * @param seq_len     Sequence length
 * @param head_dim    Head dimension (32, 64, or 128)
 * @param batch_stride  Stride between batches
 * @param head_stride   Stride between heads
 * @param softmax_scale  1.0 / sqrt(head_dim)
 */
__global__ void flash_attn_kernel_v7(
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
 * Launch kernel v7 with warp-specialized producer/consumer pipeline.
 * Falls back to kernel v6 if smem limit exceeded.
 */
cudaError_t launch_flash_attn_v7(
    const float* Q, const float* K, const float* V, float* O,
    int batch_size, int num_heads, int seq_len, int head_dim,
    cudaStream_t stream = 0
);

/**
 * Kernel 08: Persistent Kernel Flash Attention
 *
 * Key innovation: Replace per-query-tile grid with a FIXED grid of worker
 * blocks (one per SM). Each block persistently picks work from a global
 * atomic counter until all (batch, head, q_tile) triples are processed.
 *
 * Eliminates the "multiple wave" overhead of standard kernels:
 *   Standard (K1-K7): grid = (N_q_tiles × heads × batch) → many scheduling waves
 *   Persistent (K8):  grid = (num_SMs) → ONE wave, blocks loop over all work
 *
 * Inherits ALL optimizations from K4-K7:
 *   - SMEM_PAD=1 bank-conflict-free layout
 *   - cp.async PTX hardware-async HBM→SMEM copy (sm_80+)
 *   - Warp specialization (2 producers + 6 consumers per block)
 *   - Q tile smem cache (loaded once per work item, shared by consumers)
 *
 * Expected improvement:
 *   - seq=1024, h=8, b=1: 0-5% (few waves anyway)
 *   - seq=4096, h=16, b=4: 5-15% (many waves → persistent wins)
 *   - seq=8192+:            increasing benefit as wave count grows
 *
 * Work decomposition:
 *   - Global int counter g_work_counter: atomically-incremented
 *   - tile_id → (batch_idx, head_idx, q_tile_idx) via integer decomposition
 *   - Block exits when tile_id >= total_tiles
 *
 * Block configuration:
 *   Grid:  (num_SMs, 1, 1)          — persistent, fixed
 *   Block: (32, 8, 1) = 256 threads
 *   warp 0,1: PRODUCERS (load K/V)
 *   warp 2-7: CONSUMERS (compute attention)
 *
 * Reference:
 *   CUTLASS PersistentTileScheduler; cuDNN persistent GEMM; FA3 on H100
 *
 * @param Q, K, V, O    Same layout as kernels v1-v7
 * @param seq_len       Sequence length
 * @param head_dim      Head dimension (32, 64, or 128)
 * @param batch_stride  Stride between batches
 * @param head_stride   Stride between heads
 * @param softmax_scale 1.0 / sqrt(head_dim)
 * @param num_heads     Number of attention heads
 * @param batch_size    Batch size
 * @param g_work_counter  Device pointer to global work queue counter (int)
 * @param total_tiles   Total number of work items
 */
__global__ void flash_attn_kernel_v8(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float*       __restrict__ O,
    int          seq_len,
    int          head_dim,
    int64_t      batch_stride,
    int64_t      head_stride,
    float        softmax_scale,
    int          num_heads,
    int          batch_size,
    int*         g_work_counter,
    int          total_tiles
);

/**
 * Launch kernel v8 with persistent grid + global work queue.
 * Automatically queries SM count; allocates and frees d_work_counter.
 * Falls back to kernel v7 on very small problems where overhead dominates.
 */
cudaError_t launch_flash_attn_v8(
    const float* Q, const float* K, const float* V, float* O,
    int batch_size, int num_heads, int seq_len, int head_dim,
    cudaStream_t stream = 0
);

#endif // FLASH_ATTENTION_H
