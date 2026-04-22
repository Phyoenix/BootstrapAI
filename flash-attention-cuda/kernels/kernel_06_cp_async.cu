/**
 * Kernel 06: cp.async Flash Attention (Ampere+ Hardware Async Pipeline)
 *
 * Key innovation: True hardware-level async prefetch from HBM to shared memory
 * using Ampere's cp.async instruction (__cp_async_ca / __pipeline_memcpy_async).
 *
 * ============================================================================
 * Background: Why cp.async matters (vs Kernel 5 double buffering)
 * ============================================================================
 *
 * Kernel 5 used a "software pipeline" with plain global loads:
 *   s_K[next_buf][...] = K_base[global_row * HEAD_DIM + d];
 *
 * Problem: The load cannot truly overlap compute because:
 *   1. The SM must issue the load instruction AND wait for the data
 *   2. The compiler sees a control-flow dependency at __syncthreads()
 *   3. Without async instructions, the memory system stalls the SM
 *
 * cp.async instruction (introduced in sm_80 / Ampere):
 *   - Dispatches a DMA-like request to a DEDICATED async copy engine
 *   - The SM continues executing (compute) immediately after issuing cp.async
 *   - The copy engine writes directly HBM → shared memory (bypasses L1 regs)
 *   - __pipeline_wait_prior<N>() provides explicit fence without full __syncthreads
 *
 * Performance delta (theoretical):
 *   Kernel 5: overlap requires compiler cooperation (not guaranteed)
 *   Kernel 6: overlap is guaranteed by hardware
 *   Expected additional gain: ~10-20% over Kernel 5 for large sequences
 *
 * ============================================================================
 * Pipeline Architecture: Depth-3 Sliding Window
 * ============================================================================
 * We use a pipeline depth of 3 (3 shared memory buffer sets for K and V):
 *
 *   Iteration schedule (pipeline depth D=3, N tiles total):
 *     iter  action
 *      0    cp.async prefetch tiles [0,1,2]  (fill pipeline)
 *      1    wait_prior(D-1) for tile 0; compute tile 0; cp.async tile 3
 *      2    wait_prior(D-1) for tile 1; compute tile 1; cp.async tile 4
 *      ...
 *      N-1  wait_prior(D-1) for tile N-1; compute tile N-1
 *      N..N+D-1  drain: wait and compute remaining tiles
 *
 * Shared memory: D * 2 * TILE_ROWS * (HEAD_DIM + SMEM_PAD) * sizeof(float)
 *   D=3, TILE=8, HD=64, PAD=1: 3 * 2 * 8 * 65 * 4 = 12480 bytes (~12KB)
 *
 * For HD=128:  3 * 2 * 8 * 129 * 4 = 24768 bytes (~24KB)  -- still < 48KB
 *
 * ============================================================================
 * Architecture
 * ============================================================================
 * Grid:  (ceil(seq_len / QUERIES_PER_BLOCK), num_heads, batch_size)
 * Block: (WARP_SIZE, QUERIES_PER_BLOCK, 1) = (32, 8, 1) = 256 threads
 *   Same as Kernels 3/4/5 — identical occupancy
 *
 * ============================================================================
 * Reference
 * ============================================================================
 * CUDA Programming Guide: "Asynchronous Data Copies" (sm_80+)
 * CUTLASS: cute/arch/copy_sm80.hpp — __cp_async_ca_shared_global
 * FlashAttention-2 Appendix B: describes async pipeline implementation
 */

#include "flash_attention.h"
#include "utils.cuh"

#define NEG_INF            (-1e30f)
#define WARP_SIZE          32
#define QUERIES_PER_BLOCK  8      // 8 queries share each K/V tile
#define SMEM_PAD           1      // Bank-conflict-free padding (from K4)
#define PIPELINE_DEPTH     3      // Number of async pipeline stages

// ============================================================================
// cp.async compatibility macros
// ============================================================================
// sm_80+: Use __pipeline_memcpy_async (CUDA 11 cooperative groups pipeline)
// Older:  Fall back to plain synchronous copy
// We detect at compile time via __CUDA_ARCH__.

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
#  define USE_CP_ASYNC 1
#else
#  define USE_CP_ASYNC 0
#endif

// Macro to issue a single async copy of one float from global to shared.
// Copies sizeof(float) = 4 bytes per call.
// dst must be a pointer into shared memory.
// src must be a pointer into global memory.
#if USE_CP_ASYNC
#  define CP_ASYNC_F32(dst, src)                                   \
       asm volatile(                                                \
           "cp.async.ca.shared.global [%0], [%1], 4;\n"            \
           :                                                        \
           : "r"(__cvta_generic_to_shared(dst)),                   \
             "l"(src)                                              \
       )
#  define CP_ASYNC_COMMIT()                                        \
       asm volatile("cp.async.commit_group;\n" ::)
#  define CP_ASYNC_WAIT_N(N)                                       \
       asm volatile("cp.async.wait_group %0;\n" :: "n"(N))
#else
// Fallback: synchronous copy (correct on all hardware, no async benefit)
#  define CP_ASYNC_F32(dst, src)    *(dst) = *(src)
#  define CP_ASYNC_COMMIT()         /* no-op */
#  define CP_ASYNC_WAIT_N(N)        __syncthreads()
#endif

// ============================================================================
// Warp-level reduction primitives
// ============================================================================

__device__ __forceinline__ float warp_reduce_sum_v6(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}

__device__ __forceinline__ float warp_reduce_max_v6(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    return val;
}

// ============================================================================
// Helper: async-load one row of K or V tile into a shared memory buffer
//
// Each thread loads its ELEMS_PER_THREAD elements for its assigned dimension
// range. On sm_80+, this issues cp.async instructions that run in the
// background; on older hardware, it's a synchronous load.
//
// Parameters:
//   smem_row   : pointer to the destination row in shared memory
//   gmem_base  : pointer to the source row in global memory (K or V)
//   tid        : lane index (0..31)
// ============================================================================
template <int HEAD_DIM, int ELEMS_PER_THREAD>
__device__ __forceinline__ void async_load_row(
    float* __restrict__ smem_row,        // shared memory destination (row start)
    const float* __restrict__ gmem_row,  // global memory source (row start)
    int tid
) {
#pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
        const int d = tid + e * WARP_SIZE;
        if (d < HEAD_DIM) {
            CP_ASYNC_F32(&smem_row[d], &gmem_row[d]);
        }
    }
}

// ============================================================================
// Kernel v6: cp.async Hardware Pipeline Flash Attention
//
// Template parameters:
//   HEAD_DIM        : compile-time head dimension (32, 64, or 128)
//   ELEMS_PER_THREAD: HEAD_DIM / WARP_SIZE
// ============================================================================

template <int HEAD_DIM, int ELEMS_PER_THREAD>
__global__ void flash_attn_kernel_v6_impl(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float*       __restrict__ O,
    int          seq_len,
    int          head_dim,
    int64_t      batch_stride,
    int64_t      head_stride,
    float        softmax_scale
) {
    // ── Thread indexing ─────────────────────────────────────────────────────
    const int tid       = threadIdx.x;   // lane  0..31
    const int warp_id   = threadIdx.y;   // query 0..7  (QUERIES_PER_BLOCK - 1)
    const int query_row = blockIdx.x * QUERIES_PER_BLOCK + warp_id;
    const int head_idx  = blockIdx.y;
    const int batch_idx = blockIdx.z;

    if (query_row >= seq_len) return;

    const int64_t base_offset = (int64_t)batch_idx * batch_stride
                              + (int64_t)head_idx  * head_stride;
    const float* Q_base = Q + base_offset;
    const float* K_base = K + base_offset;
    const float* V_base = V + base_offset;
    float*       O_base = O + base_offset;

    // ── Shared memory layout (depth-3 pipeline) ────────────────────────────
    // We allocate PIPELINE_DEPTH ring-buffers for K and PIPELINE_DEPTH for V.
    //
    // Layout in smem:
    //   [K_buf_0 | K_buf_1 | K_buf_2 | V_buf_0 | V_buf_1 | V_buf_2]
    //
    // Each buffer holds QUERIES_PER_BLOCK rows of (HEAD_DIM + SMEM_PAD) floats.
    //
    extern __shared__ float smem[];
    const int padded_stride = HEAD_DIM + SMEM_PAD;   // 65 for HD=64
    const int buf_size      = QUERIES_PER_BLOCK * padded_stride;

    // K ring buffers
    float* s_K[PIPELINE_DEPTH];
    for (int i = 0; i < PIPELINE_DEPTH; ++i)
        s_K[i] = smem + i * buf_size;

    // V ring buffers (start after all K buffers)
    float* s_V[PIPELINE_DEPTH];
    for (int i = 0; i < PIPELINE_DEPTH; ++i)
        s_V[i] = smem + PIPELINE_DEPTH * buf_size + i * buf_size;

    // ── Load Q row into registers ───────────────────────────────────────────
    float q_reg[ELEMS_PER_THREAD];
#pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
        const int d = tid + e * WARP_SIZE;
        q_reg[e] = (d < HEAD_DIM) ? Q_base[query_row * HEAD_DIM + d] : 0.0f;
    }

    // ── Online softmax state ────────────────────────────────────────────────
    float m_cur = NEG_INF;
    float l_cur = 0.0f;
    float o_reg[ELEMS_PER_THREAD];
#pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; ++e) o_reg[e] = 0.0f;

    // ── Pipeline setup ──────────────────────────────────────────────────────
    const int TILE_ROWS = QUERIES_PER_BLOCK;   // 8
    const int num_tiles = (seq_len + TILE_ROWS - 1) / TILE_ROWS;

    // ── Stage 1: Fill the pipeline (issue first PIPELINE_DEPTH async loads) ─
    // We issue async loads for the first min(PIPELINE_DEPTH, num_tiles) tiles
    // before entering the main loop.
    const int fill_count = min(PIPELINE_DEPTH, num_tiles);
    for (int i = 0; i < fill_count; ++i) {
        const int tile_start = i * TILE_ROWS;
        const int tile_end   = min(tile_start + TILE_ROWS, seq_len);
        const int buf_idx    = i % PIPELINE_DEPTH;

        // Each warp loads its row if within bounds
        if (warp_id < (tile_end - tile_start)) {
            const int global_row = tile_start + warp_id;
            const float* k_src   = K_base + global_row * HEAD_DIM;
            const float* v_src   = V_base + global_row * HEAD_DIM;
            float* k_dst = s_K[buf_idx] + warp_id * padded_stride;
            float* v_dst = s_V[buf_idx] + warp_id * padded_stride;

            async_load_row<HEAD_DIM, ELEMS_PER_THREAD>(k_dst, k_src, tid);
            async_load_row<HEAD_DIM, ELEMS_PER_THREAD>(v_dst, v_src, tid);
        }
        CP_ASYNC_COMMIT();   // commit this stage to the async pipeline
    }

    // ── Stage 2: Main pipeline loop ─────────────────────────────────────────
    // For each tile i:
    //   1. Wait for tile i's async copy to complete (wait_prior(DEPTH-1))
    //   2. Compute attention on tile i
    //   3. Issue async load for tile (i + PIPELINE_DEPTH) if it exists
    //   4. Commit the new async stage

    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        const int cur_buf  = tile_idx % PIPELINE_DEPTH;

        // Wait for the OLDEST pending stage (i.e., tile_idx's stage) to finish.
        // After this, s_K[cur_buf] and s_V[cur_buf] are fully populated.
        //
        // cp.async.wait_group N: wait until there are at most N stages pending.
        // We want tile_idx to be done, so we wait until at most (PIPELINE_DEPTH-1)
        // later stages are still in-flight.
        CP_ASYNC_WAIT_N(PIPELINE_DEPTH - 1);
        __syncthreads();   // ensure all threads see the freshly-written smem

        // ── Compute attention scores on cur tile ────────────────────────────
        const int tile_start = tile_idx * TILE_ROWS;
        const int tile_end   = min(tile_start + TILE_ROWS, seq_len);
        const int tile_rows  = tile_end - tile_start;

        for (int r = 0; r < tile_rows; ++r) {
            float score = 0.0f;
#pragma unroll
            for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
                const int d = tid + e * WARP_SIZE;
                if (d < HEAD_DIM)
                    score += q_reg[e] * s_K[cur_buf][r * padded_stride + d];
            }
            score = warp_reduce_sum_v6(score) * softmax_scale;

            // Online softmax update
            float m_new  = fmaxf(m_cur, score);
            float alpha  = __expf(m_cur - m_new);
            float beta   = __expf(score - m_new);
            l_cur = alpha * l_cur + beta;
            m_cur = m_new;

            // Weighted V accumulation
#pragma unroll
            for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
                const int d = tid + e * WARP_SIZE;
                if (d < HEAD_DIM)
                    o_reg[e] = alpha * o_reg[e]
                             + beta  * s_V[cur_buf][r * padded_stride + d];
            }
        }

        // ── Prefetch tile (tile_idx + PIPELINE_DEPTH) if it exists ──────────
        // This re-uses cur_buf (it is now free since we just finished computing on it).
        const int next_tile = tile_idx + PIPELINE_DEPTH;
        if (next_tile < num_tiles) {
            const int next_start = next_tile * TILE_ROWS;
            const int next_end   = min(next_start + TILE_ROWS, seq_len);

            if (warp_id < (next_end - next_start)) {
                const int global_row = next_start + warp_id;
                const float* k_src   = K_base + global_row * HEAD_DIM;
                const float* v_src   = V_base + global_row * HEAD_DIM;
                float* k_dst = s_K[cur_buf] + warp_id * padded_stride;
                float* v_dst = s_V[cur_buf] + warp_id * padded_stride;

                async_load_row<HEAD_DIM, ELEMS_PER_THREAD>(k_dst, k_src, tid);
                async_load_row<HEAD_DIM, ELEMS_PER_THREAD>(v_dst, v_src, tid);
            }
            CP_ASYNC_COMMIT();
        }

        // Sync before next iteration reads from a different buf
        __syncthreads();
    }

    // ── Drain remaining in-flight stages ───────────────────────────────────
    // After the loop, there may be up to (fill_count - 1) stages still
    // pending (those issued during warm-up that didn't have a matching compute).
    // CP_ASYNC_WAIT_N(0) ensures all async copies are done.
    CP_ASYNC_WAIT_N(0);
    __syncthreads();

    // ── Normalize and write output ──────────────────────────────────────────
    const float inv_l = 1.0f / (l_cur + 1e-8f);
#pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
        const int d = tid + e * WARP_SIZE;
        if (d < HEAD_DIM)
            O_base[query_row * HEAD_DIM + d] = o_reg[e] * inv_l;
    }
}

// ============================================================================
// Host-side launcher
// ============================================================================

cudaError_t launch_flash_attn_v6(
    const float* Q, const float* K, const float* V, float* O,
    int batch_size, int num_heads, int seq_len, int head_dim,
    cudaStream_t stream
) {
    const int TILE_ROWS = QUERIES_PER_BLOCK;  // 8

    const int blocks_q = (seq_len + TILE_ROWS - 1) / TILE_ROWS;
    dim3 grid(blocks_q, num_heads, batch_size);
    dim3 block(WARP_SIZE, QUERIES_PER_BLOCK, 1);   // 256 threads

    const float   softmax_scale = 1.0f / sqrtf((float)head_dim);
    const int64_t head_stride   = (int64_t)seq_len * head_dim;
    const int64_t batch_stride  = (int64_t)num_heads * head_stride;

    // Shared memory: PIPELINE_DEPTH ring-buffers for K + PIPELINE_DEPTH for V
    // Each buffer: TILE_ROWS * (head_dim + SMEM_PAD) * sizeof(float)
    const size_t smem_bytes = (size_t)(2 * PIPELINE_DEPTH)
                            * TILE_ROWS
                            * (head_dim + SMEM_PAD)
                            * sizeof(float);

    // Safety: fall back to Kernel 5 if smem requirement exceeds 48 KB
    if (smem_bytes > 48 * 1024) {
        return launch_flash_attn_v5(Q, K, V, O,
                                    batch_size, num_heads, seq_len, head_dim,
                                    stream);
    }

    // Optional: use cudaFuncSetAttribute to request more smem on sm_80+
    // This allows up to 96-164 KB per block on A100/RTX 4080.
    // Uncomment if targeting HD > 128 in the future:
    // void* kernel_ptr = (void*)flash_attn_kernel_v6_impl<64, 2>;
    // cudaFuncSetAttribute(kernel_ptr,
    //     cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);

#define LAUNCH_V6(HD, EPT)                                                      \
    flash_attn_kernel_v6_impl<HD, EPT>                                          \
        <<<grid, block, smem_bytes, stream>>>(                                   \
            Q, K, V, O, seq_len, head_dim, batch_stride, head_stride,           \
            softmax_scale)

    if      (head_dim <= 32)  { LAUNCH_V6(32,  1); }
    else if (head_dim <= 64)  { LAUNCH_V6(64,  2); }
    else if (head_dim <= 128) { LAUNCH_V6(128, 4); }
    else {
        return launch_flash_attn_v5(Q, K, V, O,
                                    batch_size, num_heads, seq_len, head_dim,
                                    stream);
    }

#undef LAUNCH_V6

    return cudaGetLastError();
}
