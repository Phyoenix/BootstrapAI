/**
 * Kernel 09: Grouped Query Attention (GQA) Flash Attention
 *
 * Key innovation: Extend Flash Attention to support Multi-Query Attention (MQA)
 * and Grouped Query Attention (GQA) — the attention variants used in LLaMA 2/3,
 * Mistral, Gemma, GPT-4, and virtually every modern production LLM.
 *
 * ============================================================================
 * Background: From MHA → MQA → GQA
 * ============================================================================
 *
 * Standard Multi-Head Attention (MHA, Kernels 1-8):
 *   Q: [B, H, N, D]    (H query heads)
 *   K: [B, H, N, D]    (H key heads — same count as Q)
 *   V: [B, H, N, D]    (H value heads — same count as Q)
 *   KV cache size ∝ H × N × D  (memory bottleneck at large context windows)
 *
 * Multi-Query Attention (MQA, Shazeer 2019):
 *   Q: [B, H, N, D]    (H query heads)
 *   K: [B, 1, N, D]    (1 key head — all query heads share this K)
 *   V: [B, 1, N, D]    (1 value head)
 *   KV cache size ∝ 1 × N × D  (H× reduction!)
 *   Used in: GPT-4 Turbo (speculated), Falcon, many encoder-decoder models
 *
 * Grouped Query Attention (GQA, Ainslie et al. 2023):
 *   Q: [B, H, N, D]    (H query heads, grouped into G groups)
 *   K: [B, G, N, D]    (G key heads — one per group)
 *   V: [B, G, N, D]    (G value heads)
 *   KV cache size ∝ G × N × D  (H/G reduction)
 *   Used in: LLaMA 2/3 (H=32, G=8), Mistral-7B (H=32, G=8), Gemma
 *   MHA = GQA(G=H), MQA = GQA(G=1)
 *
 * Why this matters for AI Infra:
 *   - KV cache is THE memory bottleneck for LLM inference at long context
 *   - GQA reduces KV memory by H/G while maintaining near-MHA quality
 *   - Production serving frameworks (vLLM, TensorRT-LLM, TGI) implement GQA
 *   - Interview question: "How does your kernel handle LLaMA 3 with GQA?"
 *     → This kernel is the complete answer.
 *
 * ============================================================================
 * Implementation Design
 * ============================================================================
 *
 * Data layout:
 *   Q: [B, H_q, N, D]        — all query heads (H_q heads total)
 *   K: [B, H_kv, N, D]       — K/V heads (H_kv heads, H_kv divides H_q)
 *   V: [B, H_kv, N, D]       — same layout as K
 *   group_size = H_q / H_kv  — queries per KV group
 *
 * Key mapping:
 *   For query head h_q, the corresponding K/V head is:
 *   h_kv = h_q / group_size   (integer division)
 *
 * Memory savings:
 *   KV data = H_kv / H_q fraction of MHA KV data
 *   For LLaMA-3 8B: H_q=32, H_kv=8 → 4× KV cache reduction
 *
 * Block assignment:
 *   Grid: (ceil(N / Q_BLOCK), H_q, B)  — one block per (q_tile, q_head, batch)
 *   Each block computes output for Q_BLOCK rows of head h_q
 *   Loads K/V from head h_kv = h_q / group_size
 *
 * Optimizations inherited (same as K4/K5/K6):
 *   - Shared memory tiling (K2/K3 style)
 *   - SMEM_PAD=1 bank conflict elimination (K4)
 *   - cp.async PTX for hardware-async HBM→SMEM (K6)
 *   - Software pipeline (K5 double buffering principle)
 *
 * Note: We use the simpler K4-style cooperative loading (not K7 warp
 * specialization) to keep this kernel focused on GQA logic clarity.
 * A follow-up Kernel 10 could combine GQA + warp specialization + persistent.
 *
 * ============================================================================
 * Performance vs Kernel 8 (MHA)
 * ============================================================================
 *
 * For MHA (H_kv == H_q): Kernel 9 and Kernel 8 are mathematically identical,
 * with K9 carrying a tiny overhead from the head index remapping.
 * For GQA (H_kv < H_q):
 *   - HBM reads for K/V data = 1/group_size of MHA → proportionally less
 *   - If memory-bound: expect group_size× speedup vs non-GQA kernel
 *   - If compute-bound: same FLOPs (attention ops unchanged), no speedup
 *
 * ============================================================================
 * ncu profiling targets
 * ============================================================================
 *   dram__bytes_read.sum            — should scale as 1/group_size vs K8
 *   smsp__sass_inst_executed_op_ffma_pred_on.sum  — FMAs (should be equal to K8)
 *   lts__average_gcomp_input_sector_hit_rate.pct  — L2 hit rate for K/V
 *
 * ============================================================================
 * Reference
 * ============================================================================
 *   "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head
 *   Checkpoints" (Ainslie et al., EMNLP 2023) arXiv:2305.13245
 *   LLaMA 2: https://arxiv.org/abs/2307.09288
 *   FlashAttention-2: Appendix B.3 (GQA / MQA handling)
 */

#include "flash_attention.h"
#include "utils.cuh"
#include <cuda_runtime.h>
#include <algorithm>

#define NEG_INF_V9         (-1e30f)
#define WARP_SIZE_V9       32
#define NUM_WARPS_V9       8          // 8 warps = 256 threads per block
#define Q_BLOCK_V9         NUM_WARPS_V9  // each warp handles one Q row
#define SMEM_PAD_V9        1          // +1 column to eliminate bank conflicts

// cp.async support (sm_80+, Ampere and later)
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
#  define USE_CP_ASYNC_V9 1
#endif

#ifdef USE_CP_ASYNC_V9
#  define CP_ASYNC_F32_V9(dst, src) \
     asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" \
                  :: "r"(__cvta_generic_to_shared(dst)), "l"(src))
#  define CP_ASYNC_COMMIT_V9() \
     asm volatile("cp.async.commit_group;\n" ::)
#  define CP_ASYNC_WAIT_ALL_V9() \
     asm volatile("cp.async.wait_all;\n" ::)
#else
#  define CP_ASYNC_F32_V9(dst, src)  do { *(dst) = *(src); } while(0)
#  define CP_ASYNC_COMMIT_V9()
#  define CP_ASYNC_WAIT_ALL_V9()
#endif

// ============================================================================
// Warp-level reduction (same pattern as K3-K8)
// ============================================================================

__device__ __forceinline__ float warp_reduce_sum_v9(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE_V9 / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// ============================================================================
// Shared memory layout (K/V tiles + SMEM_PAD, same as K4)
// ============================================================================

template <int HEAD_DIM>
struct GqaSmem {
    static constexpr int PADDED = HEAD_DIM + SMEM_PAD_V9;

    // Double-buffered K and V tiles (ping-pong buffer)
    float K_tile[2][Q_BLOCK_V9][PADDED];
    float V_tile[2][Q_BLOCK_V9][PADDED];
};

// ============================================================================
// GQA kernel implementation (template on HEAD_DIM)
// ============================================================================

template <int HEAD_DIM>
__global__ void flash_attn_gqa_kernel_impl(
    const float* __restrict__ Q,       // [B, H_q, N, D]
    const float* __restrict__ K,       // [B, H_kv, N, D]
    const float* __restrict__ V,       // [B, H_kv, N, D]
    float*       __restrict__ O,       // [B, H_q, N, D]
    int  seq_len,
    int  num_heads_q,    // H_q (total query heads)
    int  num_heads_kv,   // H_kv (key/value heads, divides H_q)
    int  group_size,     // H_q / H_kv — queries per KV group
    int64_t  q_batch_stride,   // H_q * N * D
    int64_t  q_head_stride,    // N * D
    int64_t  kv_batch_stride,  // H_kv * N * D
    int64_t  kv_head_stride,   // N * D
    float    softmax_scale
)
{
    // Thread identity
    const int tid      = threadIdx.x;               // 0..31 (within warp)
    const int warp_id  = threadIdx.y;               // 0..7

    // Block identity
    const int q_tile_idx = blockIdx.x;  // which Q-tile
    const int h_q        = blockIdx.y;  // which Q head
    const int b          = blockIdx.z;  // which batch

    // Derive corresponding K/V head via group mapping
    const int h_kv = h_q / group_size;  // integer division

    // Compile-time helpers
    constexpr int ELEMS_PER_THREAD = (HEAD_DIM + WARP_SIZE_V9 - 1) / WARP_SIZE_V9;
    constexpr int PADDED = HEAD_DIM + SMEM_PAD_V9;

    // Row this warp is responsible for within the Q tile
    const int q_row  = q_tile_idx * Q_BLOCK_V9 + warp_id;
    const bool valid = (q_row < seq_len);

    // Number of K/V tiles
    const int num_kv_tiles = (seq_len + Q_BLOCK_V9 - 1) / Q_BLOCK_V9;

    // Pointers to Q head and K/V head (note different head strides)
    const float* Q_head = Q + b * q_batch_stride  + h_q  * q_head_stride;
    const float* K_head = K + b * kv_batch_stride + h_kv * kv_head_stride;
    const float* V_head = V + b * kv_batch_stride + h_kv * kv_head_stride;
    float*       O_head = O + b * q_batch_stride  + h_q  * q_head_stride;

    // Shared memory
    __shared__ GqaSmem<HEAD_DIM> smem;

    // ── Load Q row into registers ─────────────────────────────────────────
    float q_reg[ELEMS_PER_THREAD];
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
        const int d = tid + e * WARP_SIZE_V9;
        q_reg[e] = (d < HEAD_DIM && valid)
                   ? Q_head[q_row * HEAD_DIM + d]
                   : 0.0f;
    }

    // ── Online softmax accumulators ───────────────────────────────────────
    float m_cur = NEG_INF_V9;  // running max (for numerical stability)
    float l_cur = 0.0f;        // running normalization factor
    float o_reg[ELEMS_PER_THREAD];
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; ++e) o_reg[e] = 0.0f;

    // ── Prefetch tile 0 ───────────────────────────────────────────────────
    // All warps cooperatively load the first K/V tile into slot 0
    // (identical pattern to K4/K6 cooperative loading)
    {
        constexpr int slot = 0;
        const int kv_row_start = 0;
        const int kv_row_end   = min(Q_BLOCK_V9, seq_len);
        const int tr           = kv_row_end - kv_row_start;

        // Each warp loads its own row from K and V
        if (warp_id < tr) {
            const float* K_src = K_head + (kv_row_start + warp_id) * HEAD_DIM;
            const float* V_src = V_head + (kv_row_start + warp_id) * HEAD_DIM;
            #pragma unroll
            for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
                const int d = tid + e * WARP_SIZE_V9;
                if (d < HEAD_DIM) {
                    CP_ASYNC_F32_V9(&smem.K_tile[slot][warp_id][d], &K_src[d]);
                    CP_ASYNC_F32_V9(&smem.V_tile[slot][warp_id][d], &V_src[d]);
                }
            }
        }
        CP_ASYNC_COMMIT_V9();
    }

    // ── Main tile loop: double-buffered compute + prefetch ────────────────
    for (int kv_tile = 0; kv_tile < num_kv_tiles; ++kv_tile) {
        const int curr_slot = kv_tile % 2;
        const int next_slot = 1 - curr_slot;

        // Wait for current tile to be ready
        CP_ASYNC_WAIT_ALL_V9();
        __syncthreads();  // Ensure all warps see the loaded K/V data

        // Prefetch NEXT K/V tile into next_slot (while computing on curr_slot)
        const int next_kv_tile = kv_tile + 1;
        if (next_kv_tile < num_kv_tiles) {
            const int next_kv_start = next_kv_tile * Q_BLOCK_V9;
            const int next_kv_end   = min(next_kv_start + Q_BLOCK_V9, seq_len);
            const int tr_next       = next_kv_end - next_kv_start;

            if (warp_id < tr_next) {
                const float* K_src = K_head + (next_kv_start + warp_id) * HEAD_DIM;
                const float* V_src = V_head + (next_kv_start + warp_id) * HEAD_DIM;
                #pragma unroll
                for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
                    const int d = tid + e * WARP_SIZE_V9;
                    if (d < HEAD_DIM) {
                        CP_ASYNC_F32_V9(&smem.K_tile[next_slot][warp_id][d], &K_src[d]);
                        CP_ASYNC_F32_V9(&smem.V_tile[next_slot][warp_id][d], &V_src[d]);
                    }
                }
            }
            CP_ASYNC_COMMIT_V9();
        }

        // Compute attention for current K/V tile
        const int kv_row_start = kv_tile * Q_BLOCK_V9;
        const int kv_row_end   = min(kv_row_start + Q_BLOCK_V9, seq_len);
        const int tr = kv_row_end - kv_row_start;

        // Each warp (= one Q row) processes all K/V rows in this tile
        for (int r = 0; r < tr; ++r) {
            // Q·K dot product
            float score = 0.0f;
            #pragma unroll
            for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
                const int d = tid + e * WARP_SIZE_V9;
                if (d < HEAD_DIM) {
                    score += q_reg[e] * smem.K_tile[curr_slot][r][d];
                }
            }
            score = warp_reduce_sum_v9(score) * softmax_scale;

            // Online softmax update (Flash Attention algorithm)
            const float m_new = fmaxf(m_cur, score);
            const float alpha  = __expf(m_cur - m_new);   // rescale old
            const float beta   = __expf(score  - m_new);  // weight for new
            l_cur = alpha * l_cur + beta;
            m_cur = m_new;

            // Accumulate output
            #pragma unroll
            for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
                const int d = tid + e * WARP_SIZE_V9;
                if (d < HEAD_DIM) {
                    o_reg[e] = alpha * o_reg[e]
                             + beta  * smem.V_tile[curr_slot][r][d];
                }
            }
        }

        // __syncthreads before next iteration to keep smem access safe
        __syncthreads();
    }

    // ── Write output ──────────────────────────────────────────────────────
    if (valid) {
        float* O_row = O_head + q_row * HEAD_DIM;
        const float inv_l = 1.0f / l_cur;
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
            const int d = tid + e * WARP_SIZE_V9;
            if (d < HEAD_DIM) {
                O_row[d] = o_reg[e] * inv_l;
            }
        }
    }
}

// ============================================================================
// Dispatch wrapper (runtime head_dim → compile-time template)
// ============================================================================

__global__ void flash_attn_kernel_v9(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float*       __restrict__ O,
    int   seq_len,
    int   head_dim,
    int   num_heads_q,
    int   num_heads_kv,
    int   group_size,
    int64_t  q_batch_stride,
    int64_t  q_head_stride,
    int64_t  kv_batch_stride,
    int64_t  kv_head_stride,
    float    softmax_scale
)
{
    if (head_dim == 32) {
        flash_attn_gqa_kernel_impl<32>(
            Q, K, V, O, seq_len,
            num_heads_q, num_heads_kv, group_size,
            q_batch_stride, q_head_stride,
            kv_batch_stride, kv_head_stride,
            softmax_scale);
    } else if (head_dim == 64) {
        flash_attn_gqa_kernel_impl<64>(
            Q, K, V, O, seq_len,
            num_heads_q, num_heads_kv, group_size,
            q_batch_stride, q_head_stride,
            kv_batch_stride, kv_head_stride,
            softmax_scale);
    } else if (head_dim == 128) {
        flash_attn_gqa_kernel_impl<128>(
            Q, K, V, O, seq_len,
            num_heads_q, num_heads_kv, group_size,
            q_batch_stride, q_head_stride,
            kv_batch_stride, kv_head_stride,
            softmax_scale);
    }
}

// ============================================================================
// Host launcher
// ============================================================================

cudaError_t launch_flash_attn_v9(
    const float* Q,         // [B, H_q, N, D]
    const float* K,         // [B, H_kv, N, D]   H_kv <= H_q
    const float* V,         // [B, H_kv, N, D]
    float*       O,         // [B, H_q, N, D]
    int batch_size,
    int num_heads_q,        // total query heads (H_q)
    int num_heads_kv,       // KV heads (H_kv; must divide H_q evenly)
    int seq_len,
    int head_dim,
    cudaStream_t stream     // = 0 by default
)
{
    // Validate inputs
    if (head_dim != 32 && head_dim != 64 && head_dim != 128) {
        return cudaErrorInvalidValue;
    }
    if (num_heads_q % num_heads_kv != 0) {
        // H_q must be divisible by H_kv for valid GQA grouping
        return cudaErrorInvalidValue;
    }

    const int group_size = num_heads_q / num_heads_kv;
    const float softmax_scale = 1.0f / sqrtf((float)head_dim);

    // Stride computations
    //   Q layout: [B, H_q,  N, D] — note H_q heads for Q
    //   K layout: [B, H_kv, N, D] — note H_kv heads for K/V
    const int64_t q_head_stride   = (int64_t)seq_len * head_dim;
    const int64_t q_batch_stride  = (int64_t)num_heads_q  * q_head_stride;
    const int64_t kv_head_stride  = (int64_t)seq_len * head_dim;
    const int64_t kv_batch_stride = (int64_t)num_heads_kv * kv_head_stride;

    // Grid: (q_tiles, H_q, B)
    const int num_q_tiles = (seq_len + Q_BLOCK_V9 - 1) / Q_BLOCK_V9;
    const dim3 grid(num_q_tiles, num_heads_q, batch_size);
    const dim3 block(WARP_SIZE_V9, NUM_WARPS_V9, 1);   // 256 threads

    flash_attn_kernel_v9<<<grid, block, 0, stream>>>(
        Q, K, V, O,
        seq_len, head_dim,
        num_heads_q, num_heads_kv, group_size,
        q_batch_stride, q_head_stride,
        kv_batch_stride, kv_head_stride,
        softmax_scale
    );

    return cudaGetLastError();
}

// ============================================================================
// Convenience launcher: MHA mode (H_kv = H_q, backward compatible with K1-K8)
// ============================================================================
cudaError_t launch_flash_attn_v9_mha(
    const float* Q,
    const float* K,
    const float* V,
    float*       O,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    cudaStream_t stream
)
{
    // Degenerate case: num_heads_kv = num_heads_q → standard MHA
    return launch_flash_attn_v9(
        Q, K, V, O,
        batch_size, num_heads, num_heads,  // H_kv = H_q
        seq_len, head_dim, stream
    );
}
