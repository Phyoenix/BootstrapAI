/**
 * Flash Attention CUDA - Common Utilities
 * Shared header for all kernel implementations.
 */

#ifndef FLASH_UTILS_CUH
#define FLASH_UTILS_CUH

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

// ============================================================================
// Error checking macro
// ============================================================================

#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                        \
                    __FILE__, __LINE__, cudaGetErrorString(err));                \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

// ============================================================================
// Device-side math utilities
// ============================================================================

/**
 * Warp-level max reduction using XOR shuffle.
 * Assumes warp size = 32.
 */
__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

/**
 * Warp-level sum reduction using XOR shuffle.
 */
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

/**
 * Block-level max reduction using shared memory.
 * BLOCK_SIZE must be a power of 2 and <= 1024.
 */
template <int BLOCK_SIZE>
__device__ float block_reduce_max(float val) {
    __shared__ float smem[BLOCK_SIZE];
    int tid = threadIdx.x;

    smem[tid] = val;
    __syncthreads();

    // Sequential reduction: each halving step merges the upper half into the lower
    for (int s = BLOCK_SIZE >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        }
        __syncthreads();
    }

    return smem[0];
}

/**
 * Block-level sum reduction using shared memory.
 */
template <int BLOCK_SIZE>
__device__ float block_reduce_sum(float val) {
    __shared__ float smem[BLOCK_SIZE];
    int tid = threadIdx.x;

    smem[tid] = val;
    __syncthreads();

    for (int s = BLOCK_SIZE >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    return smem[0];
}

// ============================================================================
// Host-side timing utilities
// ============================================================================

/**
 * Simple CUDA timer using events.
 */
struct CudaTimer {
    cudaEvent_t start, stop;

    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }

    ~CudaTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void begin() {
        CUDA_CHECK(cudaEventRecord(start, 0));
    }

    float end() {
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }
};

#endif // FLASH_UTILS_CUH
