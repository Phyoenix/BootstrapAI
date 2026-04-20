/**
 * memory_bandwidth.hip — GPU Memory Bandwidth Micro-Benchmark
 *
 * Measures HBM (Global Memory) and LDS (Local Data Share / Shared Memory)
 * bandwidth under multiple access patterns:
 *
 *   1. Sequential Read   — fully coalesced, maximum bandwidth
 *   2. Strided Read      — stride=2,4,8,16  — degradation vs coalescing
 *   3. Random Read       — worst case, exposing cache miss penalties
 *   4. Sequential Write  — write bandwidth
 *   5. LDS Bandwidth     — on-chip shared memory throughput
 *
 * Output format (all results to stdout):
 *   [PATTERN]  N_elements  bandwidth_GB/s  time_ms
 *
 * Build:
 *   hipcc -O3 --offload-arch=gfx1100 amd-microbench/memory_bandwidth.hip \
 *         -o build/microbench/memory_bandwidth
 *
 * Run:
 *   ./build/microbench/memory_bandwidth
 *
 * Reference: inspired by CUDA Bandwidth Test (cuda-samples) and
 *            AMD rocm-bandwidth-test.
 */

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#define HIP_CHECK(call)                                                     \
    do {                                                                     \
        hipError_t _e = (call);                                             \
        if (_e != hipSuccess) {                                             \
            fprintf(stderr, "HIP error %s:%d  %s\n",                       \
                    __FILE__, __LINE__, hipGetErrorString(_e));             \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

static double elapsed_ms(hipEvent_t start, hipEvent_t stop) {
    float ms = 0.f;
    hipEventElapsedTime(&ms, start, stop);
    return (double)ms;
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// --- 1. Sequential (coalesced) read ---
__global__ void kernel_seq_read(
    const float* __restrict__ src, float* __restrict__ sink, int N
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float s = 0.f;
    for (int i = tid; i < N; i += blockDim.x * gridDim.x) {
        s += src[i];
    }
    // Prevent dead-code elimination
    if (tid == 0 && s != 0.f) *sink = s;
}

// --- 2. Strided read ---
__global__ void kernel_stride_read(
    const float* __restrict__ src, float* __restrict__ sink, int N, int stride
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float s = 0.f;
    for (int i = (tid * stride) % N; i < N; i = (i + stride * blockDim.x * gridDim.x) % (N + 1)) {
        if (i >= N) break;
        s += src[i];
    }
    if (tid == 0 && s != 0.f) *sink = s;
}

// --- 3. Random read (pseudo-random using LCG) ---
__global__ void kernel_rand_read(
    const float* __restrict__ src, float* __restrict__ sink, int N, int iters
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // LCG: simple, avoids extra memory for index array
    unsigned int state = (unsigned int)(tid + 12345);
    float s = 0.f;
    for (int it = 0; it < iters; it++) {
        state = state * 1664525u + 1013904223u;   // Knuth LCG
        int idx = (int)(state % (unsigned int)N);
        s += src[idx];
    }
    if (tid == 0 && s != 0.f) *sink = s;
}

// --- 4. Sequential write ---
__global__ void kernel_seq_write(float* __restrict__ dst, int N, float val) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = tid; i < N; i += blockDim.x * gridDim.x) {
        dst[i] = val;
    }
}

// --- 5. LDS (shared memory) bandwidth ---
// Each block loads a tile into shared memory, sums it, writes result.
#define LDS_TILE 1024   // floats = 4 KB (well within LDS limit)
__global__ void kernel_lds_bw(
    const float* __restrict__ src, float* __restrict__ dst, int N
) {
    __shared__ float smem[LDS_TILE];
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    float acc = 0.f;
    int start = bid * LDS_TILE;
    while (start < N) {
        int end = min(start + LDS_TILE, N);
        int len = end - start;

        // Load into LDS
        if (tid < len) smem[tid] = src[start + tid];
        __syncthreads();

        // Read from LDS (simulates LDS bandwidth)
        if (tid < len) acc += smem[tid];
        __syncthreads();

        start += gridDim.x * LDS_TILE;
    }
    if (tid == 0) dst[bid] = acc;
}

// ---------------------------------------------------------------------------
// Benchmark runner
// ---------------------------------------------------------------------------

struct BenchResult {
    const char* name;
    double bandwidth_GBs;   // GB/s
    double time_ms;
    size_t bytes_transferred;
};

static BenchResult run_seq_read(const float* d_src, float* d_sink, int N,
                                  int warmup, int iters) {
    hipEvent_t t0, t1;
    hipEventCreate(&t0); hipEventCreate(&t1);

    dim3 block(256), grid((N + 255) / 256);
    // Warmup
    for (int i = 0; i < warmup; i++)
        kernel_seq_read<<<grid, block>>>(d_src, d_sink, N);
    hipDeviceSynchronize();

    hipEventRecord(t0);
    for (int i = 0; i < iters; i++)
        kernel_seq_read<<<grid, block>>>(d_src, d_sink, N);
    hipEventRecord(t1);
    hipDeviceSynchronize();

    double ms = elapsed_ms(t0, t1) / iters;
    size_t bytes = (size_t)N * sizeof(float);
    double bw = (double)bytes / (ms * 1e-3) / 1e9;

    hipEventDestroy(t0); hipEventDestroy(t1);
    return {"SEQ_READ", bw, ms, bytes};
}

static BenchResult run_stride_read(const float* d_src, float* d_sink, int N,
                                     int stride, int warmup, int iters) {
    hipEvent_t t0, t1;
    hipEventCreate(&t0); hipEventCreate(&t1);

    dim3 block(256), grid(min((N + 255) / 256, 2048));
    for (int i = 0; i < warmup; i++)
        kernel_stride_read<<<grid, block>>>(d_src, d_sink, N, stride);
    hipDeviceSynchronize();

    hipEventRecord(t0);
    for (int i = 0; i < iters; i++)
        kernel_stride_read<<<grid, block>>>(d_src, d_sink, N, stride);
    hipEventRecord(t1);
    hipDeviceSynchronize();

    double ms = elapsed_ms(t0, t1) / iters;
    // Actual bytes: N/stride elements accessed, each 4 bytes
    size_t elems = (size_t)N / stride;
    size_t bytes = elems * sizeof(float);
    double bw = (double)bytes / (ms * 1e-3) / 1e9;

    hipEventDestroy(t0); hipEventDestroy(t1);
    char name[32];
    snprintf(name, sizeof(name), "STRIDE_%d_READ", stride);
    // Return static name — benchmark result is printed immediately
    static char sname[32];
    memcpy(sname, name, 32);
    return {sname, bw, ms, bytes};
}

static BenchResult run_rand_read(const float* d_src, float* d_sink, int N,
                                   int iters_per_thread, int warmup, int iters) {
    hipEvent_t t0, t1;
    hipEventCreate(&t0); hipEventCreate(&t1);

    dim3 block(256), grid(512);
    for (int i = 0; i < warmup; i++)
        kernel_rand_read<<<grid, block>>>(d_src, d_sink, N, iters_per_thread);
    hipDeviceSynchronize();

    hipEventRecord(t0);
    for (int i = 0; i < iters; i++)
        kernel_rand_read<<<grid, block>>>(d_src, d_sink, N, iters_per_thread);
    hipEventRecord(t1);
    hipDeviceSynchronize();

    double ms = elapsed_ms(t0, t1) / iters;
    // Each thread does iters_per_thread random 4-byte reads
    size_t total_threads = 256ULL * 512;
    size_t bytes = total_threads * (size_t)iters_per_thread * sizeof(float);
    double bw = (double)bytes / (ms * 1e-3) / 1e9;

    hipEventDestroy(t0); hipEventDestroy(t1);
    return {"RAND_READ", bw, ms, bytes};
}

static BenchResult run_seq_write(float* d_dst, int N, int warmup, int iters) {
    hipEvent_t t0, t1;
    hipEventCreate(&t0); hipEventCreate(&t1);

    dim3 block(256), grid((N + 255) / 256);
    for (int i = 0; i < warmup; i++)
        kernel_seq_write<<<grid, block>>>(d_dst, N, 1.0f);
    hipDeviceSynchronize();

    hipEventRecord(t0);
    for (int i = 0; i < iters; i++)
        kernel_seq_write<<<grid, block>>>(d_dst, N, 1.0f);
    hipEventRecord(t1);
    hipDeviceSynchronize();

    double ms = elapsed_ms(t0, t1) / iters;
    size_t bytes = (size_t)N * sizeof(float);
    double bw = (double)bytes / (ms * 1e-3) / 1e9;

    hipEventDestroy(t0); hipEventDestroy(t1);
    return {"SEQ_WRITE", bw, ms, bytes};
}

static BenchResult run_lds_bw(const float* d_src, float* d_dst, int N,
                                int warmup, int iters) {
    hipEvent_t t0, t1;
    hipEventCreate(&t0); hipEventCreate(&t1);

    int nblocks = (N + LDS_TILE - 1) / LDS_TILE;
    dim3 block(LDS_TILE), grid(nblocks);

    for (int i = 0; i < warmup; i++)
        kernel_lds_bw<<<grid, block>>>(d_src, d_dst, N);
    hipDeviceSynchronize();

    hipEventRecord(t0);
    for (int i = 0; i < iters; i++)
        kernel_lds_bw<<<grid, block>>>(d_src, d_dst, N);
    hipEventRecord(t1);
    hipDeviceSynchronize();

    double ms = elapsed_ms(t0, t1) / iters;
    // LDS: each element loaded once from HBM into LDS, then read once from LDS
    size_t bytes = (size_t)N * sizeof(float) * 2;  // 1 HBM load + 1 LDS read
    double bw = (double)bytes / (ms * 1e-3) / 1e9;

    hipEventDestroy(t0); hipEventDestroy(t1);
    return {"LDS_BW", bw, ms, bytes};
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    // Default: 256 MB of floats
    int N = 64 * 1024 * 1024;   // 256 MB
    if (argc > 1) N = atoi(argv[1]);

    size_t total_bytes = (size_t)N * sizeof(float);

    printf("=================================================================\n");
    printf("  GPU Memory Bandwidth Micro-Benchmark (AMD HIP)\n");
    printf("=================================================================\n");
    printf("  Buffer size : %d elements = %.1f MB\n",
           N, (double)total_bytes / 1024 / 1024);

    // Query device
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    printf("  Device      : %s\n", prop.name);
    printf("  HBM peak    : ~%.0f GB/s (theoretical)\n",
           prop.memoryBusWidth / 8.0 * prop.memoryClockRate * 2e-6);
    printf("=================================================================\n\n");

    // Allocate
    float *d_src, *d_dst, *d_sink;
    HIP_CHECK(hipMalloc(&d_src,  total_bytes));
    HIP_CHECK(hipMalloc(&d_dst,  total_bytes));
    HIP_CHECK(hipMalloc(&d_sink, sizeof(float)));

    // Initialize src
    kernel_seq_write<<<(N+255)/256, 256>>>(d_src, N, 1.0f / sqrtf((float)N));
    hipDeviceSynchronize();

    // --- Run benchmarks ---
    printf("%-20s  %10s  %10s  %10s\n",
           "Pattern", "BW (GB/s)", "Time (ms)", "Bytes (MB)");
    printf("%-20s  %10s  %10s  %10s\n",
           "--------------------", "----------", "----------", "----------");

    auto print = [](BenchResult r) {
        printf("%-20s  %10.2f  %10.3f  %10.1f\n",
               r.name,
               r.bandwidth_GBs,
               r.time_ms,
               (double)r.bytes_transferred / 1024 / 1024);
    };

    print(run_seq_read(d_src, d_sink, N,  2, 10));
    print(run_stride_read(d_src, d_sink, N,  2, 2, 5));
    print(run_stride_read(d_src, d_sink, N,  4, 2, 5));
    print(run_stride_read(d_src, d_sink, N, 16, 2, 5));
    print(run_rand_read(d_src, d_sink, N, 64, 2, 5));
    print(run_seq_write(d_dst, N, 2, 10));
    print(run_lds_bw(d_src, d_dst, N, 2, 10));

    printf("\n");
    printf("Interpretation:\n");
    printf("  SEQ_READ   ≈ peak HBM bandwidth (coalesced, cache-friendly)\n");
    printf("  STRIDE_N   — bandwidth drops as stride increases (cache line waste)\n");
    printf("  RAND_READ  ≈ random access penalty (cache thrashing)\n");
    printf("  SEQ_WRITE  — write bandwidth (often < read due to write-combine)\n");
    printf("  LDS_BW     — on-chip LDS throughput (should >> HBM)\n");

    hipFree(d_src); hipFree(d_dst); hipFree(d_sink);
    return 0;
}
