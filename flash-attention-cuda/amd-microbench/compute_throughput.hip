/**
 * compute_throughput.hip — GPU Compute Throughput Micro-Benchmark
 *
 * Measures peak compute throughput under controlled workloads:
 *
 *   1. FP32 FMA (Fused Multiply-Add) — peak FLOPS metric
 *   2. FP32 ADD chain              — latency-bound scalar additions
 *   3. FP32 MUL chain              — latency-bound multiplications
 *   4. FP64 FMA                    — double precision throughput
 *   5. Varying thread-count sweep  — find peak occupancy
 *
 * Design principle:
 *   To measure *throughput* (not latency), we use independent chains:
 *     val = val * a + b;  // no inter-iteration dependency
 *   To measure *latency* (serialized), we use a dependent chain:
 *     val = val * val;    // each iteration depends on previous
 *
 * Output:
 *   [TEST]  threads  GFLOPS  time_ms  iters
 *
 * Build:
 *   hipcc -O3 --offload-arch=gfx1100 amd-microbench/compute_throughput.hip \
 *         -o build/microbench/compute_throughput
 *
 * Run:
 *   ./build/microbench/compute_throughput
 */

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <climits>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#define HIP_CHECK(call)                                                      \
    do {                                                                      \
        hipError_t _e = (call);                                              \
        if (_e != hipSuccess) {                                              \
            fprintf(stderr, "HIP error %s:%d  %s\n",                        \
                    __FILE__, __LINE__, hipGetErrorString(_e));              \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

// Number of independent FMA ops per thread per iteration.
// More ops → more ILP → better throughput measurement.
#define OPS_PER_ITER 32

// ---------------------------------------------------------------------------
// Kernel 1: FP32 FMA throughput
//
// Strategy: OPS_PER_ITER independent accumulations, minimizing loop overhead.
// Compiler will unroll and issue them concurrently (ILP).
// ---------------------------------------------------------------------------
__global__ void kernel_fp32_fma(float* sink, int iters, float a, float b) {
    float v0  = (float)threadIdx.x * 0.001f + 1.0f;
    float v1  = v0 + 0.1f;
    float v2  = v0 + 0.2f;
    float v3  = v0 + 0.3f;
    float v4  = v0 + 0.4f;
    float v5  = v0 + 0.5f;
    float v6  = v0 + 0.6f;
    float v7  = v0 + 0.7f;

    // Prevent optimising constants to compile-time values
    __asm__ volatile("" : "+v"(a), "+v"(b));

    for (int i = 0; i < iters; i++) {
        // 8 independent FMAs → compiler issues them in parallel
        v0 = v0 * a + b;
        v1 = v1 * a + b;
        v2 = v2 * a + b;
        v3 = v3 * a + b;
        v4 = v4 * a + b;
        v5 = v5 * a + b;
        v6 = v6 * a + b;
        v7 = v7 * a + b;
    }

    // Accumulate all into sink to prevent DCE
    float acc = v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7;
    if (acc != 0.f && blockIdx.x == 0 && threadIdx.x == 0) *sink = acc;
}

// ---------------------------------------------------------------------------
// Kernel 2: FP32 ADD latency chain (serialised — measures add latency)
// ---------------------------------------------------------------------------
__global__ void kernel_fp32_add_chain(float* sink, int iters, float delta) {
    float val = (float)(threadIdx.x + blockIdx.x) * 0.001f;
    __asm__ volatile("" : "+v"(delta));
    for (int i = 0; i < iters; i++) {
        val = val + delta;   // serial dependency
    }
    if (val != 0.f && blockIdx.x == 0 && threadIdx.x == 0) *sink = val;
}

// ---------------------------------------------------------------------------
// Kernel 3: FP32 MUL throughput (independent chains)
// ---------------------------------------------------------------------------
__global__ void kernel_fp32_mul(float* sink, int iters, float factor) {
    float v0 = (float)threadIdx.x * 0.001f + 1.0001f;
    float v1 = v0 + 0.001f;
    float v2 = v0 + 0.002f;
    float v3 = v0 + 0.003f;
    __asm__ volatile("" : "+v"(factor));
    for (int i = 0; i < iters; i++) {
        v0 = v0 * factor;
        v1 = v1 * factor;
        v2 = v2 * factor;
        v3 = v3 * factor;
    }
    float acc = v0 + v1 + v2 + v3;
    if (acc != 0.f && blockIdx.x == 0 && threadIdx.x == 0) *sink = acc;
}

// ---------------------------------------------------------------------------
// Kernel 4: FP64 FMA throughput
// ---------------------------------------------------------------------------
__global__ void kernel_fp64_fma(double* sink, int iters, double a, double b) {
    double v0 = (double)threadIdx.x * 0.001 + 1.0;
    double v1 = v0 + 0.01;
    double v2 = v0 + 0.02;
    double v3 = v0 + 0.03;
    for (int i = 0; i < iters; i++) {
        v0 = v0 * a + b;
        v1 = v1 * a + b;
        v2 = v2 * a + b;
        v3 = v3 * a + b;
    }
    double acc = v0 + v1 + v2 + v3;
    if (acc != 0.0 && blockIdx.x == 0 && threadIdx.x == 0) *sink = acc;
}

// ---------------------------------------------------------------------------
// Benchmark infrastructure
// ---------------------------------------------------------------------------

struct Result {
    const char* name;
    double gflops;
    double time_ms;
    int total_threads;
    long long total_flops;
};

// Run a throughput kernel and compute GFLOPS.
// flops_per_thread_per_iter = number of floating-point ops in one iteration.
template<typename KernelFn>
static Result bench_compute(
    const char* name,
    KernelFn kernel_fn,          // std::function<void(hipStream_t)>
    int grid, int block,
    long long flops_per_iter,    // total FLOPs across all threads per launch
    int iters,                   // loop iterations inside kernel
    int warmup, int repeats
) {
    hipEvent_t t0, t1;
    hipEventCreate(&t0); hipEventCreate(&t1);

    for (int i = 0; i < warmup; i++) kernel_fn(0);
    hipDeviceSynchronize();

    hipEventRecord(t0);
    for (int i = 0; i < repeats; i++) kernel_fn(0);
    hipEventRecord(t1);
    hipDeviceSynchronize();

    double ms = 0.f;
    hipEventElapsedTime((float*)&ms, t0, t1);
    ms /= repeats;

    long long total_flops = flops_per_iter * (long long)iters;
    double gflops = (double)total_flops / (ms * 1e-3) / 1e9;

    hipEventDestroy(t0); hipEventDestroy(t1);
    return {name, gflops, ms, grid * block, total_flops};
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));

    printf("==================================================================\n");
    printf("  GPU Compute Throughput Micro-Benchmark (AMD HIP)\n");
    printf("==================================================================\n");
    printf("  Device            : %s\n", prop.name);
    // Peak FP32 FLOPs = CUs * SIMD units * clock * 2 (FMA counts as 2)
    // For RDNA3 (gfx1100): 60 CUs * 64 shaders/CU * 2.5 GHz * 2 = ~19.2 TFLOPS
    printf("  CUs               : %d\n", prop.multiProcessorCount);
    printf("  Peak clock        : %d MHz\n", prop.clockRate / 1000);
    printf("  Warp / Wavefront  : %d threads\n", prop.warpSize);
    printf("  Shared mem / block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("==================================================================\n\n");

    float  *d_sink_fp32;
    double *d_sink_fp64;
    HIP_CHECK(hipMalloc(&d_sink_fp32, sizeof(float)));
    HIP_CHECK(hipMalloc(&d_sink_fp64, sizeof(double)));

    // Grid/block config — launch enough threads to saturate GPU
    int sm_count = prop.multiProcessorCount;
    int block_size = 256;
    int grid_size  = sm_count * 4;   // 4 blocks per SM for good occupancy

    long long threads = (long long)grid_size * block_size;
    int kernel_iters = 10000;

    printf("%-22s  %10s  %10s  %12s  %s\n",
           "Test", "GFLOPS", "Time(ms)", "Total-FLOPs", "Grid×Block");
    printf("%-22s  %10s  %10s  %12s  %s\n",
           "----------------------", "----------", "----------",
           "------------", "----------");

    auto print = [](Result r) {
        printf("%-22s  %10.2f  %10.3f  %12.2e  %d\n",
               r.name, r.gflops, r.time_ms,
               (double)r.total_flops, r.total_threads);
    };

    // -----------------------------------------------------------------------
    // Test 1: FP32 FMA throughput
    // 8 FMAs/iter × 2 FLOPs/FMA = 16 FLOPs per thread per iter
    // -----------------------------------------------------------------------
    float a_fp32 = 1.00001f, b_fp32 = 0.00001f;
    long long fp32_fma_flops = threads * 8 * 2;   // 8 FMAs × 2
    {
        auto fn = [&](hipStream_t s) {
            kernel_fp32_fma<<<grid_size, block_size, 0, s>>>(
                d_sink_fp32, kernel_iters, a_fp32, b_fp32);
        };
        print(bench_compute("FP32_FMA (thrput)", fn, grid_size, block_size,
                             fp32_fma_flops, kernel_iters, 2, 5));
    }

    // -----------------------------------------------------------------------
    // Test 2: FP32 ADD latency chain (1 ADD/iter per thread)
    // -----------------------------------------------------------------------
    {
        long long fp32_add_flops = threads * 1;
        auto fn = [&](hipStream_t s) {
            kernel_fp32_add_chain<<<grid_size, block_size, 0, s>>>(
                d_sink_fp32, kernel_iters, 0.00001f);
        };
        print(bench_compute("FP32_ADD (latency)", fn, grid_size, block_size,
                             fp32_add_flops, kernel_iters, 2, 5));
    }

    // -----------------------------------------------------------------------
    // Test 3: FP32 MUL throughput (4 chains)
    // -----------------------------------------------------------------------
    {
        long long fp32_mul_flops = threads * 4;
        auto fn = [&](hipStream_t s) {
            kernel_fp32_mul<<<grid_size, block_size, 0, s>>>(
                d_sink_fp32, kernel_iters, 1.00001f);
        };
        print(bench_compute("FP32_MUL (thrput)", fn, grid_size, block_size,
                             fp32_mul_flops, kernel_iters, 2, 5));
    }

    // -----------------------------------------------------------------------
    // Test 4: FP64 FMA throughput (4 chains)
    // -----------------------------------------------------------------------
    {
        long long fp64_fma_flops = threads * 4 * 2;   // 4 FMAs × 2
        auto fn = [&](hipStream_t s) {
            kernel_fp64_fma<<<grid_size, block_size, 0, s>>>(
                d_sink_fp64, kernel_iters, 1.0000001, 0.0000001);
        };
        print(bench_compute("FP64_FMA (thrput)", fn, grid_size, block_size,
                             fp64_fma_flops, kernel_iters, 2, 5));
    }

    // -----------------------------------------------------------------------
    // Test 5: Occupancy sweep — vary block size, measure FP32 FMA GFLOPS
    // -----------------------------------------------------------------------
    printf("\n--- Occupancy Sweep (FP32 FMA) ---\n");
    printf("%-10s  %10s  %10s\n", "BlockSize", "GFLOPS", "Rel%%");
    double peak_gflops = 0.0;
    int    peak_block  = 0;

    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    double gflops_by_block[6] = {};

    for (int bi = 0; bi < 6; bi++) {
        int bs = block_sizes[bi];
        int gs = sm_count * 4;
        long long thr = (long long)gs * bs;
        long long flops = thr * 8 * 2;

        auto fn = [&](hipStream_t s) {
            kernel_fp32_fma<<<gs, bs, 0, s>>>(
                d_sink_fp32, kernel_iters, a_fp32, b_fp32);
        };
        Result r = bench_compute("sweep", fn, gs, bs, flops, kernel_iters, 2, 3);
        gflops_by_block[bi] = r.gflops;
        if (r.gflops > peak_gflops) {
            peak_gflops = r.gflops;
            peak_block  = bs;
        }
    }

    for (int bi = 0; bi < 6; bi++) {
        printf("%-10d  %10.2f  %9.1f%%\n",
               block_sizes[bi], gflops_by_block[bi],
               100.0 * gflops_by_block[bi] / peak_gflops);
    }
    printf("=> Peak @ block_size=%d: %.2f GFLOPS\n", peak_block, peak_gflops);

    printf("\nInterpretation:\n");
    printf("  FP32_FMA thrput >> FP32_ADD latency → GPU hides latency via ILP\n");
    printf("  FP64 << FP32 → typical 1/16 ratio on gaming GPUs\n");
    printf("  Occupancy sweep → find where latency hiding saturates\n");

    hipFree(d_sink_fp32);
    hipFree(d_sink_fp64);
    return 0;
}
