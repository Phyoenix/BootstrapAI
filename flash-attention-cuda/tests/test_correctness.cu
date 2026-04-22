/**
 * Test Correctness for Flash Attention CUDA Kernels
 *
 * Compares CUDA kernel output against CPU reference implementation.
 * Tests kernel v1 (naive), v2 (tiling), v3 (cooperative), v4 (swizzle),
 * v5 (double buffering), v6 (cp.async hardware pipeline),
 * v7 (warp specialization), v8 (persistent kernel),
 * v9 (GQA: grouped query attention — MHA/GQA/MQA unified).
 *
 * Build:
 *   nvcc -O3 -arch=sm_89 -I../include test_correctness.cu \
 *     ../kernels/kernel_01_naive.cu ../kernels/kernel_02_tiling.cu \
 *     ../kernels/kernel_03_cooperative.cu ../kernels/kernel_04_swizzle.cu \
 *     ../kernels/kernel_05_double_buffer.cu ../kernels/kernel_06_cp_async.cu \
 *     ../kernels/kernel_07_warp_specialization.cu \
 *     ../kernels/kernel_08_persistent.cu \
 *     ../kernels/kernel_09_gqa.cu \
 *     -o test_correctness -lcudart
 *
 * Run:
 *   ./test_correctness
 */

#include "flash_attention.h"
#include "utils.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>

// ============================================================================
// Test configuration
// ============================================================================

struct TestCase {
    int batch_size;
    int num_heads;
    int seq_len;
    int head_dim;
    std::string name;
};

// ============================================================================
// Reference implementation (CPU) for ground truth
// ============================================================================

/**
 * CPU reference: standard scaled dot-product attention.
 * O = softmax(Q @ K^T / sqrt(d)) @ V
 */
void standard_attention_cpu(
    const float* Q, const float* K, const float* V, float* O,
    int batch_size, int num_heads, int seq_len, int head_dim
) {
    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < num_heads; h++) {
            int64_t base = b * num_heads * seq_len * head_dim + h * seq_len * head_dim;

            for (int i = 0; i < seq_len; i++) {
                // Compute attention scores for query row i
                float max_score = -1e30f;

                // First pass: find max score for numerical stability
                std::vector<float> scores(seq_len);
                for (int j = 0; j < seq_len; j++) {
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; d++) {
                        dot += Q[base + i * head_dim + d] * K[base + j * head_dim + d];
                    }
                    scores[j] = dot * scale;
                    max_score = fmaxf(max_score, scores[j]);
                }

                // Second pass: compute exp and sum
                float sum_exp = 0.0f;
                for (int j = 0; j < seq_len; j++) {
                    scores[j] = expf(scores[j] - max_score);
                    sum_exp += scores[j];
                }

                // Third pass: compute weighted sum with V
                float inv_sum = 1.0f / sum_exp;
                for (int d = 0; d < head_dim; d++) {
                    float val = 0.0f;
                    for (int j = 0; j < seq_len; j++) {
                        val += scores[j] * V[base + j * head_dim + d];
                    }
                    O[base + i * head_dim + d] = val * inv_sum;
                }
            }
        }
    }
}

// ============================================================================
// Test utilities
// ============================================================================

void init_random(float* data, int size, unsigned int seed = 42) {
    srand(seed);
    for (int i = 0; i < size; i++) {
        data[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;  // [-0.5, 0.5]
    }
}

bool compare_results(
    const float* computed, const float* expected,
    int size, float rtol = 1e-3f, float atol = 1e-5f
) {
    float max_diff = 0.0f;
    float mean_diff = 0.0f;
    int fail_count = 0;

    for (int i = 0; i < size; i++) {
        float diff = fabsf(computed[i] - expected[i]);
        float rel_diff = fabsf(expected[i]) > 1e-6f ? diff / fabsf(expected[i]) : diff;
        max_diff = fmaxf(max_diff, diff);
        mean_diff += diff;

        if (rel_diff > rtol && diff > atol) {
            fail_count++;
        }
    }
    mean_diff /= size;

    printf("    Max diff:  %.2e\n", max_diff);
    printf("    Mean diff: %.2e\n", mean_diff);
    printf("    Failures:  %d / %d\n", fail_count, size);

    return fail_count == 0;
}

// ============================================================================
// Test runner
// ============================================================================

bool run_test(const TestCase& tc) {
    printf("\n  Testing: %s (batch=%d, heads=%d, seq=%d, dim=%d)\n",
           tc.name.c_str(), tc.batch_size, tc.num_heads, tc.seq_len, tc.head_dim);

    int64_t total_elements = (int64_t)tc.batch_size * tc.num_heads * tc.seq_len * tc.head_dim;
    size_t bytes = total_elements * sizeof(float);

    // Allocate host memory
    std::vector<float> h_Q(total_elements), h_K(total_elements), h_V(total_elements);
    std::vector<float> h_O_cuda(total_elements), h_O_ref(total_elements);

    // Initialize with random data
    init_random(h_Q.data(), total_elements, 42);
    init_random(h_K.data(), total_elements, 123);
    init_random(h_V.data(), total_elements, 456);

    // Compute CPU reference
    standard_attention_cpu(
        h_Q.data(), h_K.data(), h_V.data(), h_O_ref.data(),
        tc.batch_size, tc.num_heads, tc.seq_len, tc.head_dim
    );

    // Allocate device memory
    float *d_Q, *d_K, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, bytes));
    CUDA_CHECK(cudaMalloc(&d_K, bytes));
    CUDA_CHECK(cudaMalloc(&d_V, bytes));
    CUDA_CHECK(cudaMalloc(&d_O, bytes));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    CUDA_CHECK(launch_flash_attn_v1(
        d_Q, d_K, d_V, d_O,
        tc.batch_size, tc.num_heads, tc.seq_len, tc.head_dim
    ));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_O_cuda.data(), d_O, bytes, cudaMemcpyDeviceToHost));

    // Compare
    bool passed = compare_results(h_O_cuda.data(), h_O_ref.data(), total_elements);

    // Benchmark performance
    CudaTimer timer;
    int num_warmup = 5;
    int num_iters = 20;

    for (int i = 0; i < num_warmup; i++) {
        launch_flash_attn_v1(d_Q, d_K, d_V, d_O,
                             tc.batch_size, tc.num_heads, tc.seq_len, tc.head_dim);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    timer.begin();
    for (int i = 0; i < num_iters; i++) {
        launch_flash_attn_v1(d_Q, d_K, d_V, d_O,
                             tc.batch_size, tc.num_heads, tc.seq_len, tc.head_dim);
    }
    float ms = timer.end() / num_iters;

    double flops = compute_attention_flops(tc.batch_size, tc.num_heads, tc.seq_len, tc.head_dim);
    double tflops = flops / (ms * 1e-3) / 1e12;

    printf("    Time: %.3f ms\n", ms);
    printf("    TFLOPS: %.2f\n", tflops);
    printf("    Result: %s\n", passed ? "PASSED" : "FAILED");

    // Cleanup
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));

    return passed;
}

// ============================================================================
// Kernel v2 (Tiling) test runner
// ============================================================================

bool run_test_v2(const TestCase& tc) {
    printf("\n  Testing: %s (batch=%d, heads=%d, seq=%d, dim=%d)\n",
           tc.name.c_str(), tc.batch_size, tc.num_heads, tc.seq_len, tc.head_dim);

    int64_t total_elements = (int64_t)tc.batch_size * tc.num_heads * tc.seq_len * tc.head_dim;
    size_t bytes = total_elements * sizeof(float);

    // Allocate host memory
    std::vector<float> h_Q(total_elements), h_K(total_elements), h_V(total_elements);
    std::vector<float> h_O_cuda(total_elements), h_O_ref(total_elements);

    // Initialize with random data (SAME seed as v1 for fair comparison)
    init_random(h_Q.data(), total_elements, 42);
    init_random(h_K.data(), total_elements, 123);
    init_random(h_V.data(), total_elements, 456);

    // Compute CPU reference
    standard_attention_cpu(
        h_Q.data(), h_K.data(), h_V.data(), h_O_ref.data(),
        tc.batch_size, tc.num_heads, tc.seq_len, tc.head_dim
    );

    // Allocate device memory
    float *d_Q, *d_K, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, bytes));
    CUDA_CHECK(cudaMalloc(&d_K, bytes));
    CUDA_CHECK(cudaMalloc(&d_V, bytes));
    CUDA_CHECK(cudaMalloc(&d_O, bytes));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), bytes, cudaMemcpyHostToDevice));

    // Launch kernel v2
    CUDA_CHECK(launch_flash_attn_v2(
        d_Q, d_K, d_V, d_O,
        tc.batch_size, tc.num_heads, tc.seq_len, tc.head_dim
    ));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_O_cuda.data(), d_O, bytes, cudaMemcpyDeviceToHost));

    // Compare
    bool passed = compare_results(h_O_cuda.data(), h_O_ref.data(), total_elements);

    // Benchmark performance
    CudaTimer timer;
    int num_warmup = 5;
    int num_iters = 20;

    for (int i = 0; i < num_warmup; i++) {
        launch_flash_attn_v2(d_Q, d_K, d_V, d_O,
                             tc.batch_size, tc.num_heads, tc.seq_len, tc.head_dim);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    timer.begin();
    for (int i = 0; i < num_iters; i++) {
        launch_flash_attn_v2(d_Q, d_K, d_V, d_O,
                             tc.batch_size, tc.num_heads, tc.seq_len, tc.head_dim);
    }
    float ms = timer.end() / num_iters;

    double flops = compute_attention_flops(tc.batch_size, tc.num_heads, tc.seq_len, tc.head_dim);
    double tflops = flops / (ms * 1e-3) / 1e12;

    printf("    Time: %.3f ms\n", ms);
    printf("    TFLOPS: %.2f\n", tflops);
    printf("    Result: %s\n", passed ? "PASSED" : "FAILED");

    // Cleanup
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));

    return passed;
}

// ============================================================================
// Kernel v3 (Cooperative Loading) test runner
// ============================================================================

bool run_test_v3(const TestCase& tc) {
    printf("\n  Testing: %s (batch=%d, heads=%d, seq=%d, dim=%d)\n",
           tc.name.c_str(), tc.batch_size, tc.num_heads, tc.seq_len, tc.head_dim);

    int64_t total_elements = (int64_t)tc.batch_size * tc.num_heads * tc.seq_len * tc.head_dim;
    size_t bytes = total_elements * sizeof(float);

    // Allocate host memory
    std::vector<float> h_Q(total_elements), h_K(total_elements), h_V(total_elements);
    std::vector<float> h_O_cuda(total_elements), h_O_ref(total_elements);

    // Initialize with random data (SAME seed as v1/v2 for fair comparison)
    init_random(h_Q.data(), total_elements, 42);
    init_random(h_K.data(), total_elements, 123);
    init_random(h_V.data(), total_elements, 456);

    // Compute CPU reference
    standard_attention_cpu(
        h_Q.data(), h_K.data(), h_V.data(), h_O_ref.data(),
        tc.batch_size, tc.num_heads, tc.seq_len, tc.head_dim
    );

    // Allocate device memory
    float *d_Q, *d_K, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, bytes));
    CUDA_CHECK(cudaMalloc(&d_K, bytes));
    CUDA_CHECK(cudaMalloc(&d_V, bytes));
    CUDA_CHECK(cudaMalloc(&d_O, bytes));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), bytes, cudaMemcpyHostToDevice));

    // Launch kernel v3
    CUDA_CHECK(launch_flash_attn_v3(
        d_Q, d_K, d_V, d_O,
        tc.batch_size, tc.num_heads, tc.seq_len, tc.head_dim
    ));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_O_cuda.data(), d_O, bytes, cudaMemcpyDeviceToHost));

    // Compare
    bool passed = compare_results(h_O_cuda.data(), h_O_ref.data(), total_elements);

    // Benchmark performance
    CudaTimer timer;
    int num_warmup = 5;
    int num_iters = 20;

    for (int i = 0; i < num_warmup; i++) {
        launch_flash_attn_v3(d_Q, d_K, d_V, d_O,
                             tc.batch_size, tc.num_heads, tc.seq_len, tc.head_dim);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    timer.begin();
    for (int i = 0; i < num_iters; i++) {
        launch_flash_attn_v3(d_Q, d_K, d_V, d_O,
                             tc.batch_size, tc.num_heads, tc.seq_len, tc.head_dim);
    }
    float ms = timer.end() / num_iters;

    double flops = compute_attention_flops(tc.batch_size, tc.num_heads, tc.seq_len, tc.head_dim);
    double tflops = flops / (ms * 1e-3) / 1e12;

    printf("    Time: %.3f ms\n", ms);
    printf("    TFLOPS: %.2f\n", tflops);
    printf("    Result: %s\n", passed ? "PASSED" : "FAILED");

    // Cleanup
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));

    return passed;
}

// ============================================================================
// Kernel v4 (Swizzled Shared Memory / Bank-Conflict-Free) test runner
// ============================================================================

bool run_test_v4(const TestCase& tc) {
    printf("\n  Testing: %s (batch=%d, heads=%d, seq=%d, dim=%d)\n",
           tc.name.c_str(), tc.batch_size, tc.num_heads, tc.seq_len, tc.head_dim);

    int64_t total_elements = (int64_t)tc.batch_size * tc.num_heads * tc.seq_len * tc.head_dim;
    size_t bytes = total_elements * sizeof(float);

    std::vector<float> h_Q(total_elements), h_K(total_elements), h_V(total_elements);
    std::vector<float> h_O_cuda(total_elements), h_O_ref(total_elements);

    init_random(h_Q.data(), total_elements, 42);
    init_random(h_K.data(), total_elements, 123);
    init_random(h_V.data(), total_elements, 456);

    standard_attention_cpu(
        h_Q.data(), h_K.data(), h_V.data(), h_O_ref.data(),
        tc.batch_size, tc.num_heads, tc.seq_len, tc.head_dim
    );

    float *d_Q, *d_K, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, bytes));
    CUDA_CHECK(cudaMalloc(&d_K, bytes));
    CUDA_CHECK(cudaMalloc(&d_V, bytes));
    CUDA_CHECK(cudaMalloc(&d_O, bytes));

    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(launch_flash_attn_v4(
        d_Q, d_K, d_V, d_O,
        tc.batch_size, tc.num_heads, tc.seq_len, tc.head_dim
    ));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_O_cuda.data(), d_O, bytes, cudaMemcpyDeviceToHost));

    bool passed = compare_results(h_O_cuda.data(), h_O_ref.data(), total_elements);

    // Benchmark
    CudaTimer timer;
    int num_warmup = 5;
    int num_iters = 20;

    for (int i = 0; i < num_warmup; i++) {
        launch_flash_attn_v4(d_Q, d_K, d_V, d_O,
                             tc.batch_size, tc.num_heads, tc.seq_len, tc.head_dim);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    timer.begin();
    for (int i = 0; i < num_iters; i++) {
        launch_flash_attn_v4(d_Q, d_K, d_V, d_O,
                             tc.batch_size, tc.num_heads, tc.seq_len, tc.head_dim);
    }
    float ms = timer.end() / num_iters;

    double flops = compute_attention_flops(tc.batch_size, tc.num_heads, tc.seq_len, tc.head_dim);
    double tflops = flops / (ms * 1e-3) / 1e12;

    printf("    Time: %.3f ms\n", ms);
    printf("    TFLOPS: %.2f\n", tflops);
    printf("    Result: %s\n", passed ? "PASSED" : "FAILED");

    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));

    return passed;
}

// ============================================================================
// Kernel v5 (Double Buffering) test runner
// ============================================================================

bool run_test_v5(const TestCase& tc) {
    printf("\n  Testing: %s (batch=%d, heads=%d, seq=%d, dim=%d)\n",
           tc.name.c_str(), tc.batch_size, tc.num_heads, tc.seq_len, tc.head_dim);

    int64_t total_elements = (int64_t)tc.batch_size * tc.num_heads * tc.seq_len * tc.head_dim;
    size_t bytes = total_elements * sizeof(float);

    std::vector<float> h_Q(total_elements), h_K(total_elements), h_V(total_elements);
    std::vector<float> h_O_cuda(total_elements), h_O_ref(total_elements);

    // Same seeds as v1-v4 for fair comparison
    init_random(h_Q.data(), total_elements, 42);
    init_random(h_K.data(), total_elements, 123);
    init_random(h_V.data(), total_elements, 456);

    standard_attention_cpu(
        h_Q.data(), h_K.data(), h_V.data(), h_O_ref.data(),
        tc.batch_size, tc.num_heads, tc.seq_len, tc.head_dim
    );

    float *d_Q, *d_K, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, bytes));
    CUDA_CHECK(cudaMalloc(&d_K, bytes));
    CUDA_CHECK(cudaMalloc(&d_V, bytes));
    CUDA_CHECK(cudaMalloc(&d_O, bytes));

    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), bytes, cudaMemcpyHostToDevice));

    // Launch kernel v5
    CUDA_CHECK(launch_flash_attn_v5(
        d_Q, d_K, d_V, d_O,
        tc.batch_size, tc.num_heads, tc.seq_len, tc.head_dim
    ));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_O_cuda.data(), d_O, bytes, cudaMemcpyDeviceToHost));

    bool passed = compare_results(h_O_cuda.data(), h_O_ref.data(), total_elements);

    // Benchmark
    CudaTimer timer;
    int num_warmup = 5;
    int num_iters = 20;

    for (int i = 0; i < num_warmup; i++) {
        launch_flash_attn_v5(d_Q, d_K, d_V, d_O,
                             tc.batch_size, tc.num_heads, tc.seq_len, tc.head_dim);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    timer.begin();
    for (int i = 0; i < num_iters; i++) {
        launch_flash_attn_v5(d_Q, d_K, d_V, d_O,
                             tc.batch_size, tc.num_heads, tc.seq_len, tc.head_dim);
    }
    float ms = timer.end() / num_iters;

    double flops = compute_attention_flops(tc.batch_size, tc.num_heads, tc.seq_len, tc.head_dim);
    double tflops = flops / (ms * 1e-3) / 1e12;

    printf("    Time: %.3f ms\n", ms);
    printf("    TFLOPS: %.2f\n", tflops);
    printf("    Result: %s\n", passed ? "PASSED" : "FAILED");

    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));

    return passed;
}

// ============================================================================
// Kernel v6 (cp.async Hardware Pipeline) test runner
// ============================================================================

bool run_test_v6(const TestCase& tc) {
    printf("\n  Testing: %s (batch=%d, heads=%d, seq=%d, dim=%d)\n",
           tc.name.c_str(), tc.batch_size, tc.num_heads, tc.seq_len, tc.head_dim);

    int64_t total_elements = (int64_t)tc.batch_size * tc.num_heads * tc.seq_len * tc.head_dim;
    size_t bytes = total_elements * sizeof(float);

    std::vector<float> h_Q(total_elements), h_K(total_elements), h_V(total_elements);
    std::vector<float> h_O_cuda(total_elements), h_O_ref(total_elements);

    // Same seeds as v1-v5 for fair comparison
    init_random(h_Q.data(), total_elements, 42);
    init_random(h_K.data(), total_elements, 123);
    init_random(h_V.data(), total_elements, 456);

    standard_attention_cpu(
        h_Q.data(), h_K.data(), h_V.data(), h_O_ref.data(),
        tc.batch_size, tc.num_heads, tc.seq_len, tc.head_dim
    );

    float *d_Q, *d_K, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, bytes));
    CUDA_CHECK(cudaMalloc(&d_K, bytes));
    CUDA_CHECK(cudaMalloc(&d_V, bytes));
    CUDA_CHECK(cudaMalloc(&d_O, bytes));

    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), bytes, cudaMemcpyHostToDevice));

    // Launch kernel v6
    CUDA_CHECK(launch_flash_attn_v6(
        d_Q, d_K, d_V, d_O,
        tc.batch_size, tc.num_heads, tc.seq_len, tc.head_dim
    ));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_O_cuda.data(), d_O, bytes, cudaMemcpyDeviceToHost));

    bool passed = compare_results(h_O_cuda.data(), h_O_ref.data(), total_elements);

    // Benchmark
    CudaTimer timer;
    const int num_warmup = 5;
    const int num_iters  = 20;

    for (int i = 0; i < num_warmup; i++) {
        launch_flash_attn_v6(d_Q, d_K, d_V, d_O,
                             tc.batch_size, tc.num_heads, tc.seq_len, tc.head_dim);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    timer.begin();
    for (int i = 0; i < num_iters; i++) {
        launch_flash_attn_v6(d_Q, d_K, d_V, d_O,
                             tc.batch_size, tc.num_heads, tc.seq_len, tc.head_dim);
    }
    float ms = timer.end() / num_iters;

    double flops  = compute_attention_flops(tc.batch_size, tc.num_heads, tc.seq_len, tc.head_dim);
    double tflops = flops / (ms * 1e-3) / 1e12;

    printf("    Time: %.3f ms\n", ms);
    printf("    TFLOPS: %.2f\n", tflops);
    printf("    Result: %s\n", passed ? "PASSED" : "FAILED");

    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));

    return passed;
}

// ============================================================================
// Kernel v7 (Warp Specialization) test runner
// ============================================================================

bool run_test_v7(const TestCase& tc) {
    printf("\n  Testing: %s (batch=%d, heads=%d, seq=%d, dim=%d)\n",
           tc.name.c_str(), tc.batch_size, tc.num_heads, tc.seq_len, tc.head_dim);

    int64_t total_elements = (int64_t)tc.batch_size * tc.num_heads * tc.seq_len * tc.head_dim;
    size_t bytes = total_elements * sizeof(float);

    std::vector<float> h_Q(total_elements), h_K(total_elements), h_V(total_elements);
    std::vector<float> h_O_cuda(total_elements), h_O_ref(total_elements);

    // Same seeds as v1-v6 for fair comparison
    init_random(h_Q.data(), total_elements, 42);
    init_random(h_K.data(), total_elements, 123);
    init_random(h_V.data(), total_elements, 456);

    standard_attention_cpu(
        h_Q.data(), h_K.data(), h_V.data(), h_O_ref.data(),
        tc.batch_size, tc.num_heads, tc.seq_len, tc.head_dim
    );

    float *d_Q, *d_K, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, bytes));
    CUDA_CHECK(cudaMalloc(&d_K, bytes));
    CUDA_CHECK(cudaMalloc(&d_V, bytes));
    CUDA_CHECK(cudaMalloc(&d_O, bytes));

    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), bytes, cudaMemcpyHostToDevice));

    // Launch kernel v7
    CUDA_CHECK(launch_flash_attn_v7(
        d_Q, d_K, d_V, d_O,
        tc.batch_size, tc.num_heads, tc.seq_len, tc.head_dim
    ));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_O_cuda.data(), d_O, bytes, cudaMemcpyDeviceToHost));

    bool passed = compare_results(h_O_cuda.data(), h_O_ref.data(), total_elements);

    // Benchmark
    CudaTimer timer;
    const int num_warmup = 5;
    const int num_iters  = 20;

    for (int i = 0; i < num_warmup; i++) {
        launch_flash_attn_v7(d_Q, d_K, d_V, d_O,
                             tc.batch_size, tc.num_heads, tc.seq_len, tc.head_dim);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    timer.begin();
    for (int i = 0; i < num_iters; i++) {
        launch_flash_attn_v7(d_Q, d_K, d_V, d_O,
                             tc.batch_size, tc.num_heads, tc.seq_len, tc.head_dim);
    }
    float ms = timer.end() / num_iters;

    double flops  = compute_attention_flops(tc.batch_size, tc.num_heads, tc.seq_len, tc.head_dim);
    double tflops = flops / (ms * 1e-3) / 1e12;

    printf("    Time: %.3f ms\n", ms);
    printf("    TFLOPS: %.2f\n", tflops);
    printf("    Result: %s\n", passed ? "PASSED" : "FAILED");

    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));

    return passed;
}

// ============================================================================
// Kernel v8 (Persistent Kernel) test runner
// ============================================================================

bool run_test_v8(const TestCase& tc) {
    printf("\n  Testing: %s (batch=%d, heads=%d, seq=%d, dim=%d)\n",
           tc.name.c_str(), tc.batch_size, tc.num_heads, tc.seq_len, tc.head_dim);

    int64_t total_elements = (int64_t)tc.batch_size * tc.num_heads * tc.seq_len * tc.head_dim;
    size_t bytes = total_elements * sizeof(float);

    std::vector<float> h_Q(total_elements), h_K(total_elements), h_V(total_elements);
    std::vector<float> h_O_cuda(total_elements), h_O_ref(total_elements);

    // Same seeds as v1-v7 for fair comparison
    init_random(h_Q.data(), total_elements, 42);
    init_random(h_K.data(), total_elements, 123);
    init_random(h_V.data(), total_elements, 456);

    standard_attention_cpu(
        h_Q.data(), h_K.data(), h_V.data(), h_O_ref.data(),
        tc.batch_size, tc.num_heads, tc.seq_len, tc.head_dim
    );

    float *d_Q, *d_K, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, bytes));
    CUDA_CHECK(cudaMalloc(&d_K, bytes));
    CUDA_CHECK(cudaMalloc(&d_V, bytes));
    CUDA_CHECK(cudaMalloc(&d_O, bytes));

    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), bytes, cudaMemcpyHostToDevice));

    // Launch kernel v8 (persistent)
    CUDA_CHECK(launch_flash_attn_v8(
        d_Q, d_K, d_V, d_O,
        tc.batch_size, tc.num_heads, tc.seq_len, tc.head_dim
    ));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_O_cuda.data(), d_O, bytes, cudaMemcpyDeviceToHost));

    bool passed = compare_results(h_O_cuda.data(), h_O_ref.data(), total_elements);

    // Benchmark
    CudaTimer timer;
    const int num_warmup = 5;
    const int num_iters  = 20;

    for (int i = 0; i < num_warmup; i++) {
        launch_flash_attn_v8(d_Q, d_K, d_V, d_O,
                             tc.batch_size, tc.num_heads, tc.seq_len, tc.head_dim);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    timer.begin();
    for (int i = 0; i < num_iters; i++) {
        launch_flash_attn_v8(d_Q, d_K, d_V, d_O,
                             tc.batch_size, tc.num_heads, tc.seq_len, tc.head_dim);
    }
    float ms = timer.end() / num_iters;

    double flops  = compute_attention_flops(tc.batch_size, tc.num_heads, tc.seq_len, tc.head_dim);
    double tflops = flops / (ms * 1e-3) / 1e12;

    printf("    Time: %.3f ms\n", ms);
    printf("    TFLOPS: %.2f\n", tflops);
    printf("    Result: %s\n", passed ? "PASSED" : "FAILED");

    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));

    return passed;
}

// ============================================================================
// Kernel v9 (GQA) test runner
// Supports MHA (H_kv == H_q), MQA (H_kv == 1), and GQA (1 < H_kv < H_q).
// ============================================================================

// CPU reference for GQA: same as standard_attention_cpu but K/V have H_kv heads.
void standard_gqa_attention_cpu(
    const float* Q,   // [B, H_q,  N, D]
    const float* K,   // [B, H_kv, N, D]
    const float* V,   // [B, H_kv, N, D]
    float*       O,   // [B, H_q,  N, D]
    int batch_size, int num_heads_q, int num_heads_kv,
    int seq_len, int head_dim
) {
    const float scale = 1.0f / sqrtf((float)head_dim);
    const int group_size = num_heads_q / num_heads_kv;

    for (int b = 0; b < batch_size; b++) {
        for (int h_q = 0; h_q < num_heads_q; h_q++) {
            const int h_kv = h_q / group_size;
            const int64_t q_off  = (int64_t)b * num_heads_q  * seq_len * head_dim
                                 + (int64_t)h_q  * seq_len * head_dim;
            const int64_t kv_off = (int64_t)b * num_heads_kv * seq_len * head_dim
                                 + (int64_t)h_kv * seq_len * head_dim;

            for (int i = 0; i < seq_len; i++) {
                // Compute scores for query row i
                std::vector<float> scores(seq_len);
                float max_score = -1e30f;
                for (int j = 0; j < seq_len; j++) {
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; d++) {
                        dot += Q[q_off  + i * head_dim + d]
                             * K[kv_off + j * head_dim + d];
                    }
                    scores[j] = dot * scale;
                    max_score = fmaxf(max_score, scores[j]);
                }

                float sum_exp = 0.0f;
                for (int j = 0; j < seq_len; j++) {
                    scores[j] = expf(scores[j] - max_score);
                    sum_exp += scores[j];
                }

                const float inv_sum = 1.0f / sum_exp;
                for (int d = 0; d < head_dim; d++) {
                    float val = 0.0f;
                    for (int j = 0; j < seq_len; j++) {
                        val += scores[j] * V[kv_off + j * head_dim + d];
                    }
                    O[q_off + i * head_dim + d] = val * inv_sum;
                }
            }
        }
    }
}

// GQA test case struct
struct GqaTestCase {
    int batch_size;
    int num_heads_q;
    int num_heads_kv;
    int seq_len;
    int head_dim;
    std::string name;
};

bool run_test_v9_gqa(const GqaTestCase& tc) {
    printf("\n  GQA Testing: %s "
           "(batch=%d, H_q=%d, H_kv=%d, seq=%d, dim=%d, group=%d)\n",
           tc.name.c_str(),
           tc.batch_size, tc.num_heads_q, tc.num_heads_kv,
           tc.seq_len, tc.head_dim,
           tc.num_heads_q / tc.num_heads_kv);

    // Q has num_heads_q heads; K/V have num_heads_kv heads
    const int64_t q_elems  = (int64_t)tc.batch_size * tc.num_heads_q  * tc.seq_len * tc.head_dim;
    const int64_t kv_elems = (int64_t)tc.batch_size * tc.num_heads_kv * tc.seq_len * tc.head_dim;

    std::vector<float> h_Q(q_elems), h_K(kv_elems), h_V(kv_elems);
    std::vector<float> h_O_cuda(q_elems), h_O_ref(q_elems);

    init_random(h_Q.data(), (int)q_elems,  42);
    init_random(h_K.data(), (int)kv_elems, 123);
    init_random(h_V.data(), (int)kv_elems, 456);

    // CPU reference (GQA-aware)
    standard_gqa_attention_cpu(
        h_Q.data(), h_K.data(), h_V.data(), h_O_ref.data(),
        tc.batch_size, tc.num_heads_q, tc.num_heads_kv,
        tc.seq_len, tc.head_dim
    );

    // Allocate device memory
    float *d_Q, *d_K, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, q_elems  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, kv_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, kv_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_O, q_elems  * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), q_elems  * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), kv_elems * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), kv_elems * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel v9 (GQA)
    CUDA_CHECK(launch_flash_attn_v9(
        d_Q, d_K, d_V, d_O,
        tc.batch_size, tc.num_heads_q, tc.num_heads_kv,
        tc.seq_len, tc.head_dim
    ));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_O_cuda.data(), d_O, q_elems * sizeof(float),
                          cudaMemcpyDeviceToHost));

    bool passed = compare_results(h_O_cuda.data(), h_O_ref.data(), (int)q_elems);

    // Benchmark
    CudaTimer timer;
    const int num_warmup = 5;
    const int num_iters  = 20;

    for (int i = 0; i < num_warmup; i++) {
        launch_flash_attn_v9(d_Q, d_K, d_V, d_O,
                             tc.batch_size, tc.num_heads_q, tc.num_heads_kv,
                             tc.seq_len, tc.head_dim);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    timer.begin();
    for (int i = 0; i < num_iters; i++) {
        launch_flash_attn_v9(d_Q, d_K, d_V, d_O,
                             tc.batch_size, tc.num_heads_q, tc.num_heads_kv,
                             tc.seq_len, tc.head_dim);
    }
    float ms = timer.end() / num_iters;

    // FLOPs: same as MHA (same number of attention ops, just K/V data is shared)
    double flops  = compute_attention_flops(tc.batch_size, tc.num_heads_q,
                                            tc.seq_len, tc.head_dim);
    double tflops = flops / (ms * 1e-3) / 1e12;

    printf("    Time: %.3f ms\n", ms);
    printf("    TFLOPS: %.2f\n", tflops);
    printf("    KV memory ratio: %.2fx vs MHA (group_size=%d)\n",
           1.0f / (tc.num_heads_q / tc.num_heads_kv),
           tc.num_heads_q / tc.num_heads_kv);
    printf("    Result: %s\n", passed ? "PASSED" : "FAILED");

    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));

    return passed;
}

int main() {
    printf("============================================================\n");
    printf("Flash Attention Kernels - Correctness & Performance Test\n");
    printf("Kernel v1: Naive (HBM)       |  Kernel v2: Tiling (Shared Mem)\n");
    printf("Kernel v3: Cooperative (8q)  |  Kernel v4: Swizzle (Bank-CF-Free)\n");
    printf("Kernel v5: Double Buffering  |  Kernel v6: cp.async (Ampere HW pipeline)\n");
    printf("Kernel v7: Warp Specialization (2 producer + 6 consumer warps)\n");
    printf("Kernel v8: Persistent Kernel (global work queue, one kernel launch)\n");
    printf("Kernel v9: GQA Flash Attention (MHA/MQA/GQA unified — LLaMA-3 style)\n");
    printf("============================================================\n");

    // Print GPU info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("\nGPU: %s\n", prop.name);
    printf("SM count: %d\n", prop.multiProcessorCount);
    printf("SMEM per block: %zu KB\n", prop.sharedMemPerBlock / 1024);

    // Define test cases
    std::vector<TestCase> tests = {
        {1, 1, 64, 64, "Small"},
        {1, 1, 128, 64, "Medium"},
        {1, 1, 256, 64, "Large"},
        {1, 1, 512, 64, "Seq-512"},
        {1, 1, 1024, 64, "Seq-1024"},
        {1, 4, 128, 64, "Multi-head"},
        {2, 4, 128, 64, "Batch+Multi-head"},
        {1, 8, 512, 128, "LLM-style"},
    };

    int passed_v1 = 0, passed_v2 = 0;
    int total = static_cast<int>(tests.size());

    printf("\n========== KERNEL V1 (Naive) ==========\n");
    for (const auto& tc : tests) {
        if (run_test(tc)) {
            passed_v1++;
        }
    }

    printf("\n========== KERNEL V2 (Tiling) ==========\n");
    // Also test v2 (same test cases)
    for (const auto& tc : tests) {
        // Modify run_test to accept kernel selector — instead, we copy the logic inline
        // For brevity, we reuse run_test for v1 above and call v2 directly below.
        // Note: This requires duplicating the test loop. A cleaner approach is
        // to modify run_test to take a kernel version parameter.
        // For now, we add a separate loop for v2.
    }
    // Re-run v2 tests by calling run_test_v2 (defined below)
    for (const auto& tc : tests) {
        if (run_test_v2(tc)) {
            passed_v2++;
        }
    }

    printf("\n========== KERNEL V3 (Cooperative Loading) ==========\n");
    int passed_v3 = 0;
    for (const auto& tc : tests) {
        if (run_test_v3(tc)) {
            passed_v3++;
        }
    }

    printf("\n========== KERNEL V4 (Swizzled Shared Memory, Bank-Conflict-Free) ==========\n");
    int passed_v4 = 0;
    for (const auto& tc : tests) {
        if (run_test_v4(tc)) {
            passed_v4++;
        }
    }

    printf("\n========== KERNEL V5 (Double Buffering, Software Pipeline) ==========\n");
    int passed_v5 = 0;
    for (const auto& tc : tests) {
        if (run_test_v5(tc)) {
            passed_v5++;
        }
    }

    printf("\n========== KERNEL V6 (cp.async Hardware Pipeline, Ampere+) ==========\n");
    int passed_v6 = 0;
    for (const auto& tc : tests) {
        if (run_test_v6(tc)) {
            passed_v6++;
        }
    }

    printf("\n========== KERNEL V7 (Warp Specialization, Producer+Consumer) ==========\n");
    int passed_v7 = 0;
    for (const auto& tc : tests) {
        if (run_test_v7(tc)) {
            passed_v7++;
        }
    }

    printf("\n========== KERNEL V8 (Persistent Kernel, Global Work Queue) ==========\n");
    int passed_v8 = 0;
    for (const auto& tc : tests) {
        if (run_test_v8(tc)) {
            passed_v8++;
        }
    }

    // ── Kernel V9: GQA tests ─────────────────────────────────────────────────
    // Test three modes: MHA (H_kv==H_q), GQA-4 (group=4), MQA (H_kv==1)
    // Seq len kept at multiples of Q_BLOCK_V9=8 for clean tile alignment.
    printf("\n========== KERNEL V9 (GQA: MHA / GQA / MQA modes) ==========\n");
    std::vector<GqaTestCase> gqa_tests = {
        // MHA mode (group_size=1): backward-compatible with K1-K8
        {1, 1, 1, 64,  64,  "MHA  (H_q=1,  H_kv=1,  group=1)"},
        {1, 1, 1, 128, 64,  "MHA  (H_q=1,  H_kv=1,  seq=128)"},
        {1, 4, 4, 128, 64,  "MHA  (H_q=4,  H_kv=4,  group=1)"},
        {2, 4, 4, 128, 64,  "MHA  (B=2,    H_q=4,   H_kv=4 )"},
        // GQA mode (LLaMA-3 style: H_q=32, H_kv=8, group=4 → simulated with smaller H)
        {1, 8, 2, 64,  64,  "GQA  (H_q=8,  H_kv=2,  group=4)"},
        {1, 8, 2, 128, 64,  "GQA  (H_q=8,  H_kv=2,  seq=128)"},
        {1, 8, 4, 256, 64,  "GQA  (H_q=8,  H_kv=4,  group=2)"},
        // MQA mode (H_kv=1, maximum KV cache reduction)
        {1, 4, 1, 64,  64,  "MQA  (H_q=4,  H_kv=1,  group=4)"},
        {1, 8, 1, 128, 128, "MQA  (H_q=8,  H_kv=1,  dim=128)"},
        {2, 8, 2, 256, 64,  "GQA  (B=2,    H_q=8,   H_kv=2, seq=256)"},
    };

    int passed_v9 = 0;
    const int total_v9 = (int)gqa_tests.size();
    for (const auto& tc : gqa_tests) {
        if (run_test_v9_gqa(tc)) {
            passed_v9++;
        }
    }

    printf("\n============================================================\n");
    printf("KERNEL V1 (Naive):                   %d / %d tests passed\n", passed_v1, total);
    printf("KERNEL V2 (Tiling):                  %d / %d tests passed\n", passed_v2, total);
    printf("KERNEL V3 (Cooperative Loading):     %d / %d tests passed\n", passed_v3, total);
    printf("KERNEL V4 (Swizzled, Bank-CF-Free):  %d / %d tests passed\n", passed_v4, total);
    printf("KERNEL V5 (Double Buffering):        %d / %d tests passed\n", passed_v5, total);
    printf("KERNEL V6 (cp.async HW Pipeline):    %d / %d tests passed\n", passed_v6, total);
    printf("KERNEL V7 (Warp Specialization):     %d / %d tests passed\n", passed_v7, total);
    printf("KERNEL V8 (Persistent Kernel):       %d / %d tests passed\n", passed_v8, total);
    printf("KERNEL V9 (GQA/MQA/MHA):             %d / %d tests passed\n", passed_v9, total_v9);
    const bool all_passed = (passed_v1 == total   && passed_v2 == total &&
                              passed_v3 == total   && passed_v4 == total &&
                              passed_v5 == total   && passed_v6 == total &&
                              passed_v7 == total   && passed_v8 == total &&
                              passed_v9 == total_v9);
    if (all_passed) {
        printf("ALL TESTS PASSED (9 kernels, including GQA)\n");
    } else {
        printf("SOME TESTS FAILED\n");
    }
    printf("============================================================\n");

    return all_passed ? 0 : 1;
}
