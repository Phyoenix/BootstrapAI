/**
 * Test Correctness for Flash Attention CUDA Kernels
 *
 * Compares CUDA kernel output against CPU reference implementation.
 * Tests kernel v1 (naive), v2 (tiling), v3 (cooperative), v4 (swizzle),
 * v5 (double buffering), v6 (cp.async hardware pipeline).
 *
 * Build:
 *   nvcc -O3 -arch=sm_89 -I../include test_correctness.cu \
 *     ../kernels/kernel_01_naive.cu ../kernels/kernel_02_tiling.cu \
 *     ../kernels/kernel_03_cooperative.cu ../kernels/kernel_04_swizzle.cu \
 *     ../kernels/kernel_05_double_buffer.cu ../kernels/kernel_06_cp_async.cu \
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



int main() {
    printf("============================================================\n");
    printf("Flash Attention Kernels - Correctness & Performance Test\n");
    printf("Kernel v1: Naive (HBM)       |  Kernel v2: Tiling (Shared Mem)\n");
    printf("Kernel v3: Cooperative (8q)  |  Kernel v4: Swizzle (Bank-CF-Free)\n");
    printf("Kernel v5: Double Buffering  |  Kernel v6: cp.async (Ampere HW pipeline)\n");
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

    printf("\n============================================================\n");
    printf("KERNEL V1 (Naive):                   %d / %d tests passed\n", passed_v1, total);
    printf("KERNEL V2 (Tiling):                  %d / %d tests passed\n", passed_v2, total);
    printf("KERNEL V3 (Cooperative Loading):     %d / %d tests passed\n", passed_v3, total);
    printf("KERNEL V4 (Swizzled, Bank-CF-Free):  %d / %d tests passed\n", passed_v4, total);
    printf("KERNEL V5 (Double Buffering):        %d / %d tests passed\n", passed_v5, total);
    printf("KERNEL V6 (cp.async HW Pipeline):    %d / %d tests passed\n", passed_v6, total);
    if (passed_v1 == total && passed_v2 == total && passed_v3 == total &&
        passed_v4 == total && passed_v5 == total && passed_v6 == total) {
    printf("ALL TESTS PASSED (all 6 kernels)\n");
    } else {
        printf("SOME TESTS FAILED\n");
    }
    printf("============================================================\n");

    return (passed_v1 == total && passed_v2 == total &&
            passed_v3 == total && passed_v4 == total &&
            passed_v5 == total && passed_v6 == total) ? 0 : 1;
}
