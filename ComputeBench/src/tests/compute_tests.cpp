#include "compute_tests.h"
#include <vector>
#include <random>
#include <chrono>
#include <cstring>
#include <omp.h>

namespace cb {

// 浮点运算测试
REGISTER_BENCHMARK(FPOperationsTest);

void FPOperationsTest::warmup() {
    volatile double x = 0;
    for (int i = 0; i < 1000000; ++i) {
        x += i * 0.5;
    }
}

double FPOperationsTest::execute() {
    const int64_t N = 100000000;  // 1亿次操作
    
    // 使用OpenMP并行
    double sum = 0.0;
    auto start = std::chrono::high_resolution_clock::now();
    
    #pragma omp parallel for reduction(+:sum)
    for (int64_t i = 0; i < N; ++i) {
        // 混合浮点运算: 乘加、sqrt、除法
        double x = static_cast<double>(i) * 0.5 + 1.0;
        double y = x * x + 1.0;
        double z = y / (x + 0.1);
        sum += z;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // 每次迭代约5个浮点操作
    double gflops = (N * 5.0) / duration.count();
    return gflops;
}

// 整数运算测试
REGISTER_BENCHMARK(IntOperationsTest);

void IntOperationsTest::warmup() {
    volatile int64_t x = 0;
    for (int i = 0; i < 1000000; ++i) {
        x += i * 3;
    }
}

double IntOperationsTest::execute() {
    const int64_t N = 200000000;  // 2亿次操作
    
    int64_t sum = 0;
    auto start = std::chrono::high_resolution_clock::now();
    
    #pragma omp parallel for reduction(+:sum)
    for (int64_t i = 0; i < N; ++i) {
        // 混合整数运算
        int64_t x = i * 3 + 7;
        int64_t y = x >> 2;
        int64_t z = y * y;
        sum += z;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // 每次迭代约4个整数操作
    double giops = (N * 4.0) / duration.count();
    return giops;
}

// 内存带宽测试
REGISTER_BENCHMARK(MemoryBandwidthTest);

void MemoryBandwidthTest::warmup() {
    const size_t size = 100 * 1024 * 1024;  // 100MB
    std::vector<char> buf(size);
    std::memset(buf.data(), 0, size);
}

double MemoryBandwidthTest::execute() {
    const size_t size = 500 * 1024 * 1024;  // 500MB
    const int iterations = 10;
    
    std::vector<char> src(size);
    std::vector<char> dst(size);
    
    // 初始化源数据
    std::memset(src.data(), 'A', size);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // 多次复制以测量带宽
    for (int i = 0; i < iterations; ++i) {
        std::memcpy(dst.data(), src.data(), size);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // 计算带宽: 每次复制2倍size(读+写), 乘以迭代次数
    double total_bytes = static_cast<double>(size) * 2.0 * iterations;
    double seconds = duration.count() / 1000000.0;
    double gbps = (total_bytes / seconds) / (1024.0 * 1024.0 * 1024.0);
    
    return gbps;
}

} // namespace cb
