#include "benchmark.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace cb {

TestResult Benchmark::run(int samples) {
    warmup();
    
    std::vector<double> values;
    values.reserve(samples);
    
    auto start = std::chrono::steady_clock::now();
    
    for (int i = 0; i < samples; ++i) {
        values.push_back(execute());
    }
    
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // 计算统计值
    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    double mean = sum / values.size();
    
    double min_val = *std::min_element(values.begin(), values.end());
    double max_val = *std::max_element(values.begin(), values.end());
    
    // 计算标准差
    double sq_sum = 0.0;
    for (double v : values) {
        sq_sum += (v - mean) * (v - mean);
    }
    double stddev = std::sqrt(sq_sum / values.size());
    
    return TestResult{
        name(),
        unit(),
        mean,
        min_val,
        max_val,
        stddev,
        samples,
        duration,
        ""
    };
}

// 单例实现
BenchmarkRegistry& BenchmarkRegistry::instance() {
    static BenchmarkRegistry registry;
    return registry;
}

void BenchmarkRegistry::registerBenchmark(Benchmark* bench) {
    benchmarks.push_back(bench);
}

std::vector<Benchmark*> BenchmarkRegistry::getAll() const {
    return benchmarks;
}

Benchmark* BenchmarkRegistry::getByName(const std::string& name) const {
    for (auto* bench : benchmarks) {
        if (bench->name() == name) {
            return bench;
        }
    }
    return nullptr;
}

} // namespace cb
