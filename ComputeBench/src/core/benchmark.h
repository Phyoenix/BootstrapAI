#ifndef CB_BENCHMARK_H
#define CB_BENCHMARK_H

#include <string>
#include <vector>
#include <chrono>
#include <functional>

namespace cb {

// 测试结果结构
struct TestResult {
    std::string name;
    std::string unit;
    double value;           // 主要指标值
    double min, max;        // 最小/最大值
    double stddev;          // 标准差
    int samples;            // 采样次数
    std::chrono::milliseconds duration;
    std::string details;    // 额外信息
};

// 基准测试基类
class Benchmark {
public:
    virtual ~Benchmark() = default;
    
    virtual std::string name() const = 0;
    virtual std::string description() const = 0;
    
    // 运行测试，返回结果
    TestResult run(int samples = 5);
    
    // 是否可用（例如GPU测试在无GPU时返回false）
    virtual bool isAvailable() const { return true; }

protected:
    // 子类实现：单次测试执行
    virtual double execute() = 0;
    
    // 预热（避免冷缓存影响）
    virtual void warmup() {}
    
    // 获取单位
    virtual std::string unit() const = 0;
};

// 测试注册表
class BenchmarkRegistry {
public:
    static BenchmarkRegistry& instance();
    
    void registerBenchmark(Benchmark* bench);
    std::vector<Benchmark*> getAll() const;
    Benchmark* getByName(const std::string& name) const;

private:
    std::vector<Benchmark*> benchmarks;
};

// 自动注册宏
#define REGISTER_BENCHMARK(ClassName) \
    static ClassName* _cb_##ClassName##_instance = []() { \
        static ClassName instance; \
        BenchmarkRegistry::instance().registerBenchmark(&instance); \
        return &instance; \
    }();

} // namespace cb

#endif // CB_BENCHMARK_H
