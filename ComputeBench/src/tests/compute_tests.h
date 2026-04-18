#pragma once
#include "../core/benchmark.h"

namespace cb {

// 浮点运算性能测试
class FPOperationsTest : public Benchmark {
public:
    std::string name() const override { return "fp_ops"; }
    std::string description() const override { 
        return "Floating point operations performance (GFLOPS)"; 
    }
    
protected:
    double execute() override;
    void warmup() override;
    std::string unit() const override { return "GFLOPS"; }
};

// 整数运算性能测试
class IntOperationsTest : public Benchmark {
public:
    std::string name() const override { return "int_ops"; }
    std::string description() const override { 
        return "Integer operations performance (GIOPS)"; 
    }
    
protected:
    double execute() override;
    void warmup() override;
    std::string unit() const override { return "GIOPS"; }
};

// 内存带宽测试
class MemoryBandwidthTest : public Benchmark {
public:
    std::string name() const override { return "memory_bw"; }
    std::string description() const override { 
        return "Memory bandwidth test (GB/s)"; 
    }
    
protected:
    double execute() override;
    void warmup() override;
    std::string unit() const override { return "GB/s"; }
};

} // namespace cb
