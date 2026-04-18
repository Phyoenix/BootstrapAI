# ComputeBench
> 轻量级计算性能基准测试与分析工具
> 
> **目标**: 从CPU开始，逐步扩展到GPU (CUDA/Vulkan/OpenCL)、芯片级性能分析

## 产品定位

一个可扩展的硬件性能测试框架，专注：
- 计算密集型任务性能测量
- 历史数据追踪与趋势分析
- 可视化报告生成
- 为软硬件协同优化提供数据支撑

## 技术栈

- **语言**: C++ (核心性能测试) + Python (分析脚本)
- **架构**: 模块化插件系统，支持新测试类型动态添加
- **输出**: JSON数据 + Markdown报告 + 可选HTML可视化

## 当前版本: v0.1.0 - CPU基础版

### 已实现功能
- [x] CPU信息检测 (型号、核心数、缓存)
- [x] 内存带宽测试
- [x] 浮点运算性能测试 (GFLOPS)
- [x] 整数运算性能测试 (GIOPS)
- [x] 结果JSON序列化
- [x] 历史数据对比

### 目录结构

```
ComputeBench/
├── src/
│   ├── core/           # 核心框架
│   │   ├── benchmark.h # 基准测试基类
│   │   ├── result.h    # 结果数据结构
│   │   └── reporter.h  # 报告生成
│   ├── tests/          # 具体测试实现
│   │   ├── cpu_info.cpp
│   │   ├── memory_bw.cpp
│   │   ├── fp_ops.cpp
│   │   └── int_ops.cpp
│   └── main.cpp        # 入口
├── scripts/            # 分析工具
│   ├── analyze.py      # 数据分析
│   └── plot_history.py # 历史趋势图
├── results/            # 测试结果存储
│   └── history.json    # 历史数据
├── reports/            # 生成报告
│   └── latest.md
└── Makefile
```

## 编译与运行

```bash
cd ComputeBench
make          # 编译
./bin/cb      # 运行所有测试
./bin/cb --test=fp_ops  # 运行指定测试
```

## 迭代路线图

### v0.2.0 - 内存与缓存
- [ ] L1/L2/L3缓存延迟测试
- [ ] 内存时序分析
- [ ] NUMA感知测试

### v0.3.0 - GPU准备
- [ ] OpenCL检测与支持
- [ ] 基础GPU内存带宽测试
- [ ] 核函数启动开销测量

### v0.4.0 - CUDA支持
- [ ] CUDA设备检测
- [ ] cuBLAS性能测试
- [ ] 自定义CUDA核函数测试

### v0.5.0 - Vulkan计算
- [ ] Vulkan计算着色器支持
- [ ] GPU通用计算对比 (CUDA vs Vulkan vs OpenCL)

### v1.0.0 - 芯片级分析
- [ ] 指令级分析 (IPC, 流水线)
- [ ] 功耗与性能关联
- [ ] 硬件事件计数器 (PMU) 支持

## 设计理念

1. **准确性**: 预热、多次采样、剔除异常值
2. **可重复性**: 固定随机种子，控制频率缩放
3. **可扩展性**: 插件架构，新测试类型易于添加
4. **实用性**: 不仅测峰值性能，也测真实工作负载

---

**Created by**: Kraber (AI Evolver)  
**Created for**: BootstrapAI Project  
**Version**: v0.1.0  
**Last Updated**: 2026-04-19
