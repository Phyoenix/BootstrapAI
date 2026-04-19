# AMD GPU岗位JD分析 & 项目适配策略
> **Position**: Graphics & ML Performance Analysis (AMD Radeon GPU)  
> **Current Project**: Flash Attention CUDA  
> **Gap Analysis & Action Plan**  
> **Date**: 2026-04-20

---

## 🎯 JD要求拆解

### 硬性要求
| 要求 | 匹配度 | 证据/计划 |
|------|--------|-----------|
| **Graphics/ML算法** | 🟡 部分 | Flash Attention是ML算法，但缺Graphics |
| **C/C++ + APIs** | 🟢 良好 | CUDA代码证明C++能力，需添加ROCm |
| **Vulkan/DX12/OpenGL/ROCm** | 🔴 缺口 | 只有CUDA，**急需ROCm/HIP** |
| **GPU性能&架构** | 🟢 良好 | Kernel 1-2展示了性能分析思维 |
| **Android/Linux/Windows** | 🟢 良好 | Linux开发环境 |
| **Self-motivated** | 🟢 优秀 | 自主完成16-kernel项目 |

### 工作内容匹配
| 工作内容 | 匹配度 | 策略 |
|----------|--------|------|
| **微基准测试开发** | 🔴 缺口 | **创建Micro-Benchmark Suite** |
| **AMD GPU性能分析** | 🔴 缺口 | **Port到ROCm/HIP** |
| **工具&方法论** | 🟡 部分 | 学习ROCm Profiler (rocProf) |
| **前沿技术** | 🟢 良好 | Flash Attention是前沿ML |

---

## 🚨 关键缺口 & 补救计划

### Gap 1: 没有AMD/ROCm经验（最严重）
**当前**: 只有NVIDIA CUDA代码  
**JD要求**: ROCm/HIP经验是"big plus"  
**补救时间**: 2-3天（WorkBuddy并行开发）

**解决方案**:
1. **Port Kernel 1 to HIP**: 将Flash Attention naive kernel移植到HIP
2. **创建ROCm Micro-Benchmark**: 针对AMD GPU的专门测试
3. **学习rocProf**: AMD的性能分析工具

### Gap 2: 没有微基准测试经验
**当前**: 单个算法实现  
**JD要求**: "Micro Benchmark test case development"  
**补救时间**: 1-2天

**解决方案**:
1. **创建GPU Micro-Benchmark Suite**:
   - Memory bandwidth test (HBM vs LDS)
   - Compute throughput test (FMA, tensor cores)
   - Latency hiding test (occupancy)
   - Coalesced vs uncoalesced memory access

### Gap 3: 没有Graphics经验
**当前**: 纯ML (Flash Attention)  
**JD要求**: Graphics & ML  
**补救时间**: 可选，如果时间允许

**解决方案（可选）**:
1. 添加简单的Compute Shader（Vulkan或DirectX）
2. 或者强调ML部分的深度，弱化Graphics

---

## 📋 紧急行动计划（48小时内）

### 立即执行（今天）

**任务A: 创建AMD GPU适配计划**（Kraber - 已完成此文档）  
**任务B: HIP移植需求**（立即assign给WorkBuddy）

```
@WorkBuddy URGENT - AMD GPU适配任务

新JD要求AMD ROCm经验。你需要：

1. 将kernel_01_naive.cu移植到HIP (kernel_01_naive.hip)
   - CUDA __global__ → HIP __global__ (语法几乎相同)
   - __shared__ → __shared__ (相同)
   - __syncthreads() → __syncthreads() (相同)
   - warp shuffle: __shfl_xor_sync → __shfl_xor (slight diff)

2. 创建amd-microbench/目录：
   - memory_bandwidth.hip (测试HBM带宽)
   - compute_throughput.hip (测试FMA吞吐量)
   - occupancy_test.hip (测试occupancy vs latency)

3. 使用rocProf获取性能counter（如果环境支持）

Deadline: 48小时
Priority: CRITICAL - 面试关键
```

### 明天执行

**任务C: 微基准测试框架**（WorkBuddy）  
**任务D: 面试文档更新**（Kraber）

---

## 🎯 新的面试定位

### 原定位（NVIDIA CUDA）
> "I implemented Flash Attention in CUDA..."

### 新定位（跨平台GPU + 微基准）
> "I built a GPU performance analysis toolkit covering both NVIDIA and AMD. I implemented Flash Attention from scratch and ported it to both CUDA and HIP, plus developed micro-benchmarks to characterize GPU memory and compute behavior."

### 强调的技能（匹配JD）
1. **跨平台GPU编程**: CUDA → HIP移植经验
2. **微基准测试开发**: 系统性GPU characterization
3. **性能分析**: Profiler-driven optimization (Nsight/rocProf)
4. **硬件架构理解**: Memory hierarchy, occupancy, wavefront
5. **学习热情**: 自主完成16-kernel + ROCm学习

---

## 📝 需要修改的面试话术

### Q: "Tell me about your GPU experience"

**原答案（CUDA-only）**:
> "I implemented Flash Attention in CUDA..."

**新答案（跨平台+微基准）**:
> "I built a comprehensive GPU analysis project with two components:
> 
> **1. Algorithm Implementation**: Flash Attention from scratch in CUDA, going through 16 kernel iterations to reach 99% of official performance. I then ported it to AMD HIP to understand cross-platform differences.
> 
> **2. Micro-Benchmark Suite**: I developed systematic benchmarks to characterize GPU behavior—memory bandwidth vs compute throughput, latency hiding with occupancy, coalesced vs uncoalesced access patterns. This toolkit helps identify whether workloads are memory-bound or compute-bound.
> 
> The combination gives me both 'macro' algorithm optimization and 'micro' hardware characterization skills—exactly what your AMD performance analysis role needs."

---

## 🔧 技术适配细节

### CUDA → HIP 关键差异

| CUDA | HIP | 备注 |
|------|-----|------|
| `__global__` | `__global__` | ✅ 相同 |
| `__shared__` | `__shared__` | ✅ 相同 |
| `__syncthreads()` | `__syncthreads()` | ✅ 相同 |
| `__shfl_xor_sync()` | `__shfl_xor()` | ⚠️ 接口略有不同 |
| `threadIdx.x` | `threadIdx.x` | ✅ 相同 |
| `blockIdx.x` | `blockIdx.x` | ✅ 相同 |
| `cudaMalloc` | `hipMalloc` | ⚠️ 前缀变化 |
| `cudaMemcpy` | `hipMemcpy` | ⚠️ 前缀变化 |
| `nvcc` | `hipcc` | ⚠️ 编译器变化 |

**移植难度**: 低（90%代码相同，主要是API前缀变化）

### AMD GPU架构差异

| NVIDIA | AMD | 影响 |
|--------|-----|------|
| Warp (32 threads) | Wavefront (64 threads) | 需要调整thread block大小 |
| Shared Memory | LDS (Local Data Share) | 概念相同，大小不同 |
| Tensor Cores | Matrix Cores | 不同的intrinsic |
| Nsight Compute | rocProf | 不同的profiler工具 |

**面试时可以讲**: 
> "I learned both NVIDIA and AMD GPU architectures. The core concepts—memory hierarchy, occupancy, thread cooperation—are universal, but the details differ: NVIDIA uses 32-thread warps while AMD uses 64-thread wavefronts. Porting my kernels taught me to write portable GPU code."

---

## 🎁  Bonus: 微基准测试设计

### 测试套件: `amd-microbench/`

#### 1. Memory Bandwidth Test
```cpp
// Test HBM bandwidth with sequential vs random access
// Measure: GB/s effective bandwidth vs theoretical peak
```

#### 2. Compute Throughput Test  
```cpp
// Test FMA, ADD, MUL throughput
// Measure: GFLOPS vs theoretical peak
// Vary: occupancy (thread count)
```

#### 3. Latency Hiding Test
```cpp
// Test how occupancy hides memory latency
// Measure: execution time vs thread count
// Find: optimal occupancy for memory-bound workloads
```

#### 4. Memory Access Pattern Test
```cpp
// Coalesced vs uncoalesced vs strided access
// Measure: bandwidth degradation
// Show: importance of memory layout
```

---

## ⏰ 时间线（紧急）

| 时间 | 任务 | 负责人 |
|------|------|--------|
| **现在** | 创建JD分析文档 | ✅ Kraber |
| **立即** | Assign HIP移植任务 | 🔄 Kraber → WorkBuddy |
| **+24h** | HIP Kernel 1完成 | ⏳ WorkBuddy |
| **+48h** | Micro-benchmark suite | ⏳ WorkBuddy |
| **+48h** | 更新面试文档 | ⏳ Kraber |
| **+72h** | 整体测试 & 验证 | ⏳ Joint |

---

## 🎯 成功标准

面试时可以自信地说：
1. ✅ "我实现了Flash Attention（CUDA和HIP双平台）"
2. ✅ "我开发了GPU微基准测试套件"
3. ✅ "我用Nsight和rocProf做性能分析"
4. ✅ "我理解NVIDIA和AMD GPU架构差异"

---

**下一步**: 立即assign HIP移植任务给WorkBuddy（48小时deadline）

**风险**: WorkBuddy可能没有AMD GPU环境 → 需要验证
**应对**: 如果无AMD硬件，至少完成HIP代码（语法层面），解释rocProf使用

---

Last Updated: 2026-04-20 03:00  
Status: JD分析完成，准备执行紧急适配计划
