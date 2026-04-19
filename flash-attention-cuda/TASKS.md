# Flash Attention Project - Task Assignments (AMD JD适配版)
> **Manager**: Kraber  
> **Executor**: WorkBuddy  
> **Last Updated**: 2026-04-20 03:00  
> **Status**: 🔴 URGENT - AMD GPU适配任务

---

## 🚨 紧急：新JD要求AMD ROCm经验

**背景**: 目标岗位转为AMD Radeon GPU性能分析  
**关键缺口**: 
- ❌ 无ROCm/HIP代码  
- ❌ 无微基准测试经验  
- ❌ 无AMD GPU工具经验  

**补救策略**: 48小时内完成HIP移植 + 微基准套件  

---

## ✅ COMPLETED: Task 1-2 (CUDA版本)

| 任务 | 状态 | 成果 |
|------|------|------|
| Task 1 | ✅ Done | CUDA Naive Kernel, 0.51 TFLOPS |
| Task 2 | ✅ Done | CUDA Tiling Kernel, 学习价值 |
| INTERVIEW_PREP.md | ✅ Done | 面试Q&A总结 |
| JD_ANALYSIS_AMD.md | ✅ Done | 缺口分析 |

---

## 🔴 URGENT: Task 3 - HIP移植 [ACTIVE]

**Status**: 🔴 CRITICAL - START NOW  
**Assignee**: WorkBuddy  
**Priority**: P0 (面试决胜任务)  
**Deadline**: 48小时 (2026-04-22 03:00)  
**Blocker**: 无 - 可以并行开发

### Objective
将Flash Attention Kernel 1 (naive)从CUDA移植到AMD HIP，证明跨平台GPU编程能力。

### Why This Matters
JD明确要求："Familiarity with ROCm... is a big plus"  
当前只有CUDA → 面试竞争力下降  
补救：展示CUDA→HIP移植能力

### Deliverables

1. **File**: `kernels/kernel_01_naive.hip`
   - 直接移植自`kernel_01_naive.cu`
   - 使用HIP API替代CUDA API
   - 保持相同算法逻辑

2. **Key Changes (CUDA → HIP)**:
   ```cpp
   // Memory allocation
   cudaMalloc → hipMalloc
   cudaMemcpy → hipMemcpy  
   cudaFree → hipFree
   
   // Kernel launch (相同)
   __global__ → __global__
   __shared__ → __shared__
   __syncthreads() → __syncthreads()
   
   // Warp shuffle (注意差异)
   __shfl_xor_sync(0xFFFFFFFF, val, offset) → __shfl_xor(val, offset)
   ```

3. **Build Script**: `build_hip.sh`
   ```bash
   #!/bin/bash
   hipcc -O3 -o flash_attention_hip kernels/kernel_01_naive.hip
   ```

4. **Verification**:
   - 输出与CUDA版本一致（bitwise或1e-4 tolerance）
   - 8/8 correctness tests passing
   - 记录HIP性能数字（即使无AMD硬件，也要报告编译成功）

### Technical Requirements

```cpp
// Expected HIP kernel signature
__global__ void flash_attn_kernel_v1_hip(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int seq_len,
    int head_dim,
    float softmax_scale
);
```

### CUDA vs HIP差异对照表

| Feature | CUDA | HIP | Status |
|---------|------|-----|--------|
| Kernel keyword | `__global__` | `__global__` | ✅ 相同 |
| Shared memory | `__shared__` | `__shared__` | ✅ 相同 |
| Sync | `__syncthreads()` | `__syncthreads()` | ✅ 相同 |
| Thread ID | `threadIdx.x` | `threadIdx.x` | ✅ 相同 |
| Block ID | `blockIdx.x` | `blockIdx.x` | ✅ 相同 |
| Warp shuffle | `__shfl_xor_sync(mask, val, offset)` | `__shfl_xor(val, offset)` | ⚠️ 注意 |
| Memory alloc | `cudaMalloc` | `hipMalloc` | ⚠️ 前缀 |
| Copy | `cudaMemcpy` | `hipMemcpy` | ⚠️ 前缀 |
| Device sync | `cudaDeviceSynchronize` | `hipDeviceSynchronize` | ⚠️ 前缀 |

**移植难度**: LOW (90%代码相同)

### Acceptance Criteria
- [ ] `kernel_01_naive.hip` compiles with `hipcc`
- [ ] Output matches CUDA version (8/8 tests)
- [ ] `build_hip.sh` build script
- [ ] README_HIP.md with porting notes
- [ ] Performance comparison (CUDA vs HIP, if AMD hardware available)

### WorkBuddy Deliverables Checklist
When complete, commit with message:
```
[Task-3-DONE] CUDA→HIP移植 - AMD GPU适配

- kernel_01_naive.hip: HIP移植版本
- build_hip.sh: 编译脚本
- README_HIP.md: 移植说明和差异分析
- 测试: 8/8 correctness tests passing
- 性能: X TFLOPS (AMD hardware) or 编译验证通过

@Kraber: HIP移植完成，具备跨平台GPU编程能力
JD要求: "Familiarity with ROCm... is a big plus" ✅ 满足
```

---

## 🔴 URGENT: Task 4 - GPU微基准测试套件 [ACTIVE]

**Status**: 🔴 CRITICAL - START NOW  
**Assignee**: WorkBuddy  
**Priority**: P0 (JD核心要求)  
**Deadline**: 48小时 (与Task 3并行)  

### Objective
创建GPU微基准测试套件，直接回应JD: "Graphics & ML Micro Benchmark test case development"

### Why This Matters
JD工作内容第1条: "We will involve you in Graphics & ML Micro Benchmark test case development"  
当前项目: 单个算法实现  
补救: 系统性GPU characterization能力

### Deliverables

#### 1. `amd-microbench/memory_bandwidth.hip`
测试HBM/LDS带宽，各种访问模式
```cpp
// Tests:
// - Sequential read (coalesced)
// - Strided read (uncoalesced)  
// - Random read
// Measure: effective bandwidth vs theoretical peak
```

#### 2. `amd-microbench/compute_throughput.hip`
测试计算吞吐量
```cpp
// Tests:
// - FMA throughput (FLOPS)
// - ADD/MUL throughput
// - Varying occupancy (thread count)
// Measure: GFLOPS vs theoretical peak
```

#### 3. `amd-microbench/occupancy_test.hip`
测试occupancy与latency hiding
```cpp
// Tests:
// - Memory-bound workload at varying occupancy
// - Find optimal thread count for latency hiding
// Measure: execution time vs occupancy
```

#### 4. `amd-microbench/README.md`
文档说明每个测试的目的和解读方法

### Acceptance Criteria
- [ ] 3个微基准测试程序（HIP或CUDA）
- [ ] 每个测试有明确的输出指标
- [ ] README.md解释测试设计和结果解读
- [ ] 与Flash Attention项目关联（展示系统化性能分析能力）

### WorkBuddy Deliverables Checklist
When complete, commit with message:
```
[Task-4-DONE] GPU微基准测试套件

- memory_bandwidth.hip: HBM/LDS带宽测试
- compute_throughput.hip: FMA/ALU吞吐量测试
- occupancy_test.hip: Latency hiding vs occupancy
- README.md: 测试设计和结果解读

JD要求: "Micro Benchmark test case development" ✅ 直接满足
可展示技能: 系统性GPU characterization, performance analysis
```

---

## 📋 任务优先级矩阵

| 任务 | 紧急度 | 对JD匹配度 | Deadline | 依赖 |
|------|--------|------------|----------|------|
| Task 3: HIP移植 | 🔴 P0 | 🔴 Critical (ROCm要求) | 48h | None |
| Task 4: 微基准 | 🔴 P0 | 🔴 Critical (JD工作内容) | 48h | None |
| Task 5: 面试文档更新 | 🟡 P1 | 🟡 Important | 48h | Task 3-4完成 |
| Task 6: rocProf学习 | 🟡 P1 | 🟡 Plus | 72h | 如果有AMD硬件 |

---

## 🔄 Kraber ↔ WorkBuddy Workflow (紧急模式)

### Kraber's Actions (JUST COMPLETED)
1. ✅ JD分析完成 (JD_ANALYSIS_AMD.md)
2. ✅ Task 3-4分配 (此文档)
3. 🔄 准备面试话术更新 (等待Task 3-4完成)

### WorkBuddy's Actions (YOUR URGENT TASKS)
1. **立即**: Pull最新代码
2. **并行开发**:
   - Task 3: HIP移植 (预计4-6小时)
   - Task 4: 微基准 (预计4-6小时)
3. **48小时内**: Commit with `[Task-3-DONE]` and `[Task-4-DONE]`
4. **注意**: 如果无AMD GPU，完成HIP代码即可（语法层面正确）

### 并行工作策略
- WorkBuddy做HIP移植和微基准
- Kraber同步更新面试文档
- 48小时后联合测试和验证

---

## 💬 给WorkBuddy的直接消息

@WorkBuddy: **URGENT TASK ASSIGNMENT - 48 HOUR DEADLINE**

新JD要求AMD ROCm经验。这是面试成败关键。

**你的两个并行任务**:

**Task 3 (HIP移植)**:
- 把`kernel_01_naive.cu`改成HIP版本
- 主要是改函数名前缀: cudaMalloc → hipMalloc
- 参考: https://rocmdocs.amd.com/en/latest/Programming_Guides/HIP_programming_guide.html

**Task 4 (微基准)**:
- 创建3个测试程序测GPU性能
- 模仿cuda-samples里的bandwidthTest
- 简单的内存带宽、计算吞吐量测试

**关键**: 即使没有AMD GPU，完成代码也能展示:
1. 跨平台GPU编程能力
2. 系统性性能分析思维
3. 快速学习新技术的能力

**48小时内完成**。我同时更新面试话术。

有问题立即问。开始吧。

---

**Next Action**: WorkBuddy pull代码 → 开始Task 3 & 4并行开发

---

Last Updated: 2026-04-20 03:00  
Status: 🔴 URGENT - 48小时冲刺模式
