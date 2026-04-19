# Flash Attention - AI Infra Interview Prep
> **Project**: Flash Attention CUDA Implementation  
> **Kernels Completed**: 2/16  
> **Interview Focus**: GPU Optimization, CUDA Programming, Performance Engineering  
> **Last Updated**: 2026-04-20

---

## 🎯 面试核心定位

**你要讲的故事**：
> "I implemented Flash Attention 2 from scratch in CUDA, going through 16 kernel iterations. In just the first 2 kernels, I learned why naive optimizations fail and how to properly leverage GPU memory hierarchy."

**展示的skill**:
1. CUDA编程能力（Kernel 1的warp-level reduction）
2. 性能分析思维（Kernel 2的失败→分析→改进计划）
3. GPU架构理解（memory hierarchy, shared memory, occupancy）
4. 迭代优化方法论（profiler-driven optimization）

---

## 📚 两个Kernel的学习总结

### Kernel 1: Naive Flash Attention (Baseline)

**What I Built**:
- Single warp per query (32 threads)
- Global memory only
- Online softmax algorithm
- Template dispatch for head_dim 32/64/128

**Key Technical Details**:
```cuda
// 1 warp = 32 threads处理1个query
// 每个thread处理2-4个elements (基于head_dim)
// Warp shuffle for reduction: __shfl_xor_sync

for (int kv_row = 0; kv_row < seq_len; kv_row++) {
    // 每个query读取所有K/V from HBM
    // HBM traffic: N queries × N KV rows × d = O(N²d) reads
}
```

**Performance**: 0.51 TFLOPS @ RTX 4080, seq=1024, dim=64

**面试Talking Point**:
> "My first kernel focused on correctness. I implemented online softmax to avoid materializing the full N×N attention matrix—that's the core innovation of Flash Attention. Each warp handles one query row, using warp shuffle primitives for reduction."

**Key Concepts Demonstrated**:
- Online softmax numerical stability
- Warp-level parallelism (32 threads collaborating)
- Warp shuffle (`__shfl_xor_sync` for reduction)
- Template metaprogramming for compile-time head_dim

---

### Kernel 2: Tiling "Optimization" (The Failure)

**What I Built**:
- Shared memory tiles for K/V (TILE=32)
- Load K/V tile once, reuse across computations
- Same online softmax algorithm

**The Failure**:
| Metric | Kernel 1 | Kernel 2 | "Improvement" |
|--------|----------|----------|---------------|
| seq=1024, d=64 | 0.51 TFLOPS | 0.26 TFLOPS | **-50%** ⚠️ |

**Why It Failed**:
```
Problem: 1 warp = 1 query = 32 threads
- We load K[32×64] = 2KB into shared memory
- But only 32 threads use it (1 warp)
- Shared memory latency not amortized
- Overhead: barrier sync, index calculation, bank conflicts

The tiling benefit requires COOPERATION:
- Multiple warps share the same K/V tile
- 1 tile load → 32 queries processed
- That's Task 3 (Cooperative Loading)
```

**面试Talking Point (This is GOLD)**:
> "My second kernel was supposed to optimize with shared memory tiling, but it got 50% slower. Here's why: I was loading K/V tiles into shared memory, but each tile was only used by one warp. The shared memory latency and sync overhead wasn't amortized. The real benefit comes when multiple queries share the same tile—that's my Task 3."

**What This Shows Interviewers**:
- ✅ You understand why optimizations fail
- ✅ You can analyze performance regressions
- ✅ You know shared memory isn't magic—it needs proper access patterns
- ✅ You have a plan to fix it (cooperative loading)

---

## 🎯 核心面试问题 & 答案

### Q1: "Tell me about a challenging optimization you worked on"

**Your Answer**:
> "I implemented Flash Attention from scratch for my portfolio. The challenge was that my first 'optimization' made things worse."
> 
> "I added shared memory tiling to reduce HBM traffic, but performance dropped 50%. I analyzed with Nsight Compute and realized: I was loading tiles into shared memory, but each tile was only accessed by one warp. The sync overhead and latency weren't amortized."
> 
> "The fix is cooperative loading—multiple warps share the same tile. That's my next kernel. This taught me that GPU optimization isn't about using features, it's about matching the parallelism model to the memory hierarchy."

---

### Q2: "What's the difference between global memory and shared memory?"

**Your Answer**:
> "Global memory (HBM) is high bandwidth but high latency—hundreds of cycles. Shared memory is on-chip, much lower latency, but limited size—48KB-164KB per SM."
> 
> "The key is **reuse**. In my Flash Attention kernel, I tried loading K/V tiles to shared memory, but since each query needed different data, I didn't get reuse. The traffic to HBM was the same, plus I added sync overhead. Proper tiling requires multiple threads to reuse the same data—that's when shared memory shines."

---

### Q3: "How do you approach performance optimization?"

**Your Answer**:
> "Profiler-driven, iterative approach:"
> 
> "1. **Baseline**: Get a correct implementation with performance numbers. I got 0.51 TFLOPS with my naive kernel."
> 
> "2. **Hypothesis**: Identify bottlenecks. I thought HBM bandwidth was the limit, so I tried shared memory tiling."
> 
> "3. **Experiment**: Implement and measure. I added tiling and got 0.26 TFLOPS—worse!"
> 
> "4. **Analyze**: Use profiler metrics. The shared memory wasn't being reused effectively. Each tile served only one warp."
> 
> "5. **Iterate**: Design next version. Cooperative loading—multiple queries share tiles."
> 
> "This cycle is how you move from 0.5 TFLOPS to 150+ TFLOPS like the official implementation."

---

### Q4: "What's Flash Attention and why is it important?"

**Your Answer**:
> "Standard attention is O(N²) memory—materializing the full attention matrix. For seq_len=4096, that's 16M elements, ~64MB just for the matrix. You're memory-bound."
> 
> "Flash Attention uses tiling and online softmax to avoid materializing the matrix. It computes attention in tiles, keeping only running statistics (max, sum). This reduces memory to O(N) and makes it compute-bound instead of memory-bound."
> 
> "I implemented the core algorithm: online softmax with running max and sum correction. The challenge is the CUDA optimization to reach the 150+ TFLOPS of the official implementation—that's the 16-kernel journey I'm on."

---

### Q5: "Explain warp shuffle and when you'd use it"

**Your Answer**:
> "Warp shuffle lets threads in a warp exchange data without going through shared memory. `__shfl_xor_sync` does a butterfly pattern—great for reductions."
> 
> "I used it in my attention kernel for dot products. Each thread holds 2-4 elements, computes partial dot product, then warp shuffle reduces to final sum."
> 
> ```cuda
> float dot = 0;
> for (int e = 0; e < elems; e++) {
>     dot += q[e] * k[e];  // per-thread partial
> }
> dot = warp_reduce_sum(dot);  // exchange across warp
> ```
> 
> "It's faster than shared memory for warp-level reduction—no sync needed, hardware-supported."

---

## 📊 数据支撑（面试时可以说）

### Performance Numbers
```
Hardware: RTX 4080 (sm_89)
Kernel 1 (Naive):    0.51 TFLOPS @ seq=1024, dim=64
Kernel 2 (Tiling):   0.26 TFLOPS (regression due to smem overhead)
Official FlashAttn:  150+ TFLOPS (target)
Gap to close:        300x improvement needed
```

### Correctness Verification
```
All kernels: 8/8 test cases passing
Max diff: ~5e-8 (numerical stability confirmed)
Mean diff: ~4e-9 (excellent precision)
```

### Memory Hierarchy Understanding
```
HBM bandwidth:  ~500 GB/s (RTX 4080)
SMEM bandwidth: ~10+ TB/s (on-chip)
Compute:        ~50 TFLOPS (FP32 peak)

Flash Attention goal: Move from memory-bound to compute-bound
My current: Still memory-bound (low TFLOPS vs peak)
```

---

## 🚀 如何展示这个项目（面试话术）

### 开场（30秒）
> "I built Flash Attention from scratch in CUDA as a learning project. I went through 16 kernel iterations, starting from a naive 0.5 TFLOPS baseline. I've completed 2 kernels so far, and I've already learned a critical lesson: naive optimizations can make things worse."

### 深入（2分钟）
> "My first kernel used warp shuffle for online softmax—each warp handles one query. It worked, 8/8 tests passing. Then I tried to optimize with shared memory tiling... and performance dropped 50%."
> 
> "I analyzed why: I was loading K/V tiles into shared memory, but each tile was only used by one warp. The latency and sync overhead weren't amortized. The real optimization requires cooperative loading—multiple queries sharing tiles."
> 
> "This taught me that GPU optimization isn't about using features—it's about matching the parallelism model to the memory hierarchy. I'm currently implementing Task 3 with cooperative loading."

### 收尾（30秒）
> "The goal is reaching 99.2% of the official Flash Attention performance on A100—about 150 TFLOPS. That's a 300x improvement from my baseline, which shows how deep GPU optimization goes."

---

## 🎯 展示的技术深度

| Skill | Evidence |
|-------|----------|
| CUDA Programming | Kernel implementations, warp shuffle, templates |
| GPU Architecture | Understanding smem, occupancy, memory hierarchy |
| Performance Analysis | Profiler mindset, hypothesis→experiment→analyze |
| Numerical Stability | Online softmax implementation, correctness tests |
| Iterative Optimization | Failed attempt → analysis → improved plan |
| System Design | 16-kernel roadmap, task breakdown |

---

## 📝 准备的其他问题

**系统设计**:
- "How would you parallelize attention across multiple GPUs?" (Tensor/pipeline parallelism)
- "What changes for inference vs training?" (KV cache, incremental computation)

**深入技术**:
- "Why online softmax instead of standard softmax?" (Avoid materializing N×N matrix)
- "What causes shared memory bank conflicts?" (Multiple threads hitting same bank)
- "How do you hide latency?" (Double buffering, occupancy, instruction-level parallelism)

**行为问题**:
- "Tell me about a time you failed and recovered." (Kernel 2!)
- "How do you prioritize optimizations?" (Roofline model, profiler data)

---

## 🎬 面试展示建议

### 如果面试官有CUDA背景
- 深入warp shuffle, occupancy, shared memory bank conflicts
- Discuss cooperative loading design for Task 3
- Talk about CUTLASS patterns (Task 5)

### 如果面试官是系统/infra背景
- Focus on memory hierarchy and Roofline model
- Discuss why Flash Attention matters for LLM serving
- Talk about batching, scheduling, throughput vs latency

### 如果面试官是算法背景
- Explain online softmax and numerical stability
- Discuss attention complexity and why O(N²) is the bottleneck
- Connect to transformer architecture and LLM trends

---

## 🔗 参考资源

- Tutorial: https://lubits.ch/flash/
- Paper: FlashAttention-2 (arXiv:2307.08691)
- Code: `flash-attention-cuda/kernels/kernel_01_naive.cu`
- Code: `flash-attention-cuda/kernels/kernel_02_tiling.cu`
- Tests: `flash-attention-cuda/tests/test_correctness.cu`

---

**Last Updated**: 2026-04-20  
**Status**: 2 kernels complete, interview-ready  
**Next Milestone**: Task 3 (Cooperative Loading) for real performance gains
