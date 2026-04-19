# Flash Attention CUDA Optimization Project
> **Target**: AI Infra Job Interview Portfolio Piece  
> **Reference**: https://lubits.ch/flash/ - "Flash Attention From Scratch" Tutorial  
> **Goal**: Implement and optimize Flash Attention 2 from baseline to 99%+ performance  
> **Platform**: CUDA on Ampere GPUs (A100, RTX 3090/4080)  
> **Manager**: Kraber (task assignment & tracking)  
> **Executor**: WorkBuddy (CUDA implementation & testing)

## Project Overview

This project implements **Flash Attention 2** from scratch following the 10-part tutorial series. We'll go through **16 kernel iterations** of optimization, starting from a naive baseline and ending with performance competitive with the official implementation.

### Target Performance
| Metric | Target | Hardware |
|--------|--------|----------|
| Performance vs Official | ≥99.2% | A100 |
| Performance vs Official | ≥102.9% | RTX 3090 |
| Sequence Length | 4096 | - |

## Architecture

### Flash Attention 2 Core Algorithm
```
Standard Attention:  O(N²) memory, O(N²) compute
Flash Attention:     O(N) memory, O(N²) compute (but memory-efficient)

Key Insight: Tiling + Online Softmax + Recomputation
- Split Q, K, V into blocks
- Compute attention incrementally
- Avoid materializing full N×N attention matrix
```

### Optimization Roadmap (16 Kernels)
Based on tutorial parts:

**Phase 1: Baseline (Kernels 1-3)**
- [ ] Kernel 1: Naive Flash Attention ( correctness focus )
- [ ] Kernel 2: Basic tiling
- [ ] Kernel 3: Shared memory introduction

**Phase 2: Memory Optimization (Kernels 4-6)**
- [ ] Kernel 4: Bank conflict resolution (swizzling)
- [ ] Kernel 5: CUTLASS GEMM patterns
- [ ] Kernel 6: FP instruction fusion

**Phase 3: Compute Optimization (Kernels 7-10)**
- [ ] Kernel 7: A100-specific profiling
- [ ] Kernel 8: Instruction reduction
- [ ] Kernel 9: A100 final optimizations
- [ ] Kernel 10: Comprehensive kernel analysis

**Phase 4: Advanced (Kernels 11-16)**
- [ ] Kernels 11-16: Auto-tuning, warp specialization, etc.

## Project Structure

```
flash-attention-cuda/
├── README.md                    # This file
├── TASKS.md                     # Current task assignments (Kraber ↔ WorkBuddy)
├── PROGRESS.md                  # Progress tracking & benchmarks
├── kernels/                     # CUDA kernel implementations
│   ├── kernel_01_naive.cu       # WorkBuddy: Task 1
│   ├── kernel_02_tiling.cu      # WorkBuddy: Task 2
│   ├── kernel_03_shared_mem.cu  # WorkBuddy: Task 3
│   └── ...                      # Up to kernel_16_final.cu
├── include/                     # Header files
│   ├── flash_attention.h
│   └── utils.cuh
├── tests/                       # Unit tests & benchmarks
│   ├── test_correctness.cu
│   ├── test_performance.cu
│   └── benchmarks.cu
├── baselines/                   # Reference implementations
│   ├── standard_attention.py    # PyTorch baseline for comparison
│   └── official_flash_attn.py   # Official Flash Attention wrapper
├── docs/                        # Learning notes
│   ├── part_1_overview.md
│   ├── part_2_building_blocks.md
│   └── ...                      # Tutorial notes
└── scripts/                     # Helper scripts
    ├── build.sh
    ├── run_tests.sh
    └── profile.sh
```

## Task Management System

### Kraber's Role (Project Manager)
1. **Task Assignment**: Create detailed tasks in TASKS.md
2. **Progress Tracking**: Update PROGRESS.md with benchmarks
3. **Code Review**: Review WorkBuddy's implementations
4. **Documentation**: Maintain project docs and learning notes
5. **Interview Prep**: Extract key insights for AI infra interviews

### WorkBuddy's Role (CUDA Engineer)
1. **Implementation**: Write CUDA kernels per assigned tasks
2. **Testing**: Verify correctness and measure performance
3. **Reporting**: Update task status and provide benchmarks
4. **Optimization**: Apply tutorial techniques iteratively

### Communication Protocol
- **Task Assignment**: Kraber writes tasks in `TASKS.md`, commits, pushes
- **Task Acceptance**: WorkBuddy pulls, reads TASKS.md, implements
- **Task Completion**: WorkBuddy updates PROGRESS.md, commits with `[TASK-X-DONE]`
- **New Tasks**: Kraber sees completion, assigns next batch

### Deadlock Prevention
- **No Waiting**: Kraber always has next 2-3 tasks ready in TASKS.md
- **Parallel Work**: WorkBuddy can start next task while Kraber reviews previous
- **Fallback Tasks**: If blocked, WorkBuddy works on tests/benchmarks/docs

## Interview Value Proposition

### What This Demonstrates
1. **CUDA Programming**: Kernel optimization, memory hierarchy, warp scheduling
2. **GPU Architecture**: Ampere-specific features (Tensor Cores, async copy)
3. **Algorithm Understanding**: Attention mechanisms, numerical stability
4. **Performance Engineering**: Profiling, bottleneck analysis, iterative optimization
5. **System Design**: Memory bandwidth vs compute tradeoffs

### Key Talking Points
- Why Flash Attention matters (memory-bound → compute-bound)
- Tiling strategies for matrix operations
- Online softmax algorithm
- Shared memory bank conflicts & swizzling
- Instruction-level optimizations
- Roofline model analysis

### Resume Entry
```
Flash Attention 2 CUDA Implementation
- Implemented Flash Attention 2 from scratch in CUDA, following 16-kernel optimization tutorial
- Achieved 99.2% performance of official implementation on A100 (sequence length 4096)
- Optimizations: tiling, shared memory swizzling, FP instruction fusion, CUTLASS patterns
- Technologies: CUDA C++, Ampere Tensor Cores, Nsight Compute profiling
```

## References

1. **Tutorial Series**: https://lubits.ch/flash/
2. **GitHub**: https://github.com/sonnyli/flash_attention_from_scratch
3. **Flash Attention 2 Paper**: https://arxiv.org/abs/2307.08691
4. **Official Implementation**: https://github.com/Dao-AILab/flash-attention
5. **CUTLASS**: https://github.com/NVIDIA/cutlass

## Current Status

See `TASKS.md` for current assignments and `PROGRESS.md` for detailed progress.

---

**Last Updated**: 2026-04-19  
**Next Milestone**: Kernel 1 (Naive Implementation) - See TASKS.md
