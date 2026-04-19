# Flash Attention Project - Task Assignments
> **Manager**: Kraber  
> **Executor**: WorkBuddy  
> **Last Updated**: 2026-04-19 20:15  
> **Status**: 🟢 Active - Awaiting WorkBuddy Task 1 Completion

---

## 🎯 Current Sprint: Phase 1 - Baseline Implementation

### Task 1: Kernel 1 - Naive Flash Attention [ACTIVE]

**Status**: 🟡 Assigned to WorkBuddy  
**Priority**: P0 (Critical Path)  
**Estimated Time**: 2-3 hours  
**Due**: Next check-in (flexible)

#### Objective
Implement the first naive but correct version of Flash Attention. This establishes the correctness baseline that all optimizations will be measured against.

#### Deliverables
1. **File**: `kernels/kernel_01_naive.cu`
   - Single CUDA kernel implementing Flash Attention forward pass
   - No optimizations - focus on correctness
   - Use global memory only

2. **File**: `tests/test_correctness.cu`
   - Unit test comparing against PyTorch reference
   - Test shapes: (batch=1, heads=1, seq=64, dim=64) → (1, 1, 4096, 64)
   - Tolerance: 1e-4 relative error

3. **File**: `baselines/standard_attention.py`
   - PyTorch reference implementation
   - Generate test cases and ground truth

#### Technical Requirements
```cuda
// Expected kernel signature
__global__ void flash_attn_kernel_v1(
    const float* Q,  // (seq_len, head_dim)
    const float* K,  // (seq_len, head_dim)
    const float* V,  // (seq_len, head_dim)
    float* O,        // (seq_len, head_dim) output
    int seq_len,
    int head_dim
);
```

#### Algorithm (from tutorial Part 1-2)
1. Load Q, K, V tiles into registers
2. Compute S = Q @ K^T (attention scores)
3. Compute P = softmax(S) online (avoid materializing full matrix)
4. Compute O = P @ V
5. Write O to global memory

#### Acceptance Criteria
- [ ] Compiles with `nvcc -O3 -arch=sm_80`
- [ ] Passes correctness tests (1e-4 tolerance)
- [ ] Runs on sequence length 1024 without OOM
- [ ] Baseline performance recorded in PROGRESS.md

#### Tutorial Reference
- https://lubits.ch/flash/Part-1 (Intro)
- https://lubits.ch/flash/Part-2 (Building Blocks)
- https://lubits.ch/flash/Part-3 (Kernel 1)

#### WorkBuddy Deliverables Checklist
When complete, commit with message:
```
[Task-1-DONE] Naive Flash Attention implementation

- kernel_01_naive.cu: Basic working implementation
- test_correctness.cu: Unit tests passing
- Performance: X TFLOPS @ seq_len=1024, dim=64
- Next: Ready for Task 2 (tiling optimization)
```

---

## 📋 Upcoming Tasks (Ready for Assignment)

### Task 2: Kernel 2 - Tiling Optimization [READY]
**Status**: ⏳ Pending Task 1 Completion  
**Priority**: P0  
**Estimated Time**: 2-3 hours

#### Objective
Add tiling to reduce global memory traffic. Split Q, K, V into blocks that fit in shared memory.

#### Key Optimizations
- Tile Q, K, V into shared memory
- Compute attention in tiles
- Reuse K, V tiles across multiple Q tiles

#### Deliverables
- `kernels/kernel_02_tiling.cu`
- Performance comparison vs Kernel 1
- Memory bandwidth analysis

---

### Task 3: Kernel 3 - Shared Memory [READY]
**Status**: ⏳ Pending Task 2 Completion  
**Priority**: P0  
**Estimated Time**: 2-3 hours

#### Objective
Optimize shared memory usage and add double buffering.

#### Key Optimizations
- Shared memory layout optimization
- Double buffering for K, V tiles
- Warp-level parallelism

---

### Task 4: Documentation & Testing [READY]
**Status**: ⏳ Can be done in parallel  
**Priority**: P1  
**Estimated Time**: 1-2 hours

#### Objective
Set up comprehensive testing and profiling infrastructure.

#### Deliverables
- `tests/test_performance.cu`: Benchmark suite
- `scripts/profile.sh`: Nsight Compute profiling
- `PROGRESS.md`: Performance tracking table

---

## 🏃 Quick Commands for WorkBuddy

```bash
# Clone and setup (if not done)
cd /root/.openclaw/workspace
git pull origin master

# Navigate to project
cd flash-attention-cuda

# Create kernel file
mkdir -p kernels tests baselines

# Build (adjust sm_80 for your GPU)
nvcc -O3 -arch=sm_80 -o kernel_01 kernels/kernel_01_naive.cu

# Test
./kernel_01

# Profile with Nsight Compute
ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum ./kernel_01
```

---

## 📊 Task Dependency Graph

```
Task 1 (Naive) ────────┐
                       ├──→ Task 4 (Testing/Doc) ──→ Interview Prep
Task 2 (Tiling) ───────┤
                       ├──→ Task 5 (Bank Conflicts)
Task 3 (Shared Mem) ───┘
```

---

## 🔄 Kraber ↔ WorkBuddy Workflow

### Kraber's Actions
1. **Assign Task**: Write detailed task in TASKS.md → Commit → Push
2. **Monitor**: Check GitHub every 5-10 minutes or wait for `[TASK-X-DONE]` commit
3. **Review**: Pull WorkBuddy's code, verify tests pass
4. **Next Task**: Update TASKS.md, mark previous as complete, assign next

### WorkBuddy's Actions
1. **Accept Task**: Pull latest, read TASKS.md
2. **Implement**: Write code, test locally
3. **Complete**: Commit with `[TASK-X-DONE]` prefix, push
4. **Report**: Update PROGRESS.md with benchmarks

### Deadlock Prevention
- Kraber always has 2-3 tasks ready in "Upcoming Tasks"
- WorkBuddy can work on Task 4 (testing) in parallel if waiting
- Explicit commit messages (`[TASK-X-DONE]`) signal completion

---

## 💬 Communication Log

### 2026-04-19 20:15 - Kraber
Created Flash Attention project structure. Assigned Task 1 (Naive Kernel) to WorkBuddy.

**Message to WorkBuddy**:
> @WorkBuddy: New project started! This is for AI infra interview prep. 
> 
> **Your Task 1**: Implement naive Flash Attention kernel following https://lubits.ch/flash/
> - Focus on correctness, not speed
> - See TASKS.md for full details
> - When done, commit with `[Task-1-DONE]` in message
> - I'll immediately assign Task 2 upon completion

### 2026-04-19 20:22 - Kraber (Git Push)
Pushed complete project structure. Task 1 officially assigned and ready.

---

## 📈 Performance Targets (For PROGRESS.md)

| Kernel | Target TFLOPS | Memory Bandwidth | vs Official |
|--------|---------------|------------------|-------------|
| 1 (Naive) | Baseline | Measure only | - |
| 2 (Tiling) | +50% | Reduced | - |
| 3 (Shared) | +100% | Efficient | - |
| ... | ... | ... | ... |
| 16 (Final) | 150+ TFLOPS | Maximized | ≥99.2% |

---

## 🎯 Success Criteria

### For Task 1 Completion
- [ ] Code compiles without warnings
- [ ] Correctness tests pass (1e-4 tolerance)
- [ ] Can run seq_len=1024, dim=64 without crash
- [ ] Performance number recorded

### For Project Completion (All Tasks)
- [ ] 16 kernels implemented
- [ ] Each kernel has performance improvement over previous
- [ ] Final kernel ≥99.2% of official Flash Attention on A100
- [ ] Comprehensive documentation for interview prep
- [ ] 3-5 key talking points extracted per optimization

---

**Next Action**: WorkBuddy to complete Task 1 and commit with `[Task-1-DONE]`
