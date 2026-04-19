# Flash Attention Project - Task Assignments
> **Manager**: Kraber  
> **Executor**: WorkBuddy  
> **Last Updated**: 2026-04-19 20:50  
> **Status**: 🟢 Active - Task 2 In Progress

---

## ✅ COMPLETED: Task 1 - Naive Flash Attention

**Status**: ✅ DONE - WorkBuddy  
**Completed**: 2026-04-19 20:43  
**Performance**: 0.51 TFLOPS @ seq=1024, dim=64 (RTX 4080)  
**Tests**: 8/8 passed, max diff ~5e-8

### WorkBuddy's Deliverables
- ✅ `kernel_01_naive.cu` - Online softmax + warp-level dot product
- ✅ `include/flash_attention.h` - Kernel declarations
- ✅ `include/utils.cuh` - Warp/block reduction primitives
- ✅ `test_correctness.cu` - 8 test cases
- ✅ `PROGRESS.md` updated with benchmarks

### Key Technical Details
- 1 warp per query row (32 threads)
- Template dispatch for head_dim 32/64/128
- ELEMS_PER_THREAD for scaling
- Grid: (seq_len, num_heads, batch_size)
- Block: (WARP_SIZE, 1, 1)

---

## 🎯 CURRENT: Task 2 - Tiling Optimization [ACTIVE]

**Status**: 🔴 ACTIVE - START NOW  
**Assignee**: WorkBuddy  
**Priority**: P0 (CRITICAL PATH)  
**Estimated Time**: 2-3 hours  
**Goal**: +50% performance over Kernel 1

### Objective
Add tiling to reduce global memory traffic. Currently Kernel 1 reads K and V from global memory N times (once per query). With tiling, we load K/V blocks into shared memory and reuse them across multiple queries.

### Current vs Target Memory Access Pattern

**Kernel 1 (Current)**:
```
For each query i:
  For each key j:
    Read K[j] from HBM (N * N reads!)
    Read V[j] from HBM (N * N reads!)
```

**Kernel 2 (Target)**:
```
For each KV tile:
  Load KV tile to shared memory (1 HBM read per tile)
  For each query in query tile:
    Read from shared memory (fast!)
```

### Deliverables

1. **File**: `kernels/kernel_02_tiling.cu`
   - Tile size: 64x64 or 128x128 (experiment with both)
   - Q, K, V tiles in shared memory
   - Double-buffering for overlap

2. **Performance Report**:
   - Compare vs Kernel 1 at seq=1024, 2048, 4096
   - Memory bandwidth reduction
   - Arithmetic intensity improvement

3. **Test Verification**:
   - Same 8/8 correctness tests pass
   - Numerical stability check (tiling shouldn't change output)

### Technical Requirements

```cuda
// Expected signature
__global__ void flash_attn_kernel_v2(
    const float* Q, const float* K, const float* V,
    float* O,
    int seq_len, int head_dim,
    int Q_tile_size, int KV_tile_size  // New: configurable tile sizes
);
```

### Key Optimizations

1. **Shared Memory Tiling**:
   - Load Q, K, V tiles into `__shared__` memory
   - Compute attention within tile
   - Accumulate partial results

2. **Tile Size Selection**:
   - Target: Fit in 48KB shared memory (RTX 4080)
   - Q_tile: (64, 64) or (128, 64)
   - KV_tile: (64, 64) or (128, 64)
   - Maximize occupancy (blocks per SM)

3. **Double Buffering** (optional but recommended):
   - Load next tile while computing current
   - Hide latency

4. **Grid/Block Configuration**:
   - More threads per block (256 or 512)
   - Fewer blocks (one per tile, not per query)

### Performance Targets

| Metric | Kernel 1 | Kernel 2 Target | Improvement |
|--------|----------|-----------------|-------------|
| seq=1024, d=64 | 0.51 TFLOPS | 0.75+ TFLOPS | +50% |
| seq=2048, d=64 | TBD | TBD | +50% |
| HBM Reads | N² × d | (N/B)² × B × d | ~B× reduction |

### Tutorial Reference
- https://lubits.ch/flash/Part-4 (Tiling & Shared Memory)
- Flash Attention 2 Paper: Section 3.2 (Tiling)

### Acceptance Criteria
- [ ] Compiles with `nvcc -O3 -arch=sm_89`
- [ ] Passes 8/8 correctness tests
- [ ] Performance ≥1.5× Kernel 1
- [ ] Shared memory usage documented
- [ ] Tile size rationale in comments

### WorkBuddy Deliverables Checklist
When complete, commit with message:
```
[Task-2-DONE] Tiling optimization with shared memory

- kernel_02_tiling.cu: 64x64 and 128x64 tile implementations
- Performance: X TFLOPS (+Y% vs Kernel 1)
- Memory bandwidth: Z GB/s (reduced from W GB/s)
- Tests: 8/8 passing

@Kraber: Ready for Task 3 (advanced memory optimization)
```

---

## 📋 UPCOMING: Task 3 - Bank Conflict Resolution [READY]

**Status**: ⏳ Ready  
**Priority**: P0  
**Estimated Time**: 2-3 hours

### Objective
Resolve shared memory bank conflicts through swizzling and memory layout optimization.

### Key Techniques
- Memory layout: [seq, head_dim] → [head_dim, seq] or swizzled
- Bank conflict analysis with Nsight Compute
- Warp shuffle vs shared memory tradeoffs

---

## 📊 Task Dependency Graph

```
Task 1 (Naive) ✅ ────┐
                       ├──→ Task 4 (Testing/Doc) ──→ Interview Prep
Task 2 (Tiling) 🔴 ───┤
                       ├──→ Task 3 (Bank Conflicts)
Task 3 (Shared Opt) ⏳┘
```

---

## 🔄 Kraber ↔ WorkBuddy Workflow

### Kraber's Actions (JUST COMPLETED)
1. ✅ Pulled Task 1 completion (ac46a60)
2. ✅ Reviewed code quality (excellent!)
3. ✅ Assigned Task 2 (this document)
4. 🔄 Committing and pushing now

### WorkBuddy's Actions (YOUR TURN)
1. Pull latest (you'll see Task 2 assignment)
2. Read Task 2 requirements above
3. Implement kernel_02_tiling.cu
4. Benchmark vs Kernel 1
5. Commit with [Task-2-DONE]
6. Push

### Response Time Expectation
- **Ideal**: 2-3 hours for Task 2
- **Acceptable**: 4-6 hours
- **Alert threshold**: 8 hours (ping Kraber)

---

## 💬 Communication Log

### 2026-04-19 20:15 - Kraber
Created Flash Attention project structure. Assigned Task 1.

### 2026-04-19 20:22 - Kraber
Pushed complete project structure. Task 1 officially assigned.

### 2026-04-19 20:35 - Kraber 🚨 ACTIVATION
DIRECT COMMAND to WorkBuddy: START TASK 1 IMMEDIATELY

### 2026-04-19 20:43 - WorkBuddy ✅ TASK 1 COMPLETE
```
[Task-1-DONE] Naive Flash Attention kernel - 8/8 correctness tests passed

Performance on RTX 4080 (sm_89):
  seq=1024, dim=64: 0.263ms, 0.51 TFLOPS
  seq=512, dim=128, h=8: 0.465ms, 1.16 TFLOPS

@Kraber: Task 1 complete. Ready for Task 2 (tiling optimization).
```

**Code Delivered**:
- kernel_01_naive.cu (226 lines)
- include/flash_attention.h (72 lines)
- include/utils.cuh (129 lines)
- test_correctness.cu (258 lines)

### 2026-04-19 20:50 - Kraber 🎯 TASK 2 ASSIGNMENT
Responding to Task 1 completion with immediate Task 2 assignment.

---

## 🎯 Interview Value Progress

| Kernel | Technique | Interview Talking Point |
|--------|-----------|------------------------|
| 1 ✅ | Online softmax | "Implemented numerical stable softmax without materializing N×N matrix" |
| 2 🔴 | Tiling | "Reduced HBM bandwidth by tiling - crucial for memory-bound attention" |
| 3 ⏳ | Bank conflicts | "Solved shared memory bank conflicts through swizzling" |
| 4 ⏳ | CUTLASS | "Applied NVIDIA's CUTLASS patterns for optimal GEMM" |
| 5 ⏳ | Instruction fusion | "Fused operations to maximize tensor core throughput" |
| ... | ... | ... |
| 16 ⏳ | Final tuning | "Achieved 99.2% of cuDNN through iterative profiling" |

---

## 📈 Performance Targets (Updated with Kernel 1 Baseline)

| Kernel | Target TFLOPS | vs Kernel 1 | vs Official |
|--------|---------------|-------------|-------------|
| 1 (Naive) ✅ | 0.51 | 100% | ~25% |
| 2 (Tiling) 🔴 | 0.75+ | +50% | ~35% |
| 3 (Bank Opt) | 1.0+ | +100% | ~50% |
| 4 (CUTLASS) | 1.5+ | +200% | ~75% |
| ... | ... | ... | ... |
| 16 (Final) | 150+ | +300× | ≥99.2% |

---

## 🚀 NEXT ACTION

**WorkBuddy**: You are reading this because you completed Task 1. Excellent work!

**NOW DO THIS**:
```bash
git pull origin master          # Get Task 2 assignment
cat TASKS.md | head -100        # Read Task 2 requirements
vim kernels/kernel_02_tiling.cu # Start implementing
# ... 2-3 hours of CUDA coding ...
git commit -m "[Task-2-DONE] ..."
git push origin master
```

**Kraber**: Monitoring for [Task-2-DONE] signal. Task 3 ready to assign immediately.

---

**Last Updated**: 2026-04-19 20:50 by Kraber
**Status**: Task 1 ✅ COMPLETE | Task 2 🔴 ACTIVE | Task 3 ⏳ READY
