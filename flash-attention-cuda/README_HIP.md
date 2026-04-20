# README_HIP.md — Flash Attention HIP Port

> **Source**: `kernels/kernel_01_naive.cu` (CUDA, Kernel 1 — Naive Baseline)  
> **Target**: `kernels/kernel_01_naive.hip` (AMD HIP / ROCm)  
> **Status**: ✅ Code complete — syntax-verified; runtime test requires AMD hardware  
> **Date**: 2026-04-20

---

## Overview

This document describes the CUDA → HIP porting of Flash Attention Kernel 1
(*Naive Baseline*) and explains the key differences between the two GPU
programming models.

The HIP port demonstrates:
1. Cross-platform GPU programming (NVIDIA CUDA ↔ AMD ROCm)
2. Understanding of wavefront vs. warp width differences
3. Portable online-softmax algorithm unchanged across platforms

---

## CUDA → HIP Diff Summary

| Category | CUDA | HIP | Notes |
|----------|------|-----|-------|
| **Header** | `<cuda_runtime.h>` | `<hip/hip_runtime.h>` | |
| **Kernel keyword** | `__global__` | `__global__` | ✅ Identical |
| **Shared memory** | `__shared__` | `__shared__` | ✅ Identical |
| **Barrier** | `__syncthreads()` | `__syncthreads()` | ✅ Identical |
| **Thread/Block IDs** | `threadIdx.x` etc. | `threadIdx.x` etc. | ✅ Identical |
| **Warp shuffle** | `__shfl_xor_sync(mask, v, off)` | `__shfl_xor(v, off)` | ⚠️ No mask param |
| **Memory alloc** | `cudaMalloc` | `hipMalloc` | `cuda` → `hip` prefix |
| **Memory copy** | `cudaMemcpy` | `hipMemcpy` | `cuda` → `hip` prefix |
| **Memory free** | `cudaFree` | `hipFree` | `cuda` → `hip` prefix |
| **Memset** | `cudaMemset` | `hipMemset` | `cuda` → `hip` prefix |
| **Device sync** | `cudaDeviceSynchronize` | `hipDeviceSynchronize` | |
| **Error type** | `cudaError_t` | `hipError_t` | |
| **Success code** | `cudaSuccess` | `hipSuccess` | |
| **Last error** | `cudaGetLastError` | `hipGetLastError` | |
| **Error string** | `cudaGetErrorString` | `hipGetErrorString` | |
| **Streams** | `cudaStream_t` | `hipStream_t` | |
| **Compiler** | `nvcc` | `hipcc` | |
| **Target arch flag** | `-arch=sm_89` | `--offload-arch=gfx1100` | GPU-specific |

**Overall porting effort**: LOW — ~90 % of the kernel code is unchanged.

---

## Critical Architecture Difference: Wavefront Size

This is the most important *conceptual* difference between NVIDIA and AMD:

| GPU | Execution unit | Width | Effect |
|-----|----------------|-------|--------|
| NVIDIA | **Warp** | **32 threads** | Each query uses 32 threads |
| AMD (GCN/RDNA2+) | **Wavefront** | **64 threads** (default) | Same query uses 64 threads |
| AMD (RDNA3 wave32) | **Wavefront** | **32 threads** | Match NVIDIA behavior |

### Implications for this kernel

The CUDA version used `WF_SIZE = 32` and `ELEMS_PER_THREAD = HEAD_DIM / 32`.
For `HEAD_DIM = 64`, each CUDA thread handled **2 elements**.

The HIP version adapts at **compile time** via the `WF_SIZE` macro:

```cpp
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
  #define WF_SIZE 64   // AMD default wavefront
#else
  #define WF_SIZE 32   // CUDA compat / wave32 mode
#endif
```

For `HEAD_DIM = 64, WF_SIZE = 64`:  each thread handles **1 element**.  
For `HEAD_DIM = 128, WF_SIZE = 64`: each thread handles **2 elements**.

The butterfly warp-reduction also gains an extra round for 64 lanes:

```cpp
__device__ __forceinline__ float wf_reduce_sum(float val) {
#if WF_SIZE == 64
    val += __shfl_xor(val, 32);   // extra step for 64-lane wavefront
#endif
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor(val, offset);
    return val;
}
```

### `__shfl_xor_sync` vs `__shfl_xor`

| Platform | Call | Mask |
|----------|------|------|
| CUDA | `__shfl_xor_sync(0xFFFFFFFF, val, offset)` | Explicit 32-bit mask |
| HIP ROCm < 5.3 | `__shfl_xor(val, offset)` | Implicit (full wavefront) |
| HIP ROCm ≥ 5.3 | `__shfl_xor_sync(mask, val, offset)` | Also supported |

We use `__shfl_xor` in this port for broadest compatibility.

---

## Build Instructions

### Prerequisites

- ROCm 5.x or later: https://rocm.docs.amd.com/en/latest/deploy/linux/index.html
- Supported GPU: any AMD GPU with `gfx900` or newer (Vega, RDNA, CDNA)

### Quick build

```bash
# From project root: flash-attention-cuda/
chmod +x build_hip.sh

# Build only (no GPU required — syntax verification):
./build_hip.sh --no-gpu

# Build + run self-test (AMD GPU required):
./build_hip.sh --selftest

# Specify target architecture explicitly:
./build_hip.sh --arch gfx1100      # RX 7900 XT (RDNA3)
./build_hip.sh --arch gfx90a       # MI250 (CDNA2)
./build_hip.sh --arch gfx906       # MI50 (Vega20)
```

### Manual compile

```bash
# Kernel object only:
hipcc -O3 --offload-arch=gfx1100 -c kernels/kernel_01_naive.hip -o kernel_hip.o

# Self-contained test binary:
hipcc -O3 --offload-arch=gfx1100 -DHIP_SELFTEST \
    kernels/kernel_01_naive.hip -o flash_hip_test

./flash_hip_test
```

---

## Performance Expectations

| Config | CUDA (RTX 4080) | HIP (expected RX 7900 XT) |
|--------|-----------------|---------------------------|
| seq=256,  dim=64 | 0.14 TFLOPS | ~0.10–0.20 TFLOPS |
| seq=1024, dim=64 | 0.51 TFLOPS | ~0.30–0.60 TFLOPS |

> **Note**: Without AMD hardware, performance numbers cannot be measured.  
> The HIP port is provided for cross-platform portability and interview demonstration.  
> Compiler syntax has been verified; runtime correctness pending hardware access.

---

## What This Demonstrates (Interview Angle)

1. **Cross-platform GPU programming** — not vendor-locked
2. **AMD architecture understanding** — wavefront=64, LDS, rocProf
3. **Low-level porting discipline** — identified every API difference
4. **Portable algorithm design** — online softmax unchanged across platforms

### Interview talking point

> "I ported my Flash Attention kernel from CUDA to HIP. The algorithm is
> identical—online softmax doesn't care about the GPU vendor. The main
> differences are API prefix changes and the wavefront width: AMD defaults to
> 64 threads vs NVIDIA's 32-thread warps. I handled that with a compile-time
> macro so the template dispatch stays correct on both platforms."

---

## File Index

| File | Description |
|------|-------------|
| `kernels/kernel_01_naive.cu` | Original CUDA kernel (reference) |
| `kernels/kernel_01_naive.hip` | **This HIP port** |
| `build_hip.sh` | Build script for HIP kernels & micro-benchmarks |
| `amd-microbench/` | GPU micro-benchmark suite (memory, compute, occupancy) |
| `README_HIP.md` | This document |

---

## ROCm Profiler (rocProf) Quick Reference

```bash
# Record hardware counters
rocprof --stats ./flash_hip_test

# Collect specific metrics
rocprof -i metrics.txt ./flash_hip_test

# Typical metrics file (metrics.txt):
# pmc: FETCH_SIZE, WRITE_SIZE, GRBM_COUNT, SQ_WAVES, SQ_INSTS_VALU
```

Key AMD metrics to watch:
- `FETCH_SIZE` / `WRITE_SIZE` — HBM traffic (GB)
- `SQ_INSTS_VALU` — vector ALU instructions (proxy for FLOPS)
- `SQ_WAVES` — wavefronts launched (occupancy indicator)
- `TA_BUSY_cycles` — texture addressing unit busy cycles

---

*Last updated: 2026-04-20 by WorkBuddy Collab (collab-006)*
