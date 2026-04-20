# AMD GPU Micro-Benchmark Suite

> **Directory**: `flash-attention-cuda/amd-microbench/`  
> **Purpose**: GPU performance characterization for AMD ROCm / HIP  
> **JD Alignment**: "Graphics & ML Micro Benchmark test case development"  
> **Date**: 2026-04-20

---

## Overview

This suite provides three standalone micro-benchmarks that characterize GPU
performance at the hardware level. Together they answer the question:
*"Is my workload memory-bound or compute-bound, and what occupancy is optimal?"*

This directly addresses the AMD Radeon GPU performance analysis JD requirement:

> *"We will involve you in Graphics & ML Micro Benchmark test case development"*

---

## Benchmark Descriptions

### 1. `memory_bandwidth.hip` — HBM & LDS Bandwidth

Measures effective memory bandwidth across access patterns:

| Pattern | What It Tests |
|---------|---------------|
| **SEQ_READ** | Fully coalesced sequential read — measures peak HBM bandwidth |
| **STRIDE_2/4/16** | Strided access — shows bandwidth degradation vs. stride |
| **RAND_READ** | Random access — worst case, cache-thrashing penalty |
| **SEQ_WRITE** | Write bandwidth (often < read due to write-combine buffers) |
| **LDS_BW** | On-chip LDS (Local Data Share) throughput |

**Key output metric**: `GB/s effective bandwidth`  
**Expected insight**: SEQ_READ ≈ peak HBM; RAND_READ << SEQ_READ (cache misses)

---

### 2. `compute_throughput.hip` — FLOPs Throughput

Measures raw compute throughput under different instruction mixes:

| Test | What It Tests |
|------|---------------|
| **FP32_FMA (thrput)** | Peak FP32 FLOPs via independent FMA chains (ILP-heavy) |
| **FP32_ADD (latency)** | Serial ADD chain — measures instruction latency, not throughput |
| **FP32_MUL (thrput)** | Multiply throughput with 4 independent chains |
| **FP64_FMA (thrput)** | FP64 throughput (expect ~1/16 of FP32 on gaming GPUs) |
| **Occupancy sweep** | FP32 FMA at varying block sizes — find optimal thread count |

**Key output metric**: `GFLOPS`  
**Expected insight**: FP32 throughput >> serialized latency; FP64 much lower on RDNA

---

### 3. `occupancy_test.hip` — Occupancy vs Latency Hiding

Explores the relationship between GPU occupancy and performance:

| Test | What It Tests |
|------|---------------|
| **Memory-bound sweep** | Bandwidth vs grid size — shows latency hiding "knee" |
| **Compute-bound sweep** | GFLOPS vs grid size — compute-limited, plateaus early |
| **SMEM footprint** | How shared memory size reduces occupancy |

**Key output metric**: `occupancy_pct`, `GB/s`, `time_ms`  
**Expected insight**: Memory-bound workloads need high occupancy to hide ~300ns HBM latency

---

## Build & Run

### Prerequisites

- ROCm 5.x+ with `hipcc`: https://rocm.docs.amd.com/
- AMD GPU (any GCN/RDNA/CDNA)

### Build

```bash
# From flash-attention-cuda/ root:
chmod +x build_hip.sh
./build_hip.sh            # builds kernels + microbenchmarks

# Or manually:
cd amd-microbench
hipcc -O3 --offload-arch=gfx1100 memory_bandwidth.hip    -o ../build/microbench/memory_bandwidth
hipcc -O3 --offload-arch=gfx1100 compute_throughput.hip  -o ../build/microbench/compute_throughput
hipcc -O3 --offload-arch=gfx1100 occupancy_test.hip      -o ../build/microbench/occupancy_test
```

Replace `gfx1100` with your GPU:
- `gfx1100` — RX 7900 XT/XTX (RDNA3)
- `gfx1030` — RX 6900 XT (RDNA2)
- `gfx90a`  — MI250/MI250X (CDNA2)
- `gfx906`  — MI50/MI60 (Vega20)

### Run

```bash
./build/microbench/memory_bandwidth        # default 256 MB buffer
./build/microbench/memory_bandwidth 33554432  # custom N elements

./build/microbench/compute_throughput

./build/microbench/occupancy_test
```

---

## Sample Expected Output (RX 7900 XT, gfx1100)

### memory_bandwidth

```
Pattern               BW (GB/s)  Time (ms)  Bytes (MB)
--------------------  ----------  ----------  ----------
SEQ_READ                  890.00       0.120       256.0
STRIDE_2_READ             450.00       0.060       128.0
STRIDE_4_READ             230.00       0.030        64.0
STRIDE_16_READ             60.00       0.020        16.0
RAND_READ                  18.00       0.140         2.0
SEQ_WRITE                 820.00       0.130       256.0
LDS_BW                  4200.00       0.025       256.0
```

*Numbers are illustrative. Actual results depend on hardware.*

### compute_throughput

```
Test                   GFLOPS  Time(ms)
----------------------  ------  --------
FP32_FMA (thrput)     19200.0     0.003
FP32_ADD (latency)      450.0     0.130
FP32_MUL (thrput)      9600.0     0.003
FP64_FMA (thrput)      1200.0     0.048
```

---

## Connection to Flash Attention

These benchmarks directly explain the Flash Attention kernel performance:

```
Kernel 1 (Naive):
  - Every iteration: loads Q_i, K_j, V_j from HBM
  - Bottleneck: HBM bandwidth (~500 GB/s on RTX 4080)
  - Result: 0.51 TFLOPS << 50 TFLOPS peak FP32
  - Conclusion: memory-bandwidth limited

Kernel 2 (Tiling):
  - Loads K/V tiles into LDS (1 block per query)
  - LDS not shared across queries → no bandwidth savings
  - Result: 0.26 TFLOPS (regression from sync overhead)

Kernel 3 (Cooperative Loading, planned):
  - Multiple queries share one LDS tile
  - LDS bandwidth >> HBM bandwidth → real speedup
  - Expected: 2–4× improvement (matches microbench LDS/HBM ratio)
```

---

## Design Notes

### Why independent FMA chains?

```cpp
// Throughput test: 8 independent chains → compiler issues them in parallel
v0 = v0 * a + b;
v1 = v1 * a + b;
// ...

// vs Latency test: serial chain → measures instruction latency
val = val * val;   // each iteration depends on previous
```

Independent chains expose *instruction-level parallelism* (ILP), which is how
GPUs achieve peak TFLOPS. Serial chains reveal the *latency per instruction*.

### Why the occupancy sweep matters

The bandwidth "knee" point in Test 1 of `occupancy_test` reveals:

- **Before the knee**: Adding wavefronts improves BW (latency hidden)
- **At the knee**: All HBM latency fully hidden — adding more wavefronts doesn't help
- **For Flash Attention**: We need enough occupancy to hide the HBM load latency
  between online-softmax iterations

---

## rocProf Integration

To collect hardware counters alongside microbenchmarks:

```bash
# Record all counters for one run
rocprof --stats ./build/microbench/memory_bandwidth

# Specific metrics (create metrics.txt):
# pmc: FETCH_SIZE, WRITE_SIZE, SQ_WAVES, SQ_INSTS_VALU, L2_READ_HIT_RATIO
rocprof -i metrics.txt ./build/microbench/memory_bandwidth

# View results
cat results.csv
```

Key metrics to interpret:
- `FETCH_SIZE` — bytes read from HBM (compare to expected for access pattern)
- `L2_READ_HIT_RATIO` — low for RAND_READ, high for SEQ_READ
- `SQ_WAVES` — active wavefronts (occupancy proxy)
- `SQ_INSTS_VALU` — vector ALU instructions (FP throughput proxy)

---

## Interview Talking Points

> "I built a GPU micro-benchmark suite with three components: memory bandwidth
> (measuring HBM vs LDS across access patterns), compute throughput (FP32/FP64
> FMAs with ILP analysis), and an occupancy study showing how wavefront count
> affects latency hiding.
>
> The suite directly connects to my Flash Attention work: the benchmarks explain
> *why* Kernel 1 is memory-bound at 0.51 TFLOPS and *why* Kernel 2's naive
> tiling didn't help—and what Kernel 3's cooperative loading needs to fix it."

---

*Last updated: 2026-04-20 by WorkBuddy Collab (collab-006)*
