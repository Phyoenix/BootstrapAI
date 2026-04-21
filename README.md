# BootstrapAI

> An autonomous AI evolution system — self-improving through iterative learning, GitHub commits, and skill acquisition.

**Built by**: Kraber (AI Agent) + Phyoenix (Human)  
**Purpose**: Self-bootstrapping AI evolution platform for GPU kernel optimization, deep learning, and system engineering.

---

## 🚀 CUDA GPU Architecture Visualization

Interactive animation showing NVIDIA GPU data flow from CPU → HBM → L2 → SMEM → Register → Compute → Writeback.

**[▶ Launch Visualization](https://phyoenix.github.io/BootstrapAI/cuda_gpu_arch_visualization.html)**

### Features
- **8-step animation**: CPU preparation → H2D transfer → Kernel launch → L2 cache → SMEM cooperative loading → Register load → Warp computation → Result writeback
- **Real-time metrics**: SM occupancy, HBM bandwidth, L2 hit rate, compute TFLOPS
- **Interactive controls**: Play / pause / step-through
- **Architecture modeled**: RTX 4080 (sm_89), 76 SMs, 716 GB/s HBM
- **Design**: Blade Runner 2049 cyberpunk dark theme

---

## 📁 Project Structure

```
.
├── flash-attention-cuda/    # Flash Attention kernel optimization (16-kernel roadmap)
│   ├── kernels/             # CUDA & HIP kernels (v1 naive → v3 cooperative)
│   ├── include/             # Headers
│   └── tests/               # Correctness test suite
├── research/
│   └── neural-rendering/    # 3D Gaussian Splatting experiments
├── skills/                  # Agent skill definitions
│   ├── kimiim/              # Kimi Group Chat collaboration
│   ├── time-awareness/      # Temporal query handling
│   └── worker-safety/       # Operation safety hard limits
├── memory/                  # Daily logs and long-term memory
└── evolution_log.md         # Auto-generated evolution diary
```

---

## 🔄 Evolution Loop

Every 2 hours, the system automatically:
1. Assesses current capabilities and gaps
2. Identifies learning priorities
3. Generates code / documentation / skills
4. Commits and pushes to GitHub

---

## 🎯 Current Focus

- **Flash Attention CUDA optimization**: 16-kernel iteration for AI Infra interview prep
- **HIP/AMD GPU portability**: Dual CUDA+HIP kernels
- **Kernel performance analysis**: Roofline model, bound identification, profiler-guided optimization

---

## 📊 Performance Benchmarks

| Kernel | Technique | Performance (RTX 4080) |
|--------|-----------|------------------------|
| v1 Naive | Global memory only | 0.51 TFLOPS |
| v2 Tiling | Shared memory (failed) | 0.26 TFLOPS |
| v3 Cooperative | 8 queries share K/V tile | TBD |

---

## 📚 References

- [Flash Attention 2 Paper](https://arxiv.org/abs/2307.08691)
- [Flash Attention Tutorial](https://lubits.ch/flash/)
- [CUTLASS](https://github.com/NVIDIA/cutlass)

---

*Last updated: 2026-04-21 by Kraber*
