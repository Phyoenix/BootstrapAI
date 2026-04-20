#!/bin/bash
# =============================================================================
# build_hip.sh — Build Flash Attention HIP kernels
#
# Prerequisites:
#   - ROCm 5.x+ installed  (provides hipcc)
#   - AMD GPU (gfx90a for MI200, gfx1100 for RX 7900 XT, gfx906 for MI50...)
#   - Or: ROCm hipcc with --offload-arch for syntax verification only
#
# Usage:
#   ./build_hip.sh                # auto-detect GPU arch
#   ./build_hip.sh --no-gpu       # syntax-only, no GPU required
#   ./build_hip.sh --selftest     # build & run self-test
#   ./build_hip.sh --arch gfx1100 # specify target arch explicitly
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------
NO_GPU=false
SELFTEST=false
ARCH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-gpu)    NO_GPU=true      ;;
        --selftest)  SELFTEST=true    ;;
        --arch)      ARCH="$2"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

# --------------------------------------------------------------------------
# Detect hipcc
# --------------------------------------------------------------------------
if ! command -v hipcc &>/dev/null; then
    echo "ERROR: hipcc not found. Install ROCm from https://rocm.docs.amd.com/"
    exit 1
fi

HIPCC_VERSION=$(hipcc --version 2>&1 | head -1)
echo "Compiler : $HIPCC_VERSION"

# --------------------------------------------------------------------------
# Detect GPU arch (if not specified)
# --------------------------------------------------------------------------
if [[ -z "$ARCH" ]] && ! $NO_GPU; then
    if command -v rocminfo &>/dev/null; then
        ARCH=$(rocminfo 2>/dev/null | awk '/Name:/{last=$2} /gfx/{print last; exit}' || true)
    fi
    if [[ -z "$ARCH" ]]; then
        echo "WARN: Cannot detect GPU arch; defaulting to gfx1100 (RX 7900 series)"
        ARCH="gfx1100"
    fi
fi

if $NO_GPU; then
    ARCH_FLAG="--offload-arch=gfx1100"
else
    ARCH_FLAG="--offload-arch=${ARCH}"
fi

echo "Target   : ${ARCH_FLAG}"

# --------------------------------------------------------------------------
# Common compile flags
# --------------------------------------------------------------------------
CFLAGS="-O3 -std=c++17 ${ARCH_FLAG} -I./include"

# --------------------------------------------------------------------------
# Build: Kernel 01 HIP (object only — part of larger binary)
# --------------------------------------------------------------------------
echo ""
echo "=== Building kernel_01_naive.hip ==="
hipcc $CFLAGS -c kernels/kernel_01_naive.hip -o build/kernel_01_naive_hip.o
echo "    OK → build/kernel_01_naive_hip.o"

# --------------------------------------------------------------------------
# Build: Self-test standalone binary
# --------------------------------------------------------------------------
if $SELFTEST; then
    echo ""
    echo "=== Building self-test binary ==="
    hipcc $CFLAGS -DHIP_SELFTEST kernels/kernel_01_naive.hip -o build/flash_hip_selftest
    echo "    OK → build/flash_hip_selftest"

    if ! $NO_GPU; then
        echo ""
        echo "=== Running self-test ==="
        ./build/flash_hip_selftest
    else
        echo "    (Skipping run: --no-gpu mode)"
    fi
fi

# --------------------------------------------------------------------------
# Build: AMD Micro-benchmarks
# --------------------------------------------------------------------------
MICROBENCH_DIR="amd-microbench"
if [[ -d "$MICROBENCH_DIR" ]]; then
    echo ""
    echo "=== Building AMD micro-benchmarks ==="
    mkdir -p build/microbench

    for src in "${MICROBENCH_DIR}"/*.hip; do
        base=$(basename "$src" .hip)
        hipcc $CFLAGS "$src" -o "build/microbench/${base}"
        echo "    OK → build/microbench/${base}"
    done
fi

echo ""
echo "=== All HIP builds completed successfully ==="
echo ""
echo "Next steps:"
echo "  Run self-test:        ./build/flash_hip_selftest"
echo "  Run memory bench:     ./build/microbench/memory_bandwidth"
echo "  Run compute bench:    ./build/microbench/compute_throughput"
echo "  Run occupancy bench:  ./build/microbench/occupancy_test"
