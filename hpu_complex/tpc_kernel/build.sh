#!/bin/bash
# Run inside the Gaudi pod (needs habanalabs-tools: tpc-clang).
set -euo pipefail
cd "$(dirname "$0")"
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
echo "Built: build/libfft_tpc_perf_lib.so"
echo "Use:   GC_KERNEL_PATH=$PWD/build/libfft_tpc_perf_lib.so:/usr/lib/habanalabs/libtpc_kernels.so"
