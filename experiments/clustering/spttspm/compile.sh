#!/usr/bin/env bash
# compile.sh — Compile spttspm_splyce_scf.mlir (already Splyce-vectorized
# SCF dialect) down to a runnable binary.
#
# Unlike experiments/speedups/*/spttspm/compile.sh, this doesn't run
# sparsification or the --splyce vectorization pass itself — the input is
# already past that stage (see spttspm_splyce_scf.mlir). This script just
# runs the remaining SCF -> LLVM dialect -> LLVM IR -> binary steps:
#   splyce-opt --splyce-bufferize-restrict --lower-to-llvm
#   splyce-translate --mlir-to-llvmir
#   clang -O3
#
# Output: ./spttspm (plus the intermediate
# spttspm_llvm.mlir/spttspm_splyce.ll)
#
# Prerequisites: LLVM_INSTALL set, clang on $PATH, build/bin/splyce-opt
# and build/bin/splyce-translate already built (see repo README).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
SPLYCE_OPT="$REPO_ROOT/build/bin/splyce-opt"
SPLYCE_TRANSLATE="$REPO_ROOT/build/bin/splyce-translate"

if [ -z "${LLVM_INSTALL:-}" ]; then
  echo "error: LLVM_INSTALL is not set" >&2
  exit 1
fi

CLANG_FLAGS=(
  -O3 -march=native -fno-vectorize -fno-slp-vectorize
  -L"${LLVM_INSTALL}/lib" -lmlir_c_runner_utils -lmlir_runner_utils
  -Wl,-rpath,"${LLVM_INSTALL}/lib"
)

echo "[lower] Bufferizing + lowering spttspm_splyce_scf.mlir to LLVM dialect ..."
"$SPLYCE_OPT" ./spttspm_splyce_scf.mlir \
  --splyce-bufferize-restrict \
  --lower-to-llvm \
  -o ./spttspm_llvm.mlir

echo "[translate] Translating to LLVM IR ..."
"$SPLYCE_TRANSLATE" ./spttspm_llvm.mlir \
  --mlir-to-llvmir \
  -o ./spttspm_splyce.ll

echo "[compile] Compiling binary ..."
clang "${CLANG_FLAGS[@]}" ./spttspm_splyce.ll -o ./spttspm

echo "Done. Binary: $(pwd)/spttspm"
