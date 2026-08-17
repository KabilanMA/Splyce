#!/usr/bin/env bash
# compile.sh — Compile spgemm.mlir (this directory) into:
#   - a plain mlir-opt baseline binary (no Splyce vectorization)
#   - one Splyce-vectorized binary per vector-width in VECTOR_WIDTHS
#     (2, 4, 8, 16), phase-select 001, no --splyce-fastmath — isolating
#     vector-width's effect on its own, with everything else held fixed.
#
# This script only compiles; it's meant to be driven by another script that
# runs/benchmarks the resulting binaries.
#
# Binaries are written to:
#   ./test_benchmark_spgemm_scf
#   ./test_benchmark_spgemm_splyce_vw_<N>
#
# Prerequisites: LLVM_INSTALL set, mlir-opt/clang on $PATH (run the
# load-llvm-dev alias first if they aren't), build/bin/splyce-opt and
# build/bin/splyce-translate built (see repo README).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SPLYCE_OPT="$REPO_ROOT/build/bin/splyce-opt"
SPLYCE_TRANSLATE="$REPO_ROOT/build/bin/splyce-translate"

SRC="./spgemm.mlir"
TARGET_FUNCTION="spgemm"
PHASE="001"
VECTOR_WIDTHS=(2 4 8 16)

COMMON_LOWER_FLAGS=(
  --canonicalize
  --cse
  --loop-invariant-code-motion
  "--one-shot-bufferize=bufferize-function-boundaries=true allow-return-allocs-from-loops=true"
  --convert-bufferization-to-memref
  --lower-vector-mask
  --convert-vector-to-scf
  --canonicalize
  --cse
  --expand-realloc
  --sparse-storage-specifier-to-llvm
  --convert-linalg-to-loops
  --lower-affine
  --canonicalize
  --cse
  --convert-scf-to-cf
  --expand-strided-metadata
  --finalize-memref-to-llvm
  --convert-vector-to-llvm
  --convert-math-to-llvm
  --convert-arith-to-llvm
  --convert-func-to-llvm
  --convert-cf-to-llvm
  --reconcile-unrealized-casts
)

CLANG_FLAGS=(
  -O3 -march=native -fno-vectorize -fno-slp-vectorize
  -L"${LLVM_INSTALL}/lib" -lmlir_c_runner_utils -lmlir_runner_utils
  -Wl,-rpath,"${LLVM_INSTALL}/lib"
)

# $1 = input LLVM-dialect .mlir, $2 = output binary path
compile_llvm_mlir() {
  local llvm_mlir="$1"
  local out_bin="$2"
  local ll="${llvm_mlir%.mlir}.ll"
  "$SPLYCE_TRANSLATE" "$llvm_mlir" --mlir-to-llvmir -o "$ll"
  clang "${CLANG_FLAGS[@]}" "$ll" -o "$out_bin"
}

# ---------------------------------------------------------------------------
# 1. Sparsify -> SCF (shared input for every Splyce vector-width variant
#    below; required regardless, since splyce-opt operates on SCF-level IR)
# ---------------------------------------------------------------------------
echo "[sparsify] Lowering ${SRC} to SCF ..."
mlir-opt "$SRC" \
  --linalg-generalize-named-ops \
  --linalg-fuse-elementwise-ops \
  --pre-sparsification-rewrite \
  --empty-tensor-to-alloc-tensor \
  --sparse-reinterpret-map \
  --sparsification \
  --stage-sparse-ops \
  --lower-sparse-ops-to-foreach \
  --lower-sparse-foreach-to-scf \
  --loop-invariant-code-motion \
  --sparse-tensor-conversion \
  -o "./${TARGET_FUNCTION}_scf.mlir"

# ---------------------------------------------------------------------------
# 2. Plain mlir-opt baseline (no Splyce vectorization) — --sparsifier bundles
#    sparsification all the way down to LLVM dialect in one flag.
# ---------------------------------------------------------------------------
echo "[baseline] Sparsifying ${SRC} directly to LLVM dialect ..."
mlir-opt "$SRC" \
  --sparsifier \
  -o "./${TARGET_FUNCTION}_llvm_scf.mlir"

echo "[baseline] Compiling binary ..."
compile_llvm_mlir "./${TARGET_FUNCTION}_llvm_scf.mlir" "./test_benchmark_${TARGET_FUNCTION}_scf"

# ---------------------------------------------------------------------------
# 3. Splyce, phase-select 001, one binary per vector-width
# ---------------------------------------------------------------------------
for vw in "${VECTOR_WIDTHS[@]}"; do
  echo "[vector-width=$vw] Applying Splyce vectorization ..."
  "$SPLYCE_OPT" "./${TARGET_FUNCTION}_scf.mlir" \
    "--splyce=target-function=${TARGET_FUNCTION} vector-width=${vw} phase-select=${PHASE}" \
    --splyce-bufferize-restrict \
    -o "./${TARGET_FUNCTION}_splyce_vw_${vw}.mlir"

  echo "[vector-width=$vw] Lowering to LLVM dialect ..."
  mlir-opt "./${TARGET_FUNCTION}_splyce_vw_${vw}.mlir" \
    "${COMMON_LOWER_FLAGS[@]}" \
    -o "./${TARGET_FUNCTION}_llvm_splyce_vw_${vw}.mlir"

  echo "[vector-width=$vw] Compiling binary ..."
  compile_llvm_mlir "./${TARGET_FUNCTION}_llvm_splyce_vw_${vw}.mlir" \
    "./test_benchmark_${TARGET_FUNCTION}_splyce_vw_${vw}"
done

# ---------------------------------------------------------------------------
# Cleanup: remove all generated intermediate files, keep only binaries
# ---------------------------------------------------------------------------
echo "Cleaning up intermediate files ..."
rm -f "./${TARGET_FUNCTION}_scf.mlir" "./${TARGET_FUNCTION}_llvm_scf.mlir" "./${TARGET_FUNCTION}_llvm_scf.ll"
for vw in "${VECTOR_WIDTHS[@]}"; do
  rm -f \
    "./${TARGET_FUNCTION}_splyce_vw_${vw}.mlir" \
    "./${TARGET_FUNCTION}_llvm_splyce_vw_${vw}.mlir" \
    "./${TARGET_FUNCTION}_llvm_splyce_vw_${vw}.ll"
done

echo "Done. Binaries:"
echo "  $(pwd)/test_benchmark_${TARGET_FUNCTION}_scf"
for vw in "${VECTOR_WIDTHS[@]}"; do
  echo "  $(pwd)/test_benchmark_${TARGET_FUNCTION}_splyce_vw_${vw}"
done
