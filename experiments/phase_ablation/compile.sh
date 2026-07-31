#!/usr/bin/env bash
# compile.sh — Compile spgemm.mlir (this directory) into:
#   - a plain mlir-opt baseline binary (no Splyce vectorization)
#   - one Splyce-vectorized binary per phase-select combination (000-111)
#   - the same 8 phase-select combinations again, with --splyce-fastmath
#
# This script only compiles; it's meant to be driven by another script that
# runs/benchmarks the resulting binaries.
#
# Binaries are written to:
#   ./test_benchmark_spgemm_scf
#   ./test_benchmark_spgemm_splyce_phase_<XYZ>
#   ./test_benchmark_spgemm_splyce_phase_<XYZ>_fastmath
#
# Prerequisites: LLVM_INSTALL set, mlir-opt/clang on $PATH, build/bin/splyce-opt
# and build/bin/splyce-translate built (see repo README).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SPLYCE_OPT="$REPO_ROOT/build/bin/splyce-opt"
SPLYCE_TRANSLATE="$REPO_ROOT/build/bin/splyce-translate"

SRC="./spgemm.mlir"

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
  -O3 -mavx512f -mavx512vl -fno-vectorize -fno-slp-vectorize
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
# 1. Sparsify spgemm.mlir -> SCF (shared input for the baseline and every
#    Splyce phase-select variant below)
# ---------------------------------------------------------------------------
echo "[sparsify] Lowering spgemm.mlir to SCF ..."
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
  -o ./spgemm_scf.mlir

# ---------------------------------------------------------------------------
# 2. Plain mlir-opt baseline (no Splyce vectorization) — --sparsifier bundles
#    sparsification all the way down to LLVM dialect in one flag.
# ---------------------------------------------------------------------------
echo "[baseline] Sparsifying spgemm.mlir directly to LLVM dialect ..."
mlir-opt "$SRC" \
  --sparsifier \
  -o ./spgemm_llvm_scf.mlir

echo "[baseline] Compiling binary ..."
compile_llvm_mlir ./spgemm_llvm_scf.mlir ./test_benchmark_spgemm_scf

# ---------------------------------------------------------------------------
# 3. Splyce, all 8 phase-select combinations, with and without --splyce-fastmath
# ---------------------------------------------------------------------------
PHASES=(000 001 010 011 100 101 110 111)

for phase in "${PHASES[@]}"; do
  echo "[phase=$phase] Applying Splyce vectorization ..."
  "$SPLYCE_OPT" ./spgemm_scf.mlir \
    "--splyce=target-function=spgemm vector-width=4 phase-select=${phase}" \
    --splyce-bufferize-restrict \
    -o "./spgemm_splyce_phase_${phase}.mlir"

  echo "[phase=$phase] Lowering to LLVM dialect ..."
  mlir-opt "./spgemm_splyce_phase_${phase}.mlir" \
    "${COMMON_LOWER_FLAGS[@]}" \
    -o "./spgemm_llvm_splyce_phase_${phase}.mlir"

  echo "[phase=$phase] Compiling binary ..."
  compile_llvm_mlir "./spgemm_llvm_splyce_phase_${phase}.mlir" \
    "./test_benchmark_spgemm_splyce_phase_${phase}"

  echo "[phase=$phase, fastmath] Applying Splyce vectorization + fastmath ..."
  "$SPLYCE_OPT" ./spgemm_scf.mlir \
    "--splyce=target-function=spgemm vector-width=4 phase-select=${phase}" \
    --splyce-bufferize-restrict \
    --splyce-fastmath \
    -o "./spgemm_splyce_phase_${phase}_fastmath.mlir"

  echo "[phase=$phase, fastmath] Lowering to LLVM dialect ..."
  mlir-opt "./spgemm_splyce_phase_${phase}_fastmath.mlir" \
    "${COMMON_LOWER_FLAGS[@]}" \
    -o "./spgemm_llvm_splyce_phase_${phase}_fastmath.mlir"

  echo "[phase=$phase, fastmath] Compiling binary ..."
  compile_llvm_mlir "./spgemm_llvm_splyce_phase_${phase}_fastmath.mlir" \
    "./test_benchmark_spgemm_splyce_phase_${phase}_fastmath"
done

# ---------------------------------------------------------------------------
# Cleanup: remove all generated intermediate files, keep only binaries
# ---------------------------------------------------------------------------
echo "Cleaning up intermediate files ..."
rm -f ./spgemm_scf.mlir ./spgemm_llvm_scf.mlir ./spgemm_llvm_scf.ll
for phase in "${PHASES[@]}"; do
  rm -f \
    "./spgemm_splyce_phase_${phase}.mlir" \
    "./spgemm_llvm_splyce_phase_${phase}.mlir" \
    "./spgemm_llvm_splyce_phase_${phase}.ll" \
    "./spgemm_splyce_phase_${phase}_fastmath.mlir" \
    "./spgemm_llvm_splyce_phase_${phase}_fastmath.mlir" \
    "./spgemm_llvm_splyce_phase_${phase}_fastmath.ll"
done

echo "Done. Binaries:"
echo "  $(pwd)/test_benchmark_spgemm_scf"
for phase in "${PHASES[@]}"; do
  echo "  $(pwd)/test_benchmark_spgemm_splyce_phase_${phase}"
  echo "  $(pwd)/test_benchmark_spgemm_splyce_phase_${phase}_fastmath"
done
