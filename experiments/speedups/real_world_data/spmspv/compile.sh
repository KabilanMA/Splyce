#!/usr/bin/env bash
# compile.sh — Compile spmspv.mlir (this directory) into two binaries: a plain
# mlir-opt --sparsifier baseline (no Splyce vectorization), and a single
# Splyce-vectorized binary using phase-select 001 (no --splyce-fastmath).
#
# This script only compiles; it's meant to be driven by another script that
# runs/benchmarks the resulting binaries.
#
# Also builds an OpenMP dense-outer-loop parallel variant of each (see
# experiments/multicore/compile.sh for the same pattern): unlike the
# single-threaded pair above (which unbundles each mlir-opt/splyce-opt
# stage to match this file's existing style), the parallel pair uses
# splyce-opt's bundled --sparsify-to-scf/--lower-to-llvm flags with
# --parallelization added, since mlir-opt --sparsifier has no OpenMP-aware
# lowering and hand-unbundling the OpenMP lowering pipeline isn't worth the
# duplication — the bundled flags are documented as exactly equivalent.
#
# Binaries are written to:
#   ./test_benchmark_spmspv_scf[_parallel]
#   ./test_benchmark_spmspv_splyce_phase_001[_parallel]
#
# Prerequisites: LLVM_INSTALL set, mlir-opt/clang on $PATH, build/bin/splyce-opt
# and build/bin/splyce-translate built (see repo README). The parallel
# variants additionally need clang able to find libomp (-fopenmp).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
SPLYCE_OPT="$REPO_ROOT/build/bin/splyce-opt"
SPLYCE_TRANSLATE="$REPO_ROOT/build/bin/splyce-translate"

SRC="./spmspv.mlir"
TARGET_FUNCTION="spmspv"
PHASE="001"

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

# -fopenmp always links (clang finds its own libomp at link time), but
# rpath it explicitly too or the binary may pick up a different
# libomp.so — or none — at runtime (see playground/run.sh).
LIBOMP_PATH="$(clang -print-file-name=libomp.so)"
LIBOMP_RPATH_FLAGS=()
if [[ "$LIBOMP_PATH" == /* && -f "$LIBOMP_PATH" ]]; then
  LIBOMP_RPATH_FLAGS=(-Wl,-rpath,"$(dirname "$LIBOMP_PATH")")
else
  echo "WARNING: clang could not locate libomp.so — parallel binaries may fail to run unless a system libomp is already installed." >&2
fi
CLANG_FLAGS_PARALLEL=("${CLANG_FLAGS[@]}" -fopenmp "${LIBOMP_RPATH_FLAGS[@]}")

# $1 = input LLVM-dialect .mlir, $2 = output binary path, $3 = 1 for the
# OpenMP-linked parallel flag set, omitted/0 for single-threaded.
compile_llvm_mlir() {
  local llvm_mlir="$1"
  local out_bin="$2"
  local parallel="${3:-0}"
  local ll="${llvm_mlir%.mlir}.ll"
  "$SPLYCE_TRANSLATE" "$llvm_mlir" --mlir-to-llvmir -o "$ll"
  if (( parallel )); then
    clang "${CLANG_FLAGS_PARALLEL[@]}" "$ll" -o "$out_bin"
  else
    clang "${CLANG_FLAGS[@]}" "$ll" -o "$out_bin"
  fi
}

# ---------------------------------------------------------------------------
# 1. Sparsify -> SCF (input for the Splyce phase-select variant below)
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
# 3. Splyce, phase-select 001, no fastmath
# ---------------------------------------------------------------------------
echo "[phase=$PHASE] Applying Splyce vectorization ..."
"$SPLYCE_OPT" "./${TARGET_FUNCTION}_scf.mlir" \
  "--splyce=target-function=${TARGET_FUNCTION} vector-width=4 phase-select=${PHASE}" \
  --splyce-bufferize-restrict \
  -o "./${TARGET_FUNCTION}_splyce_phase_${PHASE}.mlir"

echo "[phase=$PHASE] Lowering to LLVM dialect ..."
mlir-opt "./${TARGET_FUNCTION}_splyce_phase_${PHASE}.mlir" \
  "${COMMON_LOWER_FLAGS[@]}" \
  -o "./${TARGET_FUNCTION}_llvm_splyce_phase_${PHASE}.mlir"

echo "[phase=$PHASE] Compiling binary ..."
compile_llvm_mlir "./${TARGET_FUNCTION}_llvm_splyce_phase_${PHASE}.mlir" \
  "./test_benchmark_${TARGET_FUNCTION}_splyce_phase_${PHASE}"

# ---------------------------------------------------------------------------
# 4. Parallel baseline (dense-outer-loop OpenMP, no Splyce vectorization).
# ---------------------------------------------------------------------------
echo "[baseline_parallel] Sparsifying + parallelizing + lowering ${SRC} ..."
"$SPLYCE_OPT" "$SRC" \
  --sparsify-to-scf --parallelization \
  --splyce-bufferize-restrict \
  --lower-to-llvm --parallelization \
  -o "./${TARGET_FUNCTION}_llvm_scf_parallel.mlir"

echo "[baseline_parallel] Compiling binary ..."
compile_llvm_mlir "./${TARGET_FUNCTION}_llvm_scf_parallel.mlir" \
  "./test_benchmark_${TARGET_FUNCTION}_scf_parallel" 1

# ---------------------------------------------------------------------------
# 5. Parallel Splyce, phase-select 001, no fastmath.
# ---------------------------------------------------------------------------
echo "[phase=$PHASE parallel] Sparsifying + parallelizing + vectorizing + lowering ${SRC} ..."
"$SPLYCE_OPT" "$SRC" \
  --sparsify-to-scf --parallelization \
  "--splyce=target-function=${TARGET_FUNCTION} vector-width=4 phase-select=${PHASE}" \
  --splyce-bufferize-restrict \
  --lower-to-llvm --parallelization \
  -o "./${TARGET_FUNCTION}_llvm_splyce_phase_${PHASE}_parallel.mlir"

echo "[phase=$PHASE parallel] Compiling binary ..."
compile_llvm_mlir "./${TARGET_FUNCTION}_llvm_splyce_phase_${PHASE}_parallel.mlir" \
  "./test_benchmark_${TARGET_FUNCTION}_splyce_phase_${PHASE}_parallel" 1

# ---------------------------------------------------------------------------
# Cleanup: remove all generated intermediate files, keep only binaries
# ---------------------------------------------------------------------------
echo "Cleaning up intermediate files ..."
rm -f \
  "./${TARGET_FUNCTION}_scf.mlir" \
  "./${TARGET_FUNCTION}_llvm_scf.mlir" \
  "./${TARGET_FUNCTION}_llvm_scf.ll" \
  "./${TARGET_FUNCTION}_splyce_phase_${PHASE}.mlir" \
  "./${TARGET_FUNCTION}_llvm_splyce_phase_${PHASE}.mlir" \
  "./${TARGET_FUNCTION}_llvm_splyce_phase_${PHASE}.ll" \
  "./${TARGET_FUNCTION}_llvm_scf_parallel.mlir" \
  "./${TARGET_FUNCTION}_llvm_scf_parallel.ll" \
  "./${TARGET_FUNCTION}_llvm_splyce_phase_${PHASE}_parallel.mlir" \
  "./${TARGET_FUNCTION}_llvm_splyce_phase_${PHASE}_parallel.ll"

echo "Done. Binaries:"
for cfg in "scf" "splyce_phase_${PHASE}" "scf_parallel" "splyce_phase_${PHASE}_parallel"; do
  echo "  $(pwd)/test_benchmark_${TARGET_FUNCTION}_${cfg}"
done
