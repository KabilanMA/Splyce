#!/usr/bin/env bash
# compile.sh — Compile spmttkrp.mlir (this directory) for parallel
# outer-loop execution: a plain (no Splyce) parallel baseline, and a
# Splyce-vectorized parallel binary using phase-select 001 (no
# --splyce-fastmath). Both use the OpenMP dense-outer-loop parallelization
# strategy — README.md "Example 2: Multi-Threaded SpGEMM (OpenMP)".
#
# --sparsifier (mlir-opt's one-shot baseline flag, used for the
# single-threaded binaries elsewhere in this repo) has no OpenMP-aware
# lowering stage, so it can't produce a parallel binary on its own — both
# binaries here instead go through splyce-opt's bundled --sparsify-to-scf /
# --lower-to-llvm flags with --parallelization added (the baseline simply
# omits --splyce in between). --splyce-bufferize-restrict is required for
# both, independent of --splyce: --parallelization's dense-outer-loop
# sparsification emits bufferization.to_tensor ops that need the restrict
# attribute stamped on for --lower-to-llvm's One-Shot Bufferize to accept
# them, whether or not Splyce ran in between.
#
# This script only compiles; it's meant to be driven by another script that
# runs/benchmarks the resulting binaries.
#
# Binaries are written to:
#   ./test_benchmark_spmttkrp_scf_parallel
#   ./test_benchmark_spmttkrp_splyce_phase_001_parallel
#
# Prerequisites: LLVM_INSTALL set, mlir-opt/clang on $PATH, build/bin/splyce-opt
# and build/bin/splyce-translate built (see repo README). clang must be able
# to find libomp (bundled with this repo's LLVM build) for -fopenmp to link.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SPLYCE_OPT="$REPO_ROOT/build/bin/splyce-opt"
SPLYCE_TRANSLATE="$REPO_ROOT/build/bin/splyce-translate"

SRC="./spmttkrp.mlir"
TARGET_FUNCTION="spmttkrp"
PHASE="001"

# clang auto-adds its own runtime lib dir (e.g. lib/<target-triple>/) to
# the link-time search path, so -fopenmp always links — but that dir isn't
# necessarily $LLVM_INSTALL/lib (the rpath below), so without this the
# resulting binaries can silently depend on whatever libomp.so (if any)
# happens to already be on the *running* machine's linker search path at
# runtime, instead of the one they were actually built against.
LIBOMP_PATH="$(clang -print-file-name=libomp.so)"
LIBOMP_RPATH_FLAGS=()
if [[ "$LIBOMP_PATH" == /* && -f "$LIBOMP_PATH" ]]; then
  LIBOMP_RPATH_FLAGS=(-Wl,-rpath,"$(dirname "$LIBOMP_PATH")")
else
  echo "WARNING: clang could not locate libomp.so — the compiled binaries may fail to run unless a system libomp is already installed." >&2
fi

CLANG_FLAGS=(
  -O3 -march=native -fno-vectorize -fno-slp-vectorize -fopenmp
  -L"${LLVM_INSTALL}/lib" -lmlir_c_runner_utils -lmlir_runner_utils
  -Wl,-rpath,"${LLVM_INSTALL}/lib"
  "${LIBOMP_RPATH_FLAGS[@]}"
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
# Baseline: dense-outer-loop parallel, no Splyce vectorization.
# ---------------------------------------------------------------------------
echo "[baseline] Sparsifying + parallelizing + lowering ${SRC} ..."
"$SPLYCE_OPT" "$SRC" \
  --sparsify-to-scf --parallelization \
  --splyce-bufferize-restrict \
  --lower-to-llvm --parallelization \
  -o "./${TARGET_FUNCTION}_llvm_scf_parallel.mlir"

echo "[baseline] Compiling binary ..."
compile_llvm_mlir "./${TARGET_FUNCTION}_llvm_scf_parallel.mlir" \
  "./test_benchmark_${TARGET_FUNCTION}_scf_parallel"

# ---------------------------------------------------------------------------
# Splyce, phase-select 001, no fastmath, dense-outer-loop parallel.
# ---------------------------------------------------------------------------
echo "[phase=$PHASE] Sparsifying + parallelizing + vectorizing + lowering ${SRC} ..."
"$SPLYCE_OPT" "$SRC" \
  --sparsify-to-scf --parallelization \
  "--splyce=target-function=${TARGET_FUNCTION} vector-width=4 phase-select=${PHASE}" \
  --splyce-bufferize-restrict \
  --lower-to-llvm --parallelization \
  -o "./${TARGET_FUNCTION}_llvm_splyce_phase_${PHASE}_parallel.mlir"

echo "[phase=$PHASE] Compiling binary ..."
compile_llvm_mlir "./${TARGET_FUNCTION}_llvm_splyce_phase_${PHASE}_parallel.mlir" \
  "./test_benchmark_${TARGET_FUNCTION}_splyce_phase_${PHASE}_parallel"

# ---------------------------------------------------------------------------
# Cleanup: remove all generated intermediate files, keep only binaries
# ---------------------------------------------------------------------------
echo "Cleaning up intermediate files ..."
rm -f \
  "./${TARGET_FUNCTION}_llvm_scf_parallel.mlir" \
  "./${TARGET_FUNCTION}_llvm_scf_parallel.ll" \
  "./${TARGET_FUNCTION}_llvm_splyce_phase_${PHASE}_parallel.mlir" \
  "./${TARGET_FUNCTION}_llvm_splyce_phase_${PHASE}_parallel.ll"

echo "Done. Binaries:"
echo "  $(pwd)/test_benchmark_${TARGET_FUNCTION}_scf_parallel"
echo "  $(pwd)/test_benchmark_${TARGET_FUNCTION}_splyce_phase_${PHASE}_parallel"
