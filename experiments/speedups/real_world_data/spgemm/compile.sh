#!/usr/bin/env bash
# compile.sh — Compile spgemm_dn.mlir (dense output) and spgemm.mlir (CSR
# output) into two binaries each: a plain mlir-opt --sparsifier baseline (no
# Splyce vectorization), and a single Splyce-vectorized binary using
# phase-select 001 (no --splyce-fastmath).
#
# Uses the two bundling flags documented in the repo README's "Usage
# Examples" section instead of spelling out each individual mlir-opt pass:
#   - baseline: `mlir-opt --sparsifier` (sparse dialect -> LLVM dialect,
#     one flag)
#   - Splyce:   `splyce-opt --sparsify-to-scf ... --splyce=... \
#                --lower-to-llvm` (sparse dialect -> SCF -> vectorize ->
#     LLVM dialect, one splyce-opt invocation)
#
# This script only compiles; it's meant to be driven by another script that
# runs/benchmarks the resulting binaries.
#
# Also builds an OpenMP dense-outer-loop parallel variant of each (see
# experiments/multicore/compile.sh for the same pattern): mlir-opt
# --sparsifier has no OpenMP-aware lowering, so the parallel baseline goes
# through splyce-opt too (just without --splyce in between).
#
# Binaries are written to:
#   ./test_benchmark_spgemm_scf[_parallel]
#   ./test_benchmark_spgemm_splyce_phase_001[_parallel]
#   ./test_benchmark_spgemm_csr_scf[_parallel]
#   ./test_benchmark_spgemm_csr_splyce_phase_001[_parallel]
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

TARGET_FUNCTION="spgemm"
PHASE="001"

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

# Compiles one source file into its baseline + Splyce phase_001 binaries.
# $1 = source .mlir (e.g. ./spgemm_dn.mlir), $2 = binary/intermediate-file
# name suffix distinguishing this variant (e.g. "" or "_csr")
build_variant() {
  local src="$1"
  local suffix="$2"
  local stem="${TARGET_FUNCTION}${suffix}"

  # -------------------------------------------------------------------------
  # Baseline (no Splyce vectorization) — --sparsifier bundles sparsification
  # all the way down to LLVM dialect in one flag.
  # -------------------------------------------------------------------------
  echo "[baseline] Sparsifying ${src} directly to LLVM dialect ..."
  mlir-opt "$src" \
    --sparsifier \
    -o "./${stem}_llvm_scf.mlir"

  echo "[baseline] Compiling binary ..."
  compile_llvm_mlir "./${stem}_llvm_scf.mlir" "./test_benchmark_${stem}_scf"

  # -------------------------------------------------------------------------
  # Splyce, phase-select 001, no fastmath — --sparsify-to-scf and
  # --lower-to-llvm bundle the sparse-dialect->SCF and SCF->LLVM-dialect
  # pipelines around the --splyce vectorization pass in one splyce-opt call.
  # -------------------------------------------------------------------------
  echo "[phase=$PHASE] Sparsifying, vectorizing, and lowering ${src} ..."
  "$SPLYCE_OPT" "$src" \
    --sparsify-to-scf \
    "--splyce=target-function=${TARGET_FUNCTION} vector-width=4 phase-select=${PHASE}" \
    --splyce-bufferize-restrict \
    --lower-to-llvm \
    -o "./${stem}_llvm_splyce_phase_${PHASE}.mlir"

  echo "[phase=$PHASE] Compiling binary ..."
  compile_llvm_mlir "./${stem}_llvm_splyce_phase_${PHASE}.mlir" \
    "./test_benchmark_${stem}_splyce_phase_${PHASE}"

  # -------------------------------------------------------------------------
  # Parallel baseline (dense-outer-loop OpenMP, no Splyce vectorization).
  # -------------------------------------------------------------------------
  echo "[baseline_parallel] Sparsifying + parallelizing + lowering ${src} ..."
  "$SPLYCE_OPT" "$src" \
    --sparsify-to-scf --parallelization \
    --splyce-bufferize-restrict \
    --lower-to-llvm --parallelization \
    -o "./${stem}_llvm_scf_parallel.mlir"

  echo "[baseline_parallel] Compiling binary ..."
  compile_llvm_mlir "./${stem}_llvm_scf_parallel.mlir" "./test_benchmark_${stem}_scf_parallel" 1

  # -------------------------------------------------------------------------
  # Parallel Splyce, phase-select 001, no fastmath.
  # -------------------------------------------------------------------------
  echo "[phase=$PHASE parallel] Sparsifying + parallelizing + vectorizing + lowering ${src} ..."
  "$SPLYCE_OPT" "$src" \
    --sparsify-to-scf --parallelization \
    "--splyce=target-function=${TARGET_FUNCTION} vector-width=4 phase-select=${PHASE}" \
    --splyce-bufferize-restrict \
    --lower-to-llvm --parallelization \
    -o "./${stem}_llvm_splyce_phase_${PHASE}_parallel.mlir"

  echo "[phase=$PHASE parallel] Compiling binary ..."
  compile_llvm_mlir "./${stem}_llvm_splyce_phase_${PHASE}_parallel.mlir" \
    "./test_benchmark_${stem}_splyce_phase_${PHASE}_parallel" 1

  # -------------------------------------------------------------------------
  # Cleanup: remove all generated intermediate files, keep only binaries
  # -------------------------------------------------------------------------
  echo "Cleaning up intermediate files ..."
  rm -f \
    "./${stem}_llvm_scf.mlir" \
    "./${stem}_llvm_scf.ll" \
    "./${stem}_llvm_splyce_phase_${PHASE}.mlir" \
    "./${stem}_llvm_splyce_phase_${PHASE}.ll" \
    "./${stem}_llvm_scf_parallel.mlir" \
    "./${stem}_llvm_scf_parallel.ll" \
    "./${stem}_llvm_splyce_phase_${PHASE}_parallel.mlir" \
    "./${stem}_llvm_splyce_phase_${PHASE}_parallel.ll"
}

build_variant "./spgemm_dn.mlir" ""
build_variant "./spgemm.mlir" "_csr"

echo "Done. Binaries:"
for stem in "${TARGET_FUNCTION}" "${TARGET_FUNCTION}_csr"; do
  for cfg in "scf" "splyce_phase_${PHASE}" "scf_parallel" "splyce_phase_${PHASE}_parallel"; do
    echo "  $(pwd)/test_benchmark_${stem}_${cfg}"
  done
done
