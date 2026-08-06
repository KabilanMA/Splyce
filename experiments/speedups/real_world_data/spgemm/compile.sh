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
# Binaries are written to:
#   ./test_benchmark_spgemm_scf
#   ./test_benchmark_spgemm_splyce_phase_001
#   ./test_benchmark_spgemm_csr_scf
#   ./test_benchmark_spgemm_csr_splyce_phase_001
#
# Prerequisites: LLVM_INSTALL set, mlir-opt/clang on $PATH, build/bin/splyce-opt
# and build/bin/splyce-translate built (see repo README).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
SPLYCE_OPT="$REPO_ROOT/build/bin/splyce-opt"
SPLYCE_TRANSLATE="$REPO_ROOT/build/bin/splyce-translate"

TARGET_FUNCTION="spgemm"
PHASE="001"

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
  # Cleanup: remove all generated intermediate files, keep only binaries
  # -------------------------------------------------------------------------
  echo "Cleaning up intermediate files ..."
  rm -f \
    "./${stem}_llvm_scf.mlir" \
    "./${stem}_llvm_scf.ll" \
    "./${stem}_llvm_splyce_phase_${PHASE}.mlir" \
    "./${stem}_llvm_splyce_phase_${PHASE}.ll"
}

build_variant "./spgemm_dn.mlir" ""
build_variant "./spgemm.mlir" "_csr"

echo "Done. Binaries:"
echo "  $(pwd)/test_benchmark_${TARGET_FUNCTION}_scf"
echo "  $(pwd)/test_benchmark_${TARGET_FUNCTION}_splyce_phase_${PHASE}"
echo "  $(pwd)/test_benchmark_${TARGET_FUNCTION}_csr_scf"
echo "  $(pwd)/test_benchmark_${TARGET_FUNCTION}_csr_splyce_phase_${PHASE}"
