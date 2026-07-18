#!/usr/bin/env bash

set -euo pipefail

# =============================================================================
# Generic MLIR application build script
#
# Usage:
#   ./compile_mlir_application.sh <application> [normal|optimized|both] [--run]
#
# Examples:
#   ./compile_mlir_application.sh spttspm
#   ./compile_mlir_application.sh spttspm normal
#   ./compile_mlir_application.sh spttspm optimized --run
#   ./compile_mlir_application.sh spmm both
#
# The input file is expected at:
#   playground/<application>.mlir
#
# Optional environment variables:
#   LLVM_INSTALL=/path/to/llvm/build
#   PROJECT_ROOT=/path/to/Splyce
#   CLANG=/path/to/clang
#   TARGET_FUNCTION=<MLIR function name>
#   VECTOR_WIDTH=4
#   PHASE_SELECT=001
#
# By default, TARGET_FUNCTION is set to the application name.
# =============================================================================

LLVM_INSTALL="${LLVM_INSTALL:-/home/grads/poornag/llvm-project/build}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/grads/poornag/Splyce/}"
CLANG="${CLANG:-clang}"

VECTOR_WIDTH="${VECTOR_WIDTH:-4}"
PHASE_SELECT="${PHASE_SELECT:-101}"

usage() {
    cat <<EOF
Usage:
  $0 <application> [normal|optimized|both] [--run]

Arguments:
  application   Input MLIR filename without the .mlir extension.
                Example: "spttspm" refers to playground/spttspm.mlir.

  build type    normal, optimized, or both.
                Default: both

  --run         Run each executable after compiling it.

Examples:
  $0 spttspm
  $0 spttspm normal
  $0 spttspm optimized --run
  $0 spmm both

Optional environment variables:
  LLVM_INSTALL
  PROJECT_ROOT
  CLANG
  TARGET_FUNCTION
  VECTOR_WIDTH
  PHASE_SELECT
EOF
}

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

APPLICATION="$1"
shift

BUILD_TYPE="both"
RUN_AFTER_BUILD=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        normal|baseline)
            BUILD_TYPE="normal"
            ;;
        optimized|opt)
            BUILD_TYPE="optimized"
            ;;
        both)
            BUILD_TYPE="both"
            ;;
        --run)
            RUN_AFTER_BUILD=true
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
    shift
done

TARGET_FUNCTION="${TARGET_FUNCTION:-${APPLICATION}}"

MLIR_OPT="${LLVM_INSTALL}/bin/mlir-opt"
MLIR_TRANSLATE="${LLVM_INSTALL}/bin/mlir-translate"
SPLYCE_OPT="${PROJECT_ROOT}/build/bin/splyce-opt"

PLAYGROUND_DIR="${PROJECT_ROOT}/playground"

INPUT_MLIR="${PLAYGROUND_DIR}/${APPLICATION}.mlir"
SCF_MLIR="${PLAYGROUND_DIR}/${APPLICATION}_scf.mlir"

NORMAL_LLVM_MLIR="${PLAYGROUND_DIR}/${APPLICATION}_llvm.mlir"
NORMAL_LLVM_IR="${PLAYGROUND_DIR}/${APPLICATION}.ll"
NORMAL_EXECUTABLE="${PLAYGROUND_DIR}/test_benchmark_${APPLICATION}"

SPLYCE_MLIR="${PLAYGROUND_DIR}/${APPLICATION}_splyce.mlir"
OPT_LLVM_MLIR="${PLAYGROUND_DIR}/${APPLICATION}_llvm_opt.mlir"
OPT_LLVM_IR="${PLAYGROUND_DIR}/${APPLICATION}_opt.ll"
OPT_EXECUTABLE="${PLAYGROUND_DIR}/test_benchmark_${APPLICATION}_optimized"

check_executable() {
    local executable="$1"

    if [[ ! -x "${executable}" ]]; then
        echo "Error: required executable not found: ${executable}" >&2
        exit 1
    fi
}

check_file() {
    local file="$1"

    if [[ ! -f "${file}" ]]; then
        echo "Error: required input file not found: ${file}" >&2
        exit 1
    fi
}

check_executable "${MLIR_OPT}"
check_executable "${MLIR_TRANSLATE}"

if ! command -v "${CLANG}" >/dev/null 2>&1; then
    echo "Error: clang executable not found: ${CLANG}" >&2
    exit 1
fi

if [[ "${BUILD_TYPE}" == "optimized" || "${BUILD_TYPE}" == "both" ]]; then
    check_executable "${SPLYCE_OPT}"
fi

check_file "${INPUT_MLIR}"
mkdir -p "${PLAYGROUND_DIR}"

lower_to_scf() {
    echo
    echo "============================================================"
    echo "Lowering ${APPLICATION}.mlir to SCF"
    echo "============================================================"

    "${MLIR_OPT}" "${INPUT_MLIR}" \
        --linalg-generalize-named-ops \
        --pre-sparsification-rewrite \
        --empty-tensor-to-alloc-tensor \
        --sparse-reinterpret-map \
        --sparsification \
        --stage-sparse-ops \
        --lower-sparse-ops-to-foreach \
        --lower-sparse-foreach-to-scf \
        --sparse-tensor-conversion \
        -o "${SCF_MLIR}"

    echo "Generated: ${SCF_MLIR}"
}

lower_to_llvm() {
    local input_mlir="$1"
    local output_mlir="$2"

    "${MLIR_OPT}" "${input_mlir}" \
        --one-shot-bufferize="bufferize-function-boundaries=true allow-return-allocs-from-loops=true" \
        --convert-bufferization-to-memref \
        --convert-linalg-to-loops \
        --convert-scf-to-cf \
        --lower-vector-mask \
        --convert-vector-to-scf \
        --expand-realloc \
        --finalize-memref-to-llvm \
        --convert-vector-to-llvm \
        --convert-math-to-llvm \
        --convert-arith-to-llvm \
        --convert-func-to-llvm \
        --convert-cf-to-llvm \
        --reconcile-unrealized-casts \
        -o "${output_mlir}"
}

translate_to_llvm_ir() {
    local input_mlir="$1"
    local output_ll="$2"

    "${MLIR_TRANSLATE}" "${input_mlir}" \
        --mlir-to-llvmir \
        -o "${output_ll}"
}

compile_llvm_ir() {
    local input_ll="$1"
    local executable="$2"

    "${CLANG}" -O3 "${input_ll}" \
        -mavx512f \
        -mavx512vl \
        -fno-vectorize \
        -fno-slp-vectorize \
        -L"${LLVM_INSTALL}/lib" \
        -lmlir_c_runner_utils \
        -lmlir_runner_utils \
        -Wl,-rpath,"${LLVM_INSTALL}/lib" \
        -o "${executable}"
}

build_normal() {
    echo
    echo "============================================================"
    echo "Building normal version: ${APPLICATION}"
    echo "============================================================"

    echo "[1/3] Lowering SCF to LLVM dialect..."
    lower_to_llvm "${SCF_MLIR}" "${NORMAL_LLVM_MLIR}"

    echo "[2/3] Translating LLVM dialect to LLVM IR..."
    translate_to_llvm_ir "${NORMAL_LLVM_MLIR}" "${NORMAL_LLVM_IR}"

    echo "[3/3] Compiling executable..."
    compile_llvm_ir "${NORMAL_LLVM_IR}" "${NORMAL_EXECUTABLE}"

    echo
    echo "Normal build completed."
    echo "Executable: ${NORMAL_EXECUTABLE}"

    if [[ "${RUN_AFTER_BUILD}" == true ]]; then
        echo
        echo "Running normal executable..."
        "${NORMAL_EXECUTABLE}"
    fi
}

build_optimized() {
    echo
    echo "============================================================"
    echo "Building Splyce-optimized version: ${APPLICATION}"
    echo "============================================================"

    echo "[1/4] Applying Splyce optimization..."
    "${SPLYCE_OPT}" "${SCF_MLIR}" \
        --splyce="target-function=${TARGET_FUNCTION} vector-width=${VECTOR_WIDTH} phase-select=${PHASE_SELECT}" \
        --splyce-bufferize-restrict \
        -o "${SPLYCE_MLIR}"

    echo "[2/4] Lowering optimized IR to LLVM dialect..."
    lower_to_llvm "${SPLYCE_MLIR}" "${OPT_LLVM_MLIR}"

    echo "[3/4] Translating LLVM dialect to LLVM IR..."
    translate_to_llvm_ir "${OPT_LLVM_MLIR}" "${OPT_LLVM_IR}"

    echo "[4/4] Compiling optimized executable..."
    compile_llvm_ir "${OPT_LLVM_IR}" "${OPT_EXECUTABLE}"

    echo
    echo "Optimized build completed."
    echo "Executable: ${OPT_EXECUTABLE}"
    echo "Target function: ${TARGET_FUNCTION}"
    echo "Vector width: ${VECTOR_WIDTH}"
    echo "Phase select: ${PHASE_SELECT}"

    if [[ "${RUN_AFTER_BUILD}" == true ]]; then
        echo
        echo "Running optimized executable..."
        "${OPT_EXECUTABLE}"
    fi
}

echo "Application:     ${APPLICATION}"
echo "Input file:      ${INPUT_MLIR}"
echo "Build type:      ${BUILD_TYPE}"
echo "Target function: ${TARGET_FUNCTION}"

lower_to_scf

case "${BUILD_TYPE}" in
    normal)
        build_normal
        ;;
    optimized)
        build_optimized
        ;;
    both)
        build_normal
        build_optimized
        ;;
esac

echo
echo "============================================================"
echo "Build process completed successfully."
echo "============================================================"
