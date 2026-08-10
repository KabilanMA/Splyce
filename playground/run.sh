#!/usr/bin/env bash
# run.sh — compile <kernel>.mlir into a baseline and a Splyce binary,
# generate input tensors if missing, run both, print the speedup.
#
# Usage: ./playground/run.sh <kernel> <mode> [cores]
#   kernel   e.g. spgemm — needs a matching <kernel>.mlir in this directory
#   mode     single      mlir-opt --sparsifier baseline vs. splyce-opt
#            multicore   both via splyce-opt --parallelization, pinned to
#                        the same core count (default: nproc, or the full
#                        core count of NUMA_NODE if numactl is available)
#   cores    (multicore only) override the core count
#
# Prerequisites: LLVM_INSTALL set, mlir-opt/mlir-translate/clang on $PATH,
# build/bin/splyce-{opt,translate} built. multicore needs libomp for
# -fopenmp to link.
#
# Uses -march=native, not experiments/'s -mavx512f -mavx512vl — this is a
# quick check on whatever CPU you have, not a controlled benchmark.

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <kernel> <mode> [cores]" >&2
  echo "  kernel   e.g. spgemm — must have a matching playground/<kernel>.mlir" >&2
  echo "  mode     single | multicore" >&2
  echo "  cores    (multicore only) how many CPUs to use" >&2
  exit 1
fi

KERNEL="$1"
MODE="$2"
CORES_ARG="${3:-}"

case "$MODE" in
  single|multicore) ;;
  *) echo "error: unsupported mode '$MODE' (supported: single, multicore)" >&2; exit 1 ;;
esac

if [[ -n "$CORES_ARG" && "$MODE" != "multicore" ]]; then
  echo "error: [cores] is only meaningful for mode 'multicore'" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SPLYCE_OPT="$REPO_ROOT/build/bin/splyce-opt"
SPLYCE_TRANSLATE="$REPO_ROOT/build/bin/splyce-translate"

SRC="$SCRIPT_DIR/${KERNEL}.mlir"
[[ -f "$SRC" ]] || { echo "error: $SRC not found" >&2; exit 1; }

: "${LLVM_INSTALL:?LLVM_INSTALL is not set — see README 'Building and Installation'.}"
for tool in mlir-opt mlir-translate clang; do
  command -v "$tool" >/dev/null 2>&1 || { echo "error: '$tool' not found on \$PATH — see README 'Building and Installation'." >&2; exit 1; }
done
for tool in "$SPLYCE_OPT" "$SPLYCE_TRANSLATE"; do
  [[ -x "$tool" ]] || { echo "error: $tool not found — build it first (see README 'Building and Installation' Step 2)." >&2; exit 1; }
done

cd "$SCRIPT_DIR"

if [[ ! -f tensor_B.tns || ! -f tensor_C.tns ]]; then
  echo "[gen_data] tensor_B.tns/tensor_C.tns not found — generating ..."
  ./gen_data.sh
fi

# ---------------------------------------------------------------------------
# multicore: pick a core count and how to pin to it, mirroring
# experiments/multicore/run.sh (single-shot instead of a core-count sweep).
# ---------------------------------------------------------------------------
RUNNER=()
CORES=""
if [[ "$MODE" == "multicore" ]]; then
  NUMA_NODE="${NUMA_NODE:-0}"
  USE_NUMACTL=0
  NODE_CPUS=()

  if command -v numactl >/dev/null 2>&1 && [[ -d "/sys/devices/system/node/node${NUMA_NODE}" ]]; then
    USE_NUMACTL=1
    # Expands a Linux cpulist ("0,2,4,6-10,12") into one CPU id per line —
    # a node's list isn't necessarily a contiguous range.
    expand_cpulist() {
      local part
      local IFS=','
      for part in $1; do
        if [[ "$part" == *-* ]]; then
          seq "${part%-*}" "${part#*-}"
        else
          echo "$part"
        fi
      done
    }
    mapfile -t NODE_CPUS < <(expand_cpulist "$(cat "/sys/devices/system/node/node${NUMA_NODE}/cpulist")" | sort -n)
  else
    echo "WARNING: numactl unavailable or node${NUMA_NODE} not found — falling back to taskset (no NUMA memory binding)." >&2
  fi

  if [[ -n "$CORES_ARG" ]]; then
    CORES="$CORES_ARG"
  elif (( USE_NUMACTL )); then
    CORES="${#NODE_CPUS[@]}"
  else
    CORES="$(nproc)"
  fi

  if (( USE_NUMACTL )); then
    if (( CORES > ${#NODE_CPUS[@]} )); then
      echo "error: requested ${CORES} core(s) but NUMA node ${NUMA_NODE} only has ${#NODE_CPUS[@]}." >&2
      exit 1
    fi
    PIN_CPUS="$(IFS=,; echo "${NODE_CPUS[*]:0:$CORES}")"
    RUNNER=(numactl --physcpubind="$PIN_CPUS" --membind="$NUMA_NODE")
  else
    PIN_CPUS="0-$((CORES - 1))"
    RUNNER=(taskset -c "$PIN_CPUS")
  fi
  echo "[multicore] Using ${CORES} core(s) (CPUs ${PIN_CPUS}), OMP_NUM_THREADS=${CORES}."
fi

# ---------------------------------------------------------------------------
# Compile
# ---------------------------------------------------------------------------
SUFFIX=""
PARALLEL_FLAGS=()
if [[ "$MODE" == "multicore" ]]; then
  SUFFIX="_parallel"
  PARALLEL_FLAGS=(--parallelization)
fi

CLANG_FLAGS=(-O3 -march=native -fno-vectorize -fno-slp-vectorize)
LIBOMP_RPATH_FLAGS=()
if [[ "$MODE" == "multicore" ]]; then
  CLANG_FLAGS+=(-fopenmp)
  # -fopenmp always links (clang finds its own libomp at link time), but
  # rpath it explicitly too or the binary may pick up a different
  # libomp.so — or none — at runtime.
  LIBOMP_PATH="$(clang -print-file-name=libomp.so)"
  if [[ "$LIBOMP_PATH" == /* && -f "$LIBOMP_PATH" ]]; then
    LIBOMP_RPATH_FLAGS=(-Wl,-rpath,"$(dirname "$LIBOMP_PATH")")
  else
    echo "WARNING: clang could not locate libomp.so — the compiled binaries may fail to run unless a system libomp is already installed." >&2
  fi
fi
CLANG_FLAGS+=(
  -L"${LLVM_INSTALL}/lib" -lmlir_c_runner_utils -lmlir_runner_utils
  -Wl,-rpath,"${LLVM_INSTALL}/lib"
  "${LIBOMP_RPATH_FLAGS[@]}"
)

# $1 = input LLVM-dialect .mlir, $2 = output binary path
compile_llvm_mlir() {
  local llvm_mlir="$1" out_bin="$2"
  local ll="${llvm_mlir%.mlir}.ll"
  "$SPLYCE_TRANSLATE" "$llvm_mlir" --mlir-to-llvmir -o "$ll"
  clang "${CLANG_FLAGS[@]}" "$ll" -o "$out_bin"
}

BASELINE_BIN="test_benchmark_${KERNEL}_baseline${SUFFIX}"
SPLYCE_BIN="test_benchmark_${KERNEL}_splyce${SUFFIX}"

echo "[compile] Baseline ..."
if [[ "$MODE" == "single" ]]; then
  mlir-opt "$SRC" --sparsifier -o "${KERNEL}_baseline${SUFFIX}.mlir"
else
  "$SPLYCE_OPT" "$SRC" \
    --sparsify-to-scf "${PARALLEL_FLAGS[@]}" \
    --splyce-bufferize-restrict \
    --lower-to-llvm "${PARALLEL_FLAGS[@]}" \
    -o "${KERNEL}_baseline${SUFFIX}.mlir"
fi
compile_llvm_mlir "${KERNEL}_baseline${SUFFIX}.mlir" "$BASELINE_BIN"

echo "[compile] Splyce (vector-width=4, phase-select=001) ..."
"$SPLYCE_OPT" "$SRC" \
  --sparsify-to-scf "${PARALLEL_FLAGS[@]}" \
  --splyce="target-function=${KERNEL} vector-width=4 phase-select=001" \
  --splyce-bufferize-restrict \
  --lower-to-llvm "${PARALLEL_FLAGS[@]}" \
  -o "${KERNEL}_splyce${SUFFIX}.mlir"
compile_llvm_mlir "${KERNEL}_splyce${SUFFIX}.mlir" "$SPLYCE_BIN"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

# $1 = binary path -> prints the execution time (seconds) it wrote to its
# own "benchmark" file.
run_and_read_time() {
  local bin="$1"
  rm -f benchmark
  if [[ "$MODE" == "multicore" ]]; then
    OMP_NUM_THREADS="$CORES" "${RUNNER[@]}" "$bin" >/dev/null
  else
    "$bin" >/dev/null
  fi
  if [[ ! -f benchmark ]]; then
    echo "error: $bin did not produce a 'benchmark' file" >&2
    exit 1
  fi
  local t
  t="$(cat benchmark)"
  rm -f benchmark
  echo "$t"
}

echo "[run] Baseline ..."
BASELINE_TIME="$(run_and_read_time "./${BASELINE_BIN}")"

echo "[run] Splyce ..."
SPLYCE_TIME="$(run_and_read_time "./${SPLYCE_BIN}")"

SPEEDUP="$(awk -v b="$BASELINE_TIME" -v s="$SPLYCE_TIME" 'BEGIN { if (s + 0 == 0) print "NA"; else printf "%.2f", b / s }')"

echo ""
echo "Kernel:       ${KERNEL}"
echo "Mode:         ${MODE}$([[ "$MODE" == "multicore" ]] && echo " (${CORES} core(s))")"
printf "Baseline (s): %s\n" "$BASELINE_TIME"
printf "Splyce (s):   %s\n" "$SPLYCE_TIME"
printf "Speedup (x):  %s\n" "$SPEEDUP"
