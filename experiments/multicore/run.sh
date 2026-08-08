#!/usr/bin/env bash
# run.sh — Vectorization-vs-threading study for the binaries compiled by
# compile.sh in this directory (spmttkrp's parallel baseline and parallel+
# Splyce binary — see compile.sh).
#
#   1. Baseline (no Splyce SIMD vectorization) — run once, pinned to a
#      single core. The fixed reference point everything else is measured
#      against.
#   2. Splyce (vectorized) — run once per core count in CORE_COUNTS
#      (1 2 4 8 16 32). The 1-core row isolates pure SIMD vectorization's
#      win over the baseline; 2..32 isolate how much threading adds on
#      top of that.
#
# Each run pins to that many CPUs *within a single NUMA node* (compute and
# memory both local — via numactl --physcpubind/--membind) and sets
# OMP_NUM_THREADS to match, so the OpenMP dense-outer-loop parallelization
# actually uses that many threads instead of over- or under-subscribing.
# Each run's median execution time (across 6 iterations, excluding the
# first/cold-start one) is appended to results.csv.
#
# A node's CPU list (/sys/devices/system/node/nodeN/cpulist) is not
# necessarily a single contiguous range — e.g. a dual-socket machine can
# interleave sockets (node0 = 0,2,4,...; node1 = 1,3,5,...), so this
# expands it properly (ranges and comma-separated singles both) rather
# than assuming "N-M". If a requested core count exceeds how many CPUs
# the node actually has (e.g. a smaller dev machine asked for 32), that
# count is skipped with a warning instead of failing the whole sweep —
# this script is meant to run unmodified on machines of different sizes.
#
# Once every run is done and results.csv is fully written, this also
# prints the results table (print_results.py) and regenerates
# speedup_plot.png (plot_results.py — which also prints the per-core-count
# speedup numbers the plot is drawn from, since the image itself isn't
# visible from a terminal).
#
# Usage:
#   ./run.sh [--clean] [results_csv]
#     --clean       Delete the compile.sh-generated binaries afterward.
#     results_csv   Path to write "kernel,configuration,cores,time_s" rows
#                    to. Defaults to ./results.csv if omitted.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

KERNEL_NAME="spmttkrp"
BASELINE_BIN="./test_benchmark_${KERNEL_NAME}_scf_parallel"
SPLYCE_BIN="./test_benchmark_${KERNEL_NAME}_splyce_phase_001_parallel"
CORE_COUNTS=(1 2 4 8 16 32)

CLEAN=0
RESULTS_CSV="./results.csv"
for arg in "$@"; do
  if [[ "$arg" == "--clean" ]]; then
    CLEAN=1
  else
    RESULTS_CSV="$arg"
  fi
done

if [[ ! -x "$BASELINE_BIN" || ! -x "$SPLYCE_BIN" ]]; then
  echo "error: binaries not found — run ./compile.sh first" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# NUMA/CPU pinning for this machine (see phase_ablation/run.sh for the
# single-CPU version this generalizes).
# ---------------------------------------------------------------------------
NUMA_NODE="${NUMA_NODE:-0}"

if [[ ! -d "/sys/devices/system/node/node${NUMA_NODE}" ]]; then
  echo "error: /sys/devices/system/node/node${NUMA_NODE} not found — set NUMA_NODE to a valid node." >&2
  exit 1
fi

USE_NUMACTL=0
if command -v numactl >/dev/null 2>&1; then
  USE_NUMACTL=1
else
  echo "WARNING: numactl unavailable — falling back to taskset (no NUMA memory binding)." >&2
fi

# Expands a Linux cpulist ("0,2,4,6-10,12") into one CPU id per line.
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
echo "NUMA node ${NUMA_NODE} has ${#NODE_CPUS[@]} CPUs available: ${NODE_CPUS[*]}"

# Reads a benchmark file (one exec time per line), drops line 1 (cold
# start), and prints the median of the rest.
median_excl_first() {
  awk '
    NR == 1 { next }
    { a[++n] = $1 }
    END {
      if (n == 0) { print "NA"; exit }
      for (i = 1; i <= n; i++)
        for (j = i + 1; j <= n; j++)
          if (a[j] < a[i]) { t = a[i]; a[i] = a[j]; a[j] = t }
      if (n % 2 == 1) print a[(n + 1) / 2]
      else            print (a[n / 2] + a[n / 2 + 1]) / 2
    }
  '
}

echo "kernel,configuration,cores,exec_time_s" > "$RESULTS_CSV"

# $1 = binary path, $2 = config label (for the CSV), $3 = core count
run_one() {
  local bin="$1" label="$2" cores="$3"

  if (( cores > ${#NODE_CPUS[@]} )); then
    echo "  [skip] ${label} at ${cores} core(s): node${NUMA_NODE} only has ${#NODE_CPUS[@]} CPUs."
    return
  fi

  local pin_cpus
  pin_cpus="$(IFS=,; echo "${NODE_CPUS[*]:0:$cores}")"
  local runner
  if (( USE_NUMACTL )); then
    runner=(numactl --physcpubind="$pin_cpus" --membind="$NUMA_NODE")
  else
    runner=(taskset -c "$pin_cpus")
  fi

  echo "  -> running ${label} on ${cores} core(s) (CPUs ${pin_cpus}) ..."
  OMP_NUM_THREADS="$cores" "${runner[@]}" "$bin" || true

  local med
  if [[ -f benchmark ]]; then
    med=$(median_excl_first < benchmark)
    rm -f benchmark
  else
    med="NA"
  fi

  echo "     median (excl. first): ${med}"
  echo "${KERNEL_NAME},${label},${cores},${med}" >> "$RESULTS_CSV"
}

echo "=== Baseline (no Splyce vectorization), single core ==="
run_one "$BASELINE_BIN" "scf_parallel" 1

echo "=== Splyce (vectorized), scaling across cores ==="
for cores in "${CORE_COUNTS[@]}"; do
  run_one "$SPLYCE_BIN" "splyce_phase_001_parallel" "$cores"
done

echo ""
echo "Results (${RESULTS_CSV}):"
python3 ./print_results.py "$RESULTS_CSV"

echo ""
echo "Generating plot ..."
./plot_results.py --csv "$RESULTS_CSV"

if [[ $CLEAN -eq 1 ]]; then
  echo "Cleaning up compiled binaries ..."
  rm -f ./test_benchmark_*
fi
