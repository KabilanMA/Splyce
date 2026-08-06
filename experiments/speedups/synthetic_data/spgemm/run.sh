#!/usr/bin/env bash
# run.sh — Run every binary compiled by compile.sh in this directory (the
# plain mlir-opt --sparsifier baseline, plus one per Splyce phase-select/
# fastmath configuration), and for each one append its median execution
# time across its 6 iterations (excluding the first, cold-start iteration)
# to a shared results file — one row per configuration, so every runtime is
# reported, not just the fastest.
#
# Usage:
#   ./run.sh [--clean] [results_csv]
#     --clean       Delete the compile.sh-generated binaries afterward.
#     results_csv   Path to append "<kernel>,<config>,<time_s>" rows to.
#                    Defaults to ../speedup_results.csv (relative to this
#                    directory) if omitted.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

KERNEL_NAME="spgemm"

CLEAN=0
RESULTS_CSV="../speedup_results.csv"
for arg in "$@"; do
  if [[ "$arg" == "--clean" ]]; then
    CLEAN=1
  else
    RESULTS_CSV="$arg"
  fi
done

# ---------------------------------------------------------------------------
# NUMA/CPU pinning for this machine (see phase_ablation/run.sh for details).
# ---------------------------------------------------------------------------
NUMA_NODE="${NUMA_NODE:-0}"

if command -v numactl >/dev/null 2>&1 && [[ -d "/sys/devices/system/node/node${NUMA_NODE}" ]]; then
  PIN_CPU="$(cut -d',' -f1 "/sys/devices/system/node/node${NUMA_NODE}/cpulist" | cut -d'-' -f1)"
  RUNNER=(numactl --physcpubind="$PIN_CPU" --membind="$NUMA_NODE")
else
  echo "WARNING: numactl unavailable or node${NUMA_NODE} not found — falling back to taskset (no NUMA memory binding)." >&2
  PIN_CPU=0
  RUNNER=(taskset -c "$PIN_CPU")
fi

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

for bin in ./test_benchmark_*; do
  [[ -x "$bin" ]] || continue
  name="${bin#./test_benchmark_${KERNEL_NAME}_}"
  echo "  -> running $name ..."

  "${RUNNER[@]}" "$bin" || true

  if [[ -f benchmark ]]; then
    med=$(median_excl_first < benchmark)
    rm -f benchmark
  else
    med="NA"
  fi

  echo "     median (excl. first): ${med}"
  echo "${KERNEL_NAME},${name},${med}" >> "$RESULTS_CSV"
done

if [[ $CLEAN -eq 1 ]]; then
  echo "Cleaning up compiled binaries ..."
  rm -f ./test_benchmark_*
fi
