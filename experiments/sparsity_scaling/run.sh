#!/usr/bin/env bash
# run.sh — Sparsity-scaling study for the two binaries compiled by
# compile.sh in this directory (spgemm's single-threaded baseline and
# Splyce phase-001 binary — see compile.sh).
#
# For each density in SPARSITY_LEVELS (percent of nonzero entries: 0.01,
# 0.1, 1, 2, 5, 10), this regenerates tensor_B.tns/tensor_C.tns at that
# density (gen_data.py sparsity_scaling <pct> — same 5000x5000 shape as
# speedups/synthetic_data/spgemm, just with density swept instead of held
# fixed at 5%) and re-runs both binaries against it, since the compiled
# binaries don't embed any data themselves — only the sparsification
# strategy — so runtime must be re-measured from scratch at each density.
#
# Each run is pinned to a single CPU on a single NUMA node (compute and
# memory both local, via numactl — see phase_ablation/run.sh for the
# details this generalizes). Each run's median execution time (across 6
# iterations, excluding the first/cold-start one) is appended to
# results.csv.
#
# Once every sparsity level is done and results.csv is fully written, this
# also prints the results table (print_results.py) and regenerates
# sparsity_scaling_plot.png (plot_results.py — which also prints the
# baseline/Splyce time at each sparsity level the plot is drawn from, since
# the image itself isn't visible from a terminal).
#
# Usage:
#   ./run.sh [--clean] [results_csv]
#     --clean       Delete the compile.sh-generated binaries afterward.
#     results_csv   Path to write "kernel,configuration,sparsity_pct,
#                    exec_time_s" rows to. Defaults to ./results.csv if
#                    omitted.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

EXPERIMENTS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

KERNEL_NAME="spgemm"
BASELINE_BIN="./test_benchmark_${KERNEL_NAME}_scf"
SPLYCE_BIN="./test_benchmark_${KERNEL_NAME}_splyce_phase_001"
SPARSITY_LEVELS=(0.01 0.1 1 2 5 10)

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

echo "kernel,configuration,sparsity_pct,exec_time_s" > "$RESULTS_CSV"

# $1 = binary path, $2 = config label (for the CSV), $3 = sparsity pct (for the CSV)
run_one() {
  local bin="$1" label="$2" sparsity_pct="$3"

  echo "  -> running ${label} at ${sparsity_pct}% density ..."
  "${RUNNER[@]}" "$bin" || true

  local med
  if [[ -f benchmark ]]; then
    med=$(median_excl_first < benchmark)
    rm -f benchmark
  else
    med="NA"
  fi

  echo "     median (excl. first): ${med}"
  echo "${KERNEL_NAME},${label},${sparsity_pct},${med}" >> "$RESULTS_CSV"
}

for sparsity_pct in "${SPARSITY_LEVELS[@]}"; do
  echo "=== Sparsity ${sparsity_pct}% (nonzero density) ==="
  echo "  -> regenerating data at ${sparsity_pct}% density ..."
  ( cd "$EXPERIMENTS_DIR" && python3 ./gen_data.py sparsity_scaling "$sparsity_pct" )

  run_one "$BASELINE_BIN" "scf" "$sparsity_pct"
  run_one "$SPLYCE_BIN" "splyce_phase_001" "$sparsity_pct"
done

rm -f ./tensor_B.tns ./tensor_C.tns

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
