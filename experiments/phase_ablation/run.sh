#!/usr/bin/env bash
# run.sh — Run every binary compiled by compile.sh in this directory,
# collecting both wall-clock time and TMA (Top-Down Microarchitecture
# Analysis) counters for each one.
#
# Per binary:
#   - Its own "benchmark" output (rtclock, printed by the kernel itself) is
#     saved as ./benchmark_<name>: one line per iteration.
#   - Its own "tma_raw" output (perf_helper.c, see spgemm.mlir) is saved as
#     ./tma_raw_<name>: one line per iteration (same line numbering as
#     benchmark_<name>), 8 raw hardware-counter values (slots,
#     topdown-retiring/bad-spec/fe-bound/be-bound, instructions, cycles,
#     branch-misses), read via an in-process perf_event_open() group
#     bracketed tightly around that iteration's @spgemm call — scoped to
#     exactly the computation, unlike wrapping the whole binary in
#     `perf stat`, which would also count tensor loading, rtclock calls,
#     printf, and the untimed correctness check.
#
#   Iteration 1 (cold start: first touch of B/C, cold caches) is excluded
#   entirely before any statistic is computed. Among the remaining
#   iterations, exec_time_s is the median wall-clock time (exec_time_min_s/
#   max_s/stdev_s over the same set), and retiring_pct/backend_bound_pct/
#   frontend_bound_pct/bad_speculation_pct/branch_misses/instructions/ipc
#   are read from that *exact same* median iteration's tma_raw line — not a
#   separate median-per-column — so every row describes one real,
#   internally consistent execution rather than a mix of statistics from
#   different iterations. branch_misses is the raw hardware branch-
#   misprediction count, distinct from bad_speculation_pct — TMA's Bad
#   Speculation also includes machine clears unrelated to branch
#   prediction, so it's not the same quantity (see perf_helper.c).
#
# Every run is pinned to a single CPU on a single NUMA node (compute and
# memory both local to that node, via numactl) for deterministic,
# cycle-accurate profiling — set NUMA_NODE to override which node (default
# 0).
#
# Usage:
#   ./run.sh            Run binaries, keep the compiled binaries afterward.
#   ./run.sh --clean    Run binaries, then delete the compile.sh-generated
#                        binaries (test_benchmark_*) afterward.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CLEAN=0
if [[ "${1:-}" == "--clean" ]]; then
  CLEAN=1
fi

# ---------------------------------------------------------------------------
# NUMA/CPU pinning for this machine.
# ---------------------------------------------------------------------------
NUMA_NODE="${NUMA_NODE:-0}"

if command -v numactl >/dev/null 2>&1 && [[ -d "/sys/devices/system/node/node${NUMA_NODE}" ]]; then
  # Pin to the first CPU listed for this node, and bind memory to the same
  # node, so neither compute nor allocations cross socket boundaries.
  PIN_CPU="$(cut -d',' -f1 "/sys/devices/system/node/node${NUMA_NODE}/cpulist" | cut -d'-' -f1)"
  RUNNER=(numactl --physcpubind="$PIN_CPU" --membind="$NUMA_NODE")
else
  echo "WARNING: numactl unavailable or node${NUMA_NODE} not found — falling back to taskset (no NUMA memory binding)." >&2
  PIN_CPU=0
  RUNNER=(taskset -c "$PIN_CPU")
fi

if [[ -f /sys/devices/system/cpu/smt/active ]] && [[ "$(cat /sys/devices/system/cpu/smt/active)" == "1" ]]; then
  echo "WARNING: SMT is active on this system — deterministic profiling assumes it's disabled." >&2
fi

TMA_CSV="./tma_results.csv"
echo "name,retiring_pct,backend_bound_pct,frontend_bound_pct,bad_speculation_pct,branch_misses,exec_time_s,exec_time_min_s,exec_time_max_s,exec_time_stdev_s,instructions,ipc" > "$TMA_CSV"

# Reads benchmark_<name> (one line per iteration), drops line 1 (cold
# start), and among the rest finds the median exec time along with the
# *original line number* it came from — the lower-middle element for an
# even count, so there's always exactly one matched line, never an
# interpolated pair. Prints: median,min,max,stdev,matched_line_number
find_median_line() {
  awk '
    NR == 1 { next }
    { n++; lineno[n] = NR; val[n] = $1 }
    END {
      if (n == 0) { print "NA,NA,NA,NA,NA"; exit }
      for (i = 1; i <= n; i++)
        for (j = i + 1; j <= n; j++)
          if (val[j] < val[i]) {
            t = val[i]; val[i] = val[j]; val[j] = t
            t = lineno[i]; lineno[i] = lineno[j]; lineno[j] = t
          }
      mid = int((n + 1) / 2)
      sum = 0
      for (i = 1; i <= n; i++) sum += val[i]
      mean = sum / n
      ss = 0
      for (i = 1; i <= n; i++) ss += (val[i] - mean) ^ 2
      stdev = sqrt(ss / n)
      printf "%.6f,%.6f,%.6f,%.6f,%d\n", val[mid], val[1], val[n], stdev, lineno[mid]
    }
  '
}

# $1 = tma_raw_<name>, $2 = matched line number. Computes the derived TMA
# metrics from that one line's raw counters only.
tma_line_metrics() {
  awk -v line="$2" '
    NR == line {
      slots = $1; ret = $2; bs = $3; fe = $4; be = $5; instr = $6; cyc = $7
      br_miss = $8
      if (slots > 0 && cyc > 0) {
        printf "%.2f,%.2f,%.2f,%.2f,%d,%d,%.3f\n", \
          ret / slots * 100, be / slots * 100, fe / slots * 100, bs / slots * 100, \
          br_miss, instr, instr / cyc
      } else {
        print "NA,NA,NA,NA,NA,NA,NA"
      }
    }
  ' "$1"
}

for bin in ./test_benchmark_*; do
  [[ -x "$bin" ]] || continue
  name="${bin#./test_benchmark_}"
  echo "  -> running $name ..."

  "${RUNNER[@]}" "$bin" || true

  [[ -f benchmark ]] && mv benchmark "benchmark_${name}"
  [[ -f tma_raw ]] && mv tma_raw "tma_raw_${name}"

  if [[ -f "benchmark_${name}" ]]; then
    IFS=, read -r exec_time exec_min exec_max exec_stdev matched_line <<< "$(find_median_line < "benchmark_${name}")"
  else
    exec_time="NA"; exec_min="NA"; exec_max="NA"; exec_stdev="NA"; matched_line="NA"
  fi

  if [[ "$matched_line" != "NA" && -f "tma_raw_${name}" ]]; then
    IFS=, read -r retiring be fe badspec brmiss instr ipc <<< "$(tma_line_metrics "tma_raw_${name}" "$matched_line")"
  else
    retiring="NA"; be="NA"; fe="NA"; badspec="NA"; brmiss="NA"; instr="NA"; ipc="NA"
  fi

  echo "${name},${retiring},${be},${fe},${badspec},${brmiss},${exec_time},${exec_min},${exec_max},${exec_stdev},${instr},${ipc}" >> "$TMA_CSV"
done

if [[ $CLEAN -eq 1 ]]; then
  echo "Cleaning up compiled binaries ..."
  rm -f ./test_benchmark_*
fi
