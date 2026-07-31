#!/usr/bin/env bash
# run.sh — Run every binary compiled by compile.sh in this directory,
# collecting both wall-clock time and TMA (Top-Down Microarchitecture
# Analysis) counters for each one.
#
# Per binary:
#   - Its own "benchmark" output (rtclock, printed by the kernel itself) is
#     saved as ./benchmark_<name>. This is also the source of exec_time_s,
#     exec_time_min_s, exec_time_max_s, and exec_time_stdev_s in
#     tma_results.csv (median/min/max/population-stdev across all
#     iterations in that file), since it times only the @spgemm call
#     itself — perf's own wall-clock would instead cover the whole process
#     (tensor file loading, an untimed correctness-check call, etc.),
#     which isn't what we want to report as "execution time".
#   - `perf stat` wraps the run to collect the four top-level TMA
#     categories (retiring, backend-bound, frontend-bound, bad
#     speculation), total instructions, and IPC. All 17 binaries' results
#     are appended as rows to ./tma_results.csv.
#
# Every run is pinned to a single CPU on a single NUMA node (compute and
# memory both local to that node, via numactl) for deterministic,
# cycle-accurate profiling — set NUMA_NODE to override which node (default
# 0). On hybrid Intel client CPUs (P-core/E-core), perf's hybrid PMU
# handling also makes --topdown ambiguous (it reports both PMUs' counters,
# one of them "not counted"), so on those machines this additionally reads
# the pinned core's PMU (cpu_core) raw topdown-*/slots counters directly
# instead of the --topdown/-M convenience wrappers; server Xeons/EPYCs have
# no such split and use plain event names.
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
# NUMA/CPU pinning + perf event list for this machine.
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

# cpu_core/cpu_atom PMU split only exists on hybrid client CPUs (e.g. Alder
# Lake+); server Xeons/EPYCs expose one uniform PMU with plain event names.
if [[ -f /sys/devices/cpu_core/cpus ]]; then
  EVENTS="cpu_core/slots/,cpu_core/topdown-retiring/,cpu_core/topdown-bad-spec/,cpu_core/topdown-fe-bound/,cpu_core/topdown-be-bound/,instructions,cycles"
else
  EVENTS="slots,topdown-retiring,topdown-bad-spec,topdown-fe-bound,topdown-be-bound,instructions,cycles"
fi

TMA_CSV="./tma_results.csv"
echo "name,retiring_pct,backend_bound_pct,frontend_bound_pct,bad_speculation_pct,exec_time_s,exec_time_min_s,exec_time_max_s,exec_time_stdev_s,instructions,ipc" > "$TMA_CSV"

# $1 = perf's raw `-x,` CSV stderr output for one run
# Prints: retiring_pct,backend_bound_pct,frontend_bound_pct,bad_speculation_pct,instructions,ipc
parse_tma() {
  awk -F, '
    $3 ~ /slots/                    { slots = $1 }
    $3 ~ /topdown-retiring/         { retiring = $1 }
    $3 ~ /topdown-bad-spec/         { badspec = $1 }
    $3 ~ /topdown-fe-bound/         { fe = $1 }
    $3 ~ /topdown-be-bound/         { be = $1 }
    $3 ~ /instructions/ && $1 != "<not counted>" { instr = $1 }
    $3 ~ /cycles/       && $1 != "<not counted>" { cyc = $1 }
    END {
      if (slots > 0 && cyc > 0) {
        printf "%.2f,%.2f,%.2f,%.2f,%d,%.3f\n", \
          retiring/slots*100, be/slots*100, fe/slots*100, badspec/slots*100, instr, instr/cyc
      } else {
        print "NA,NA,NA,NA,NA,NA"
      }
    }
  '
}

for bin in ./test_benchmark_*; do
  [[ -x "$bin" ]] || continue
  name="${bin#./test_benchmark_}"
  echo "  -> running $name ..."

  perf_out=$("${RUNNER[@]}" perf stat -x, -e "$EVENTS" -- "$bin" 2>&1 >/dev/null) || true

  [[ -f benchmark ]] && mv benchmark "benchmark_${name}"

  if [[ -f "benchmark_${name}" ]]; then
    # Median/min/max/stdev across all iterations in the benchmark file —
    # median is robust to outlier iterations, min/max/stdev capture the
    # spread for reviewer-facing error bars.
    IFS=, read -r exec_time exec_min exec_max exec_stdev <<< "$(sort -n "benchmark_${name}" | awk '
      { a[NR] = $1; sum += $1 }
      END {
        n = NR
        if (n == 0) { print "NA,NA,NA,NA"; exit }
        mean = sum / n
        if (n % 2 == 1) median = a[(n + 1) / 2]
        else            median = (a[n / 2] + a[n / 2 + 1]) / 2
        ss = 0
        for (i = 1; i <= n; i++) ss += (a[i] - mean) ^ 2
        stdev = sqrt(ss / n)
        printf "%.6f,%.6f,%.6f,%.6f", median, a[1], a[n], stdev
      }
    ')"
  else
    exec_time="NA"; exec_min="NA"; exec_max="NA"; exec_stdev="NA"
  fi

  IFS=, read -r retiring be fe badspec instr ipc <<< "$(echo "$perf_out" | parse_tma)"
  echo "${name},${retiring},${be},${fe},${badspec},${exec_time},${exec_min},${exec_max},${exec_stdev},${instr},${ipc}" >> "$TMA_CSV"
done

if [[ $CLEAN -eq 1 ]]; then
  echo "Cleaning up compiled binaries ..."
  rm -f ./test_benchmark_*
fi
