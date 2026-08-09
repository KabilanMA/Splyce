#!/usr/bin/env python3
# print_speedup_summary.py — Read every kernel's results.csv (this
# directory's spgemm/spmmh/spmspv/spmttkrp/spttspm subdirectories, each
# written by that kernel's own run.sh — columns: kernel,configuration,
# exec_time_s) and print one combined CSV summarizing all of them:
#   Kernel Name,Baseline (s),Splyce (s),Speedup (x)
#
# Baseline is the "scf" configuration row; Splyce is the "splyce_phase_001"
# row (see each kernel's compile.sh — every kernel here compiles just
# those two binaries). Speedup is baseline / splyce.
#
# Usage:
#   ./print_speedup_summary.py                  # reads ./<kernel>/results.csv,
#                                                 # writes ./speedup_summary.csv
#   ./print_speedup_summary.py --out summary.csv # writes to a custom path instead

import csv
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KERNELS = ["spgemm", "spmmh", "spmspv", "spmttkrp", "spttspm"]
BASELINE_CONFIG = "scf"
SPLYCE_CONFIG = "splyce_phase_001"


def load_times(csv_path):
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {row["configuration"]: row["exec_time_s"] for row in rows}


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main():
    args = sys.argv[1:]
    out_path = os.path.join(SCRIPT_DIR, "speedup_summary.csv")
    if "--out" in args:
        out_path = args[args.index("--out") + 1]

    header = ["Kernel Name", "Baseline (s)", "Splyce (s)", "Speedup (x)"]
    out_rows = [header]
    missing = []

    for kernel in KERNELS:
        csv_path = os.path.join(SCRIPT_DIR, kernel, "results.csv")
        if not os.path.isfile(csv_path):
            missing.append(f"{kernel}: {csv_path} not found — run its compile.sh/run.sh first")
            out_rows.append([kernel, "NA", "NA", "NA"])
            continue

        times = load_times(csv_path)
        baseline = to_float(times.get(BASELINE_CONFIG))
        splyce = to_float(times.get(SPLYCE_CONFIG))

        if baseline is None or splyce is None:
            missing.append(f"{kernel}: missing '{BASELINE_CONFIG}' or '{SPLYCE_CONFIG}' row in {csv_path}")
            out_rows.append([
                kernel,
                f"{baseline:.6f}" if baseline is not None else "NA",
                f"{splyce:.6f}" if splyce is not None else "NA",
                "NA",
            ])
            continue

        speedup = baseline / splyce if splyce != 0 else None
        out_rows.append([
            kernel,
            f"{baseline:.6f}",
            f"{splyce:.6f}",
            f"{speedup:.2f}" if speedup is not None else "NA",
        ])

    writer = csv.writer(sys.stdout)
    writer.writerows(out_rows)

    with open(out_path, "w", newline="") as f:
        csv.writer(f).writerows(out_rows)
    print(f"Wrote {out_path}", file=sys.stderr)

    if missing:
        print("Incomplete data for:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)


if __name__ == "__main__":
    main()
