#!/usr/bin/env python3
# print_realworld_summary.py — Read every kernel's <kernel>_realworld_results.csv
# (this directory's spgemm/spmmh/spmspv/spmttkrp/spttspm subdirectories, each
# written by that kernel's own run_suitesparse_benchmark.py — every one of
# them has at least "dataset", "scf_median_s", "splyce_median_s" columns,
# alongside kernel-specific extras) and print one combined CSV summarizing
# all of them:
#   Kernel Name,Dataset,Baseline (s),Splyce (s),Speedup (x)
#
# Baseline is scf_median_s, Splyce is splyce_median_s (each the median of
# the 5 non-cold-start iterations — see each kernel's run_suitesparse_
# benchmark.py). Speedup is baseline / splyce. spgemm has multiple curated
# jobs (multiple rows/datasets); every other kernel has exactly one. A
# baseline of "SKIPPED" (Splyce timed out) or "SKIPPED_MEMORY" (job skipped
# outright, spmmh only) is passed through as-is with speedup "NA".
#
# Usage:
#   ./print_realworld_summary.py                  # reads ./<kernel>/<kernel>_realworld_results.csv,
#                                                   # writes ./realworld_summary.csv
#   ./print_realworld_summary.py --out summary.csv # writes to a custom path instead

import csv
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KERNELS = ["spgemm", "spmmh", "spmspv", "spmttkrp", "spttspm"]


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main():
    args = sys.argv[1:]
    out_path = os.path.join(SCRIPT_DIR, "realworld_summary.csv")
    if "--out" in args:
        out_path = args[args.index("--out") + 1]

    header = ["Kernel Name", "Dataset", "Baseline (s)", "Splyce (s)", "Speedup (x)"]
    out_rows = [header]
    missing = []

    for kernel in KERNELS:
        csv_path = os.path.join(SCRIPT_DIR, kernel, f"{kernel}_realworld_results.csv")
        if not os.path.isfile(csv_path):
            missing.append(f"{kernel}: {csv_path} not found — run its compile.sh/run_suitesparse_benchmark.py first")
            out_rows.append([kernel, "NA", "NA", "NA", "NA"])
            continue

        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))

        if not rows:
            missing.append(f"{kernel}: {csv_path} has no rows yet")
            out_rows.append([kernel, "NA", "NA", "NA", "NA"])
            continue

        for row in rows:
            dataset = row["dataset"]
            baseline_raw = row["scf_median_s"]
            splyce_raw = row["splyce_median_s"]
            baseline = to_float(baseline_raw)
            splyce = to_float(splyce_raw)

            if splyce is None:
                missing.append(f"{kernel}/{dataset}: missing/invalid splyce_median_s in {csv_path}")
                out_rows.append([kernel, dataset, baseline_raw, splyce_raw, "NA"])
                continue

            if baseline is None:
                # e.g. "SKIPPED" (Splyce timed out) or "SKIPPED_MEMORY".
                out_rows.append([kernel, dataset, baseline_raw, f"{splyce:.6f}", "NA"])
                continue

            speedup = baseline / splyce if splyce != 0 else None
            out_rows.append([
                kernel, dataset,
                f"{baseline:.6f}", f"{splyce:.6f}",
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
