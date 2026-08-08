#!/usr/bin/env python3
# print_results.py — Print sparsity_scaling/run.sh's results.csv (columns:
# kernel,configuration,sparsity_pct,exec_time_s) as an aligned table, with
# an added speedup column: at each sparsity level, the scf baseline's
# exec_time_s divided by that same level's splyce_phase_001 exec_time_s
# (1.00x on the baseline row itself).
#
# Usage:
#   ./print_results.py               # reads ./results.csv
#   ./print_results.py path/to.csv

import csv
import sys

BASELINE_CONFIG = "scf"


def print_table(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))

    if not rows:
        print(f"{path} is empty")
        return

    header, data_rows = rows[0], rows[1:]
    config_idx = header.index("configuration")
    sparsity_idx = header.index("sparsity_pct")
    exec_time_idx = header.index("exec_time_s")

    # Ascending sparsity level, baseline row before Splyce within each level.
    data_rows.sort(key=lambda row: (float(row[sparsity_idx]), row[config_idx] != BASELINE_CONFIG))

    baseline_by_sparsity = {}
    for row in data_rows:
        if row[config_idx] == BASELINE_CONFIG:
            try:
                baseline_by_sparsity[row[sparsity_idx]] = float(row[exec_time_idx])
            except ValueError:
                pass

    header = header + ["speedup"]
    for row in data_rows:
        speedup = "NA"
        baseline_time = baseline_by_sparsity.get(row[sparsity_idx])
        if baseline_time is not None:
            try:
                speedup = f"{baseline_time / float(row[exec_time_idx]):.2f}x"
            except (ValueError, ZeroDivisionError):
                pass
        row.append(speedup)

    all_rows = [header] + data_rows
    widths = [max(len(row[i]) for row in all_rows) for i in range(len(header))]

    def print_row(row):
        print("  ".join(cell.ljust(w) for cell, w in zip(row, widths)))

    print_row(header)
    print("  ".join("-" * w for w in widths))
    for row in data_rows:
        print_row(row)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "./results.csv"
    print_table(path)


if __name__ == "__main__":
    main()
