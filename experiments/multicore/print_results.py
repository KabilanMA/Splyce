#!/usr/bin/env python3
# print_results.py — Print multicore/run.sh's results.csv (columns:
# kernel,configuration,cores,exec_time_s) as an aligned table, with an
# added speedup column: the scf_parallel baseline (always the single-core
# row) divided by each row's exec_time_s.
#
# Usage:
#   ./print_results.py               # reads ./results.csv
#   ./print_results.py path/to.csv

import csv
import sys

BASELINE_CONFIG = "scf_parallel"


def print_table(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))

    if not rows:
        print(f"{path} is empty")
        return

    header, data_rows = rows[0], rows[1:]
    config_idx = header.index("configuration")
    cores_idx = header.index("cores")
    exec_time_idx = header.index("exec_time_s")

    # Baseline row first, then the Splyce rows in ascending core-count order.
    data_rows.sort(key=lambda row: (row[config_idx] != BASELINE_CONFIG, int(row[cores_idx])))

    baseline_time = None
    for row in data_rows:
        if row[config_idx] == BASELINE_CONFIG:
            try:
                baseline_time = float(row[exec_time_idx])
            except ValueError:
                pass
            break

    header = header + ["speedup"]
    for row in data_rows:
        speedup = "NA"
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
