#!/usr/bin/env python3
# plot_results.py — Plot core-count scaling from multicore/run.sh's
# results.csv (columns: kernel,configuration,cores,exec_time_s).
#
# Draws two speedup lines against core count (log2-scaled x-axis):
#   - "vs. scalar baseline" (blue, squares): the single-core scf_parallel
#     (no Splyce) time divided by the Splyce time at each core count. This
#     is the end-to-end win — vectorization and threading combined.
#   - "vs. Splyce @ 1 core" (red, filled circles): the single-core Splyce
#     time divided by the Splyce time at each core count. This isolates
#     pure threading scaling, since both numerator and denominator already
#     include Splyce's SIMD vectorization.
#
# X-axis (core count) is a normal linear scale.
#
# Usage:
#   ./plot_results.py                        # reads ./results.csv
#   ./plot_results.py --csv path/to.csv --out path/to.png

import csv
import os
import sys

import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Palette: same diverging blue/red pair used by
# speedups/real_world_data/spgemm/plot_results.py (dataviz skill default).
COLOR_SURFACE = "#fcfcfb"
COLOR_INK_PRIMARY = "#0b0b0b"
COLOR_INK_SECONDARY = "#52514e"
COLOR_INK_MUTED = "#898781"
COLOR_GRIDLINE = "#e1e0d9"
COLOR_BASELINE_LINE = "#2a78d6"   # blue — speedup vs. scalar baseline
COLOR_THREADING_LINE = "#e34948"  # red — speedup vs. Splyce @ 1 core


def load_rows(csv_path):
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    args = sys.argv[1:]
    csv_path = os.path.join(SCRIPT_DIR, "results.csv")
    out_path = os.path.join(SCRIPT_DIR, "speedup_plot.png")
    if "--csv" in args:
        csv_path = args[args.index("--csv") + 1]
    if "--out" in args:
        out_path = args[args.index("--out") + 1]

    if not os.path.isfile(csv_path):
        sys.exit(f"error: {csv_path} not found — run ./run.sh first")

    rows = load_rows(csv_path)
    if not rows:
        sys.exit(f"error: {csv_path} has no data rows")

    baseline_time = None
    splyce_by_cores = {}
    for row in rows:
        cores = int(row["cores"])
        exec_time = float(row["exec_time_s"])
        config = row["configuration"]
        if "splyce" in config:
            splyce_by_cores[cores] = exec_time
        elif "scf" in config:
            if cores != 1:
                sys.exit(f"error: expected the scf baseline at 1 core, found it at {cores}")
            baseline_time = exec_time

    if baseline_time is None:
        sys.exit(f"error: {csv_path} has no scf (baseline) row")
    if 1 not in splyce_by_cores:
        sys.exit(f"error: {csv_path} has no Splyce row at 1 core")

    cores = sorted(splyce_by_cores)
    splyce_core1_time = splyce_by_cores[1]

    speedup_vs_baseline = [baseline_time / splyce_by_cores[c] for c in cores]
    speedup_vs_splyce_1core = [splyce_core1_time / splyce_by_cores[c] for c in cores]

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150, facecolor=COLOR_SURFACE, layout="constrained")
    ax.set_facecolor(COLOR_SURFACE)

    ax.plot(cores, speedup_vs_baseline, color=COLOR_BASELINE_LINE, linewidth=1.75,
            marker="s", markersize=6, markerfacecolor=COLOR_BASELINE_LINE,
            markeredgecolor=COLOR_BASELINE_LINE, label="Splyce Parallel vs. scalar baseline", zorder=3)
    ax.plot(cores, speedup_vs_splyce_1core, color=COLOR_THREADING_LINE, linewidth=1.75,
            marker="o", markersize=6, markerfacecolor=COLOR_THREADING_LINE,
            markeredgecolor=COLOR_THREADING_LINE, label="Splyce Parallel vs. Splyce @ 1 core", zorder=3)

    ax.set_xticks(cores)
    ax.set_xticklabels([str(c) for c in cores])

    ax.set_xlabel("Core count", color=COLOR_INK_SECONDARY, fontsize=10)
    ax.set_ylabel("Speedup (×)", color=COLOR_INK_SECONDARY, fontsize=10)

    ax.grid(True, color=COLOR_GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0, colors=COLOR_INK_MUTED, labelsize=9)

    fig.suptitle("Multicore Scaling — spmttkrp", x=0.0, ha="left",
                 color=COLOR_INK_PRIMARY, fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=COLOR_INK_SECONDARY)

    fig.savefig(out_path, facecolor=COLOR_SURFACE)
    print(f"Wrote {out_path}")
    for c, sb, ss in zip(cores, speedup_vs_baseline, speedup_vs_splyce_1core):
        print(f"  cores={c:>2}  vs_baseline={sb:.2f}x  vs_splyce_1core={ss:.2f}x")


if __name__ == "__main__":
    main()
