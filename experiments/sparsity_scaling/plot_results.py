#!/usr/bin/env python3
# plot_results.py — Plot sparsity-scaling results from run.sh's results.csv
# (columns: kernel,configuration,sparsity_pct,exec_time_s).
#
# Draws two execution-time lines against input matrix sparsity (log-scaled
# x-axis):
#   - baseline (blue, filled circles): the scf configuration (no Splyce).
#   - Splyce (red, filled squares): the splyce_phase_001 configuration.
#
# Usage:
#   ./plot_results.py                        # reads ./results.csv
#   ./plot_results.py --csv path/to.csv --out path/to.png

import csv
import os
import sys

import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Palette: same diverging blue/red pair used by multicore/plot_results.py
# and speedups/real_world_data/spgemm/plot_results.py (dataviz skill default).
COLOR_SURFACE = "#fcfcfb"
COLOR_INK_PRIMARY = "#0b0b0b"
COLOR_INK_SECONDARY = "#52514e"
COLOR_INK_MUTED = "#898781"
COLOR_GRIDLINE = "#e1e0d9"
COLOR_BASELINE_LINE = "#2a78d6"  # blue — scf baseline
COLOR_SPLYCE_LINE = "#e34948"    # red — Splyce


def load_rows(csv_path):
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    args = sys.argv[1:]
    csv_path = os.path.join(SCRIPT_DIR, "results.csv")
    out_path = os.path.join(SCRIPT_DIR, "sparsity_scaling_plot.png")
    if "--csv" in args:
        csv_path = args[args.index("--csv") + 1]
    if "--out" in args:
        out_path = args[args.index("--out") + 1]

    if not os.path.isfile(csv_path):
        sys.exit(f"error: {csv_path} not found — run ./run.sh first")

    rows = load_rows(csv_path)
    if not rows:
        sys.exit(f"error: {csv_path} has no data rows")

    baseline_by_sparsity = {}
    splyce_by_sparsity = {}
    for row in rows:
        sparsity_pct = float(row["sparsity_pct"])
        exec_time = float(row["exec_time_s"])
        config = row["configuration"]
        if "splyce" in config:
            splyce_by_sparsity[sparsity_pct] = exec_time
        elif "scf" in config:
            baseline_by_sparsity[sparsity_pct] = exec_time

    if not baseline_by_sparsity:
        sys.exit(f"error: {csv_path} has no scf (baseline) rows")
    if not splyce_by_sparsity:
        sys.exit(f"error: {csv_path} has no Splyce rows")

    sparsity_levels = sorted(set(baseline_by_sparsity) & set(splyce_by_sparsity))
    baseline_times = [baseline_by_sparsity[s] for s in sparsity_levels]
    splyce_times = [splyce_by_sparsity[s] for s in sparsity_levels]

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150, facecolor=COLOR_SURFACE, layout="constrained")
    ax.set_facecolor(COLOR_SURFACE)

    ax.plot(sparsity_levels, baseline_times, color=COLOR_BASELINE_LINE, linewidth=1.75,
            marker="o", markersize=6, markerfacecolor=COLOR_BASELINE_LINE,
            markeredgecolor=COLOR_BASELINE_LINE, label="Baseline", zorder=3)
    ax.plot(sparsity_levels, splyce_times, color=COLOR_SPLYCE_LINE, linewidth=1.75,
            marker="s", markersize=6, markerfacecolor=COLOR_SPLYCE_LINE,
            markeredgecolor=COLOR_SPLYCE_LINE, label="Splyce", zorder=3)

    ax.set_xscale("log")
    ax.set_xticks(sparsity_levels)
    ax.set_xticklabels([f"{s:g}" for s in sparsity_levels])
    ax.minorticks_off()

    ax.set_xlabel("Input Matrix Sparsity (%)", color=COLOR_INK_SECONDARY, fontsize=10)
    ax.set_ylabel("Execution Time (s)", color=COLOR_INK_SECONDARY, fontsize=10)

    ax.grid(True, color=COLOR_GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0, colors=COLOR_INK_MUTED, labelsize=9)

    fig.suptitle("Sparsity Scaling — spgemm", x=0.0, ha="left",
                 color=COLOR_INK_PRIMARY, fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=COLOR_INK_SECONDARY)

    fig.savefig(out_path, facecolor=COLOR_SURFACE)
    print(f"Wrote {out_path}")
    for s, b, sp in zip(sparsity_levels, baseline_times, splyce_times):
        print(f"  sparsity={s:g}%  baseline={b:.3f}s  splyce={sp:.3f}s")


if __name__ == "__main__":
    main()
