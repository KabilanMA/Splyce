#!/usr/bin/env python3
# plot_results.py — Plot Splyce speedup on real-world SpGEMM datasets.
#
# Reads spgemm_realworld_results.csv (produced by
# run_suitesparse_benchmark.py) and draws one horizontal bar per dataset,
# showing Splyce's speedup over the scalar baseline (scf_median_s /
# splyce_median_s). Datasets where no baseline time exists (Splyce timed
# out, or the baseline crashed) are excluded from the chart and reported
# on stdout instead, since a ratio needs both numbers.
#
# Usage:
#   ./plot_results.py                     # reads ./spgemm_realworld_results.csv
#   ./plot_results.py --csv path/to.csv --out path/to.png

import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Palette: dataviz skill's validated default (references/palette.md).
# This is a baseline-comparison bar ("above/below a baseline" -> diverging
# color job), not series identity, so it uses the blue/red diverging pair
# rather than the categorical ramp.
COLOR_SURFACE = "#fcfcfb"
COLOR_INK_PRIMARY = "#0b0b0b"
COLOR_INK_SECONDARY = "#52514e"
COLOR_INK_MUTED = "#898781"
COLOR_GRIDLINE = "#e1e0d9"
COLOR_BASELINE_AXIS = "#c3c2b7"
COLOR_FASTER = "#2a78d6"   # diverging pole: speedup >= 1x
COLOR_SLOWER = "#e34948"   # diverging pole: speedup < 1x

MAX_LABELED_BARS = 40  # above this, per-bar tip labels would just be noise


def load_rows(csv_path):
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main():
    args = sys.argv[1:]
    csv_path = os.path.join(SCRIPT_DIR, "spgemm_realworld_results.csv")
    out_path = os.path.join(SCRIPT_DIR, "spgemm_realworld_speedup.png")
    if "--csv" in args:
        csv_path = args[args.index("--csv") + 1]
    if "--out" in args:
        out_path = args[args.index("--out") + 1]

    if not os.path.isfile(csv_path):
        sys.exit(f"error: {csv_path} not found — run run_suitesparse_benchmark.py first")

    rows = load_rows(csv_path)
    if not rows:
        sys.exit(f"error: {csv_path} has no data rows")

    datasets, speedups, formats = [], [], []
    excluded = []
    for row in rows:
        scf = to_float(row["scf_median_s"])
        splyce = to_float(row["splyce_median_s"])
        if scf is None or splyce is None or splyce == 0:
            excluded.append((row["dataset"], row["scf_median_s"], row["splyce_median_s"]))
            continue
        datasets.append(row["dataset"])
        speedups.append(scf / splyce)
        formats.append(row.get("format", ""))

    if not datasets:
        sys.exit("error: no rows have both a baseline and a Splyce time to compare")

    # Largest speedup at the top of the chart.
    order = np.argsort(speedups)
    datasets = [datasets[i] for i in order]
    speedups = [speedups[i] for i in order]
    formats = [formats[i] for i in order]

    labels = [
        f"{name} [{fmt}]" if fmt and fmt != "dense" else name
        for name, fmt in zip(datasets, formats)
    ]
    colors = [COLOR_FASTER if s >= 1.0 else COLOR_SLOWER for s in speedups]
    geomean = float(np.exp(np.mean(np.log(speedups))))

    n = len(datasets)
    fig_height = max(4.0, 0.32 * n + 1.6)
    fig, ax = plt.subplots(figsize=(10, fig_height), dpi=150, facecolor=COLOR_SURFACE,
                            layout="constrained")
    ax.set_facecolor(COLOR_SURFACE)

    y = np.arange(n)
    ax.barh(y, speedups, height=0.6, color=colors, zorder=3)
    ax.axvline(1.0, color=COLOR_BASELINE_AXIS, linewidth=1, linestyle="--", zorder=2)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=COLOR_INK_MUTED, fontsize=8)
    ax.set_xlabel("Speedup vs. scalar baseline (scf_median / splyce_median), ×",
                  color=COLOR_INK_SECONDARY, fontsize=10)

    ax.grid(axis="x", color=COLOR_GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0, colors=COLOR_INK_MUTED, labelsize=8)
    ax.set_ylim(-0.6, n - 0.4)

    # Bars all grow from 0, so the tip is always the bar's right edge —
    # labeling just past it (never inside the fill) keeps text on the
    # surface color, not the data color, regardless of speedup vs slowdown.
    max_speedup = max(speedups)
    offset = max_speedup * 0.015
    if n <= MAX_LABELED_BARS:
        for yi, s in zip(y, speedups):
            ax.text(s + offset, yi, f"{s:.2f}×", va="center", ha="left",
                     color=COLOR_INK_PRIMARY, fontsize=7.5)
    else:
        for idx in (0, -1):
            ax.text(speedups[idx] + offset, y[idx], f"{speedups[idx]:.2f}×",
                     va="center", ha="left", color=COLOR_INK_PRIMARY, fontsize=8)

    fig.suptitle("Splyce Speedup — Real-World SpGEMM (SuiteSparse)",
                 x=0.0, ha="left", color=COLOR_INK_PRIMARY, fontsize=13, fontweight="bold")
    ax.set_title(f"n={n} datasets · geometric mean speedup: {geomean:.2f}×",
                 loc="left", color=COLOR_INK_SECONDARY, fontsize=9.5, pad=10)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_FASTER),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_SLOWER),
    ]
    ax.legend(handles, ["Faster than baseline (≥ 1×)", "Slower than baseline (< 1×)"],
              loc="lower right", frameon=False, fontsize=8, labelcolor=COLOR_INK_SECONDARY)

    fig.savefig(out_path, facecolor=COLOR_SURFACE)
    print(f"Wrote {out_path}  ({n} datasets plotted, geomean speedup {geomean:.2f}×)")

    if excluded:
        print(f"Excluded {len(excluded)} dataset(s) with no baseline-vs-Splyce comparison available:")
        for name, scf, splyce in excluded:
            print(f"  {name}: scf_median_s={scf!r} splyce_median_s={splyce!r}")


if __name__ == "__main__":
    main()
