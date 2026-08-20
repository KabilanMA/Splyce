#!/usr/bin/env python3
# plot_results.py — Plot Splyce speedup on real-world SpMSpV datasets.
#
# Reads spmspv_realworld_sweep_results.csv (produced by
# run_suitesparse_sweep.py) and draws one vertical column per dataset,
# showing Splyce's speedup over the scalar baseline (scf_median_s /
# splyce_median_s). Datasets where no baseline time exists (Splyce timed
# out, or the baseline crashed) are excluded from the chart and reported
# on stdout instead, since a ratio needs both numbers.
#
# The sweep CSV can hold both single- and multicore-mode rows for the same
# dataset (see run_suitesparse_sweep.py's --mode) — only one config makes
# sense per dataset here, so rows are filtered to --mode (default:
# multicore) before plotting.
#
# Saves as PDF by default — a vector format, with pdf.fonttype=42 below so
# text embeds as real (searchable/editable) glyphs rather than Type 3
# bitmaps, which is what camera-ready/LaTeX pipelines expect.
#
# Usage:
#   ./plot_results.py
#       Reads ./spmspv_realworld_sweep_results.csv and writes both
#       spmspv_realworld_speedup_by_density.pdf (highest matrix nnz
#       density left to lowest right) and
#       spmspv_realworld_speedup_by_speedup.pdf (highest speedup left to
#       lowest right).
#   ./plot_results.py --csv path/to.csv --sort density|speedup --out path/to.pdf
#       Writes just the one ordering, to the given path.
#   ./plot_results.py --mode single       # plot single-threaded rows instead

import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Palette: dataviz skill's validated default (references/palette.md).
# This is a baseline-comparison bar ("above/below a baseline" -> diverging
# color job), not series identity, so it uses the blue/red diverging pair
# rather than the categorical ramp.
COLOR_INK_PRIMARY = "#0b0b0b"
COLOR_INK_SECONDARY = "#52514e"
COLOR_INK_MUTED = "#0b0b0b"
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


def nnz_density(row):
    # matrix_sparsity is already computed in the sweep CSV (unlike
    # spgemm's, which only has shape/nnz and needs it derived) — density
    # is just its complement.
    return 1.0 - float(row["matrix_sparsity"])


def load_data(csv_path, mode):
    rows = load_rows(csv_path)
    if not rows:
        sys.exit(f"error: {csv_path} has no data rows")
    rows = [row for row in rows if row.get("mode", "single") == mode]
    if not rows:
        sys.exit(f"error: {csv_path} has no rows with mode={mode!r}")

    datasets, speedups, densities = [], [], []
    excluded = []
    for row in rows:
        scf = to_float(row["scf_median_s"])
        splyce = to_float(row["splyce_median_s"])
        if scf is None or splyce is None or splyce == 0:
            excluded.append((row["dataset"], row["scf_median_s"], row["splyce_median_s"]))
            continue
        datasets.append(row["dataset"])
        speedups.append(scf / splyce)
        densities.append(nnz_density(row))

    if not datasets:
        sys.exit("error: no rows have both a baseline and a Splyce time to compare")

    return datasets, speedups, densities, excluded


# sort_by: "density" (highest matrix nnz density left to lowest right) or
# "speedup" (highest speedup left to lowest right).
def plot(datasets, speedups, densities, excluded, sort_by, out_path):
    sort_key = densities if sort_by == "density" else speedups
    order = np.argsort(sort_key)[::-1]
    datasets = [datasets[i] for i in order]
    speedups = [speedups[i] for i in order]

    labels = datasets
    colors = [COLOR_FASTER if s >= 1.0 else COLOR_SLOWER for s in speedups]
    geomean = float(np.exp(np.mean(np.log(speedups))))

    n = len(datasets)
    # Tight packing for print: a narrow per-bar allocation plus a bar
    # width close to 1 minimizes both the column width and the gap
    # between columns. Labels go fully vertical (rotation=90) so their
    # footprint is just the font height, not the string length — that's
    # what makes packing this tight possible without adjacent labels
    # colliding (see spgemm/plot_results.py for the same design, arrived
    # at the same way).
    fig_width = max(3.2, 0.155 * n + 0.9)
    fig, ax = plt.subplots(figsize=(fig_width, 7.0), dpi=300,
                            layout="constrained")

    x = np.arange(n)
    ax.bar(x, speedups, width=0.92, color=colors, zorder=3)
    # zorder above the bars' (3), same reasoning as the axis spines below —
    # with almost no gap between columns, a line drawn *under* them would
    # only be visible in the thin slivers between bars.
    ax.axhline(1.0, color=COLOR_BASELINE_AXIS, linewidth=1.2, linestyle="--", zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=COLOR_INK_MUTED, fontsize=6.5,
                        rotation=90, ha="center", va="top")
    ax.set_xlabel("Dataset", color=COLOR_INK_PRIMARY, fontsize=18)
    ax.set_ylabel("Speedup", color=COLOR_INK_PRIMARY, fontsize=18)

    # No grid — the 1x dashed line above is the only horizontal reference.
    # Spine zorder raised above the bars' (3) so the axis line draws on top
    # of them instead of being painted over at the bars' base.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_INK_MUTED)
    ax.spines["bottom"].set_color(COLOR_INK_MUTED)
    ax.spines["left"].set_zorder(6)
    ax.spines["bottom"].set_zorder(10)
    ax.tick_params(axis="y", length=3, colors=COLOR_INK_MUTED, labelsize=11)
    ax.tick_params(axis="x", length=3, colors=COLOR_INK_MUTED, labelsize=8)
    ax.set_xlim(-0.5, n - 0.5)

    # Bars all grow from 0, so the tip is always the bar's top edge —
    # labeling just past it (never inside the fill) keeps text on the
    # surface color, not the data color, regardless of speedup vs slowdown.
    max_speedup = max(speedups)
    # Rotated (90°) tip labels extend upward past the bar itself — headroom
    # here keeps the tallest one clear of the title above the axes.
    ax.set_ylim(0, max_speedup * 1.22)
    offset = max_speedup * 0.02
    if n <= MAX_LABELED_BARS:
        for xi, s in zip(x, speedups):
            ax.text(float(xi), s + offset, f"{s:.2f}×", va="bottom", ha="center",
                     color=COLOR_INK_PRIMARY, fontsize=6.5, rotation=90)
    else:
        for idx in (0, -1):
            ax.text(float(x[idx]), speedups[idx] + offset, f"{speedups[idx]:.2f}×",
                     va="bottom", ha="center", color=COLOR_INK_PRIMARY, fontsize=6.5, rotation=90)

    # ax.set_title(f"n={n} datasets · geometric mean speedup: {geomean:.2f}×",
    #              loc="left", color=COLOR_INK_SECONDARY, fontsize=9.5, pad=10)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_FASTER),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_SLOWER),
    ]
    ax.legend(handles, ["Faster than baseline (≥ 1×)", "Slower than baseline (< 1×)"],
              loc="upper right", frameon=False, fontsize=19, labelcolor=COLOR_INK_SECONDARY)

    fig.savefig(out_path)
    print(f"Wrote {out_path}  ({n} datasets plotted, geomean speedup {geomean:.2f}×)")

    if excluded:
        print(f"Excluded {len(excluded)} dataset(s) with no baseline-vs-Splyce comparison available:")
        for name, scf, splyce in excluded:
            print(f"  {name}: scf_median_s={scf!r} splyce_median_s={splyce!r}")


def main():
    args = sys.argv[1:]
    csv_path = os.path.join(SCRIPT_DIR, "spmspv_realworld_sweep_results.csv")
    out_path = None
    mode = "multicore"
    sort_by = "density"
    if "--csv" in args:
        csv_path = args[args.index("--csv") + 1]
    if "--out" in args:
        out_path = args[args.index("--out") + 1]
    if "--mode" in args:
        mode = args[args.index("--mode") + 1]
    if "--sort" in args:
        sort_by = args[args.index("--sort") + 1]
    if sort_by not in ("density", "speedup"):
        sys.exit(f"error: unsupported --sort '{sort_by}' (supported: density, speedup)")

    if not os.path.isfile(csv_path):
        sys.exit(f"error: {csv_path} not found — run run_suitesparse_sweep.py first")

    data = load_data(csv_path, mode)

    if out_path is not None:
        # Single explicit output requested — just the one ordering.
        plot(*data, sort_by=sort_by, out_path=out_path)
    else:
        # Default: both orderings, one file each.
        plot(*data, sort_by="density",
             out_path=os.path.join(SCRIPT_DIR, "spmspv_realworld_speedup_by_density.pdf"))
        plot(*data, sort_by="speedup",
             out_path=os.path.join(SCRIPT_DIR, "spmspv_realworld_speedup_by_speedup.pdf"))


if __name__ == "__main__":
    main()
