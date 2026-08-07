#!/usr/bin/env python3
# plot_results.py — Plot the TMA (Top-Down Microarchitecture Analysis)
# pipeline-slot breakdown from run.sh's tma_results.csv (columns: name,
# retiring_pct,backend_bound_pct,frontend_bound_pct,bad_speculation_pct,
# branch_misses,exec_time_s,exec_time_min_s,exec_time_max_s,
# exec_time_stdev_s,instructions,ipc).
#
# Draws one 100%-stacked column per configuration (9 total: the scf
# baseline, plus one per Splyce phase-select combination in CONFIG_ORDER),
# each column split into the same four TMA categories that partition a
# CPU's pipeline slots: Retiring, Backend Bound, Frontend Bound, and Bad
# Speculation. Stacked bottom-to-top as Retiring / Backend Bound / Frontend
# Bound / Bad Speculation, with Retiring at the base of each column.
#
# Usage:
#   ./plot_results.py                        # reads ./tma_results.csv
#   ./plot_results.py --csv path/to.csv --out path/to.png

import csv
import os
import sys

import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

COLOR_SURFACE = "#fcfcfb"
COLOR_INK_PRIMARY = "#0b0b0b"

COLOR_LEGEND_BOX = "#d6d6d6"

COLOR_RETIRING = "#009e73"          # green
COLOR_BACKEND_BOUND = "#d55e00"     # orange
COLOR_FRONTEND_BOUND = "#56b4e9"    # blue
COLOR_BAD_SPECULATION = "#e69f00"   # yellow

# This is a bit-reversal permutation of 000..111 (each phase-select code's
# bits read back-to-front), not numeric order — that's the ablation
# ordering the phase_ablation study is built around, not something derived
# from the data.
CONFIG_ORDER = ["000", "100", "010", "110", "001", "101", "011", "111"]

# Unicode "circled digit" characters (U+2460 CIRCLED DIGIT ONE, etc.) — one
# per phase-select config, 1-indexed to match CONFIG_ORDER.
CIRCLED_DIGITS = ["①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧"]


def load_rows(csv_path):
    with open(csv_path, newline="") as f:
        return {row["name"]: row for row in csv.DictReader(f)}


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main():
    args = sys.argv[1:]
    csv_path = os.path.join(SCRIPT_DIR, "tma_results.csv")
    out_path = os.path.join(SCRIPT_DIR, "tma_breakdown_plot.png")
    if "--csv" in args:
        csv_path = args[args.index("--csv") + 1]
    if "--out" in args:
        out_path = args[args.index("--out") + 1]

    if not os.path.isfile(csv_path):
        sys.exit(f"error: {csv_path} not found — run ./run.sh first")

    rows_by_name = load_rows(csv_path)
    if not rows_by_name:
        sys.exit(f"error: {csv_path} has no data rows")

    labels = ["Baseline"] + CIRCLED_DIGITS
    row_names = ["spgemm_scf"] + [f"spgemm_splyce_phase_{phase}" for phase in CONFIG_ORDER]

    retiring, backend, frontend, bad_spec = [], [], [], []
    missing = []
    for label, name in zip(labels, row_names):
        row = rows_by_name.get(name)
        r = to_float(row["retiring_pct"]) if row else None
        be = to_float(row["backend_bound_pct"]) if row else None
        fe = to_float(row["frontend_bound_pct"]) if row else None
        bs = to_float(row["bad_speculation_pct"]) if row else None
        if None in (r, be, fe, bs):
            missing.append(label)
            r = be = fe = bs = 0.0
        retiring.append(r)
        backend.append(be)
        frontend.append(fe)
        bad_spec.append(bs)

    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150, facecolor=COLOR_SURFACE, layout="constrained")
    ax.set_facecolor(COLOR_SURFACE)

    # Stacked bottom-to-top: Retiring, Backend Bound, Frontend Bound, Bad
    # Speculation — so visually, top-to-bottom, it's Bad Speculation /
    # Frontend Bound / Backend Bound / Retiring, with Retiring at the base.
    BAR_WIDTH = 0.8
    bottom = [0.0] * len(labels)
    ax.bar(x, retiring, bottom=bottom, width=BAR_WIDTH, color=COLOR_RETIRING, label="Retiring", zorder=3)
    seam1 = [b + v for b, v in zip(bottom, retiring)]
    ax.bar(x, backend, bottom=seam1, width=BAR_WIDTH, color=COLOR_BACKEND_BOUND, label="Backend Bound", zorder=3)
    seam2 = [b + v for b, v in zip(seam1, backend)]
    ax.bar(x, frontend, bottom=seam2, width=BAR_WIDTH, color=COLOR_FRONTEND_BOUND, label="Frontend Bound", zorder=3)
    seam3 = [b + v for b, v in zip(seam2, frontend)]
    ax.bar(x, bad_spec, bottom=seam3, width=BAR_WIDTH, color=COLOR_BAD_SPECULATION, label="Bad Speculation", zorder=3)

    # White seam lines only at the 3 internal boundaries between segments —
    # not at the very top or bottom of the column, so the gap reads as
    # separation between sections rather than a border around the whole bar.
    for seam in (seam1, seam2, seam3):
        for xi, yi in zip(x, seam):
            ax.hlines(yi, xi - BAR_WIDTH / 2, xi + BAR_WIDTH / 2, color="white", linewidth=1, zorder=4)

    if missing:
        for xi, label in zip(x, labels):
            if label in missing:
                ax.text(xi, 2, "NA", ha="center", va="bottom", fontsize=8, color="black")

    # "Baseline" and the circled digits read at very different sizes for
    # the same fontsize (a word vs. a single glyph), so each gets its own
    # size here rather than one shared fontsize via set_xticklabels.
    BASELINE_FONTSIZE = 14
    CIRCLED_DIGIT_FONTSIZE = 20
    ax.set_xticks(list(x))
    tick_labels = ax.set_xticklabels(labels)
    for i, tick_label in enumerate(tick_labels):
        tick_label.set_fontsize(BASELINE_FONTSIZE if i == 0 else CIRCLED_DIGIT_FONTSIZE)
    ax.set_xlabel("Configuration", color="black", fontsize=20, fontweight="bold")
    ax.set_ylabel("Pipeline Slots (%)", color="black", fontsize=20, fontweight="bold")
    ax.set_ylim(0, 100)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_color(COLOR_INK_PRIMARY)
    ax.spines["bottom"].set_color(COLOR_INK_PRIMARY)
    ax.spines["left"].set_linewidth(1)
    ax.spines["bottom"].set_linewidth(1)
    # Spines default to a lower zorder than the bars (zorder=3), so the
    # bottom spine was getting painted over by the Retiring segment sitting
    # right on top of it at y=0 — raise both above the bars/seam lines.
    ax.spines["left"].set_zorder(5)
    ax.spines["bottom"].set_zorder(5)
    # Split x/y so this doesn't clobber set_xticklabels's fontsize above —
    # tick_params(labelsize=...) applies to both axes and would otherwise
    # silently override whatever size was set there.
    ax.tick_params(axis="x", length=4, colors=COLOR_INK_PRIMARY, labelcolor="black")
    ax.tick_params(axis="y", length=4, colors=COLOR_INK_PRIMARY, labelsize=9, labelcolor="black")

    # Legend order matches the requested 1-4 (Retiring, Backend Bound,
    # Frontend Bound, Bad Speculation), independent of stacking order.
    handles, labels_ = ax.get_legend_handles_labels()
    order = ["Retiring", "Backend Bound", "Frontend Bound", "Bad Speculation"]
    handles = [handles[labels_.index(name)] for name in order]
    legend = ax.legend(handles, order, loc="lower center", bbox_to_anchor=(0.5, 1.06), ncol=4,
              frameon=True, fontsize=11, labelcolor="black",
              handlelength=2.2, columnspacing=3)
    legend.get_frame().set_edgecolor(COLOR_LEGEND_BOX)
    legend.get_frame().set_facecolor(COLOR_SURFACE)
    legend.get_frame().set_linewidth(1)

    fig.suptitle("TMA Pipeline-Slot Breakdown — spgemm Phase Ablation", x=0.0, ha="left",
                 color="black", fontsize=13, fontweight="bold")
    subtitle = "  ".join(f"{d} {phase}" for d, phase in zip(CIRCLED_DIGITS, CONFIG_ORDER))
    ax.set_title(subtitle, loc="left", color="black", fontsize=8.5, pad=6)

    fig.savefig(out_path, facecolor=COLOR_SURFACE)
    print(f"Wrote {out_path}")
    if missing:
        print(f"Missing/NA data for: {', '.join(missing)}")


if __name__ == "__main__":
    main()
