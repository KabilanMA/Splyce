#!/usr/bin/env python3
# plot_results.py — Plot vector-width scaling from run.sh's results.csv
# (columns: kernel,configuration,sparsity_pct,exec_time_s).
#
# Draws one speedup-vs-vector-width line per sparsity level in
# SPARSITY_LEVELS (10%, 5%, 1%), each Splyce vector-width's speedup
# computed against that same sparsity level's scf baseline execution time
# (baseline_time / splyce_vw_N_time) — so each line isolates vector-width's
# effect on its own, with sparsity held fixed within the line.
#
# Usage:
#   ./plot_results.py                        # reads ./results.csv
#   ./plot_results.py --csv path/to.csv --out path/to.png

import csv
import os
import sys

import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

COLOR_SURFACE = "#fcfcfb"
COLOR_INK_PRIMARY = "#0b0b0b"
COLOR_INK_SECONDARY = "#52514e"
COLOR_INK_MUTED = "#898781"
COLOR_GRIDLINE = "#e1e0d9"

# One color per sparsity line — blue/red match the diverging pair used
# elsewhere in this repo's plots; brown is added as a third, visually
# distinct series color (this chart has 3 lines, not 2).
SPARSITY_STYLE = {
    10: {"color": "#2a78d6", "marker": "o", "label": "10% sparsity"},
    5:  {"color": "#e34948", "marker": "s", "label": "5% sparsity"},
    1:  {"color": "#8b5a2b", "marker": "o", "label": "1% sparsity"},  # + an 'x' overlay, see below
}
VECTOR_WIDTHS = (2, 4, 8, 16)


def load_rows(csv_path):
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    args = sys.argv[1:]
    csv_path = os.path.join(SCRIPT_DIR, "results.csv")
    out_path = os.path.join(SCRIPT_DIR, "vector_width_speedup_plot.png")
    if "--csv" in args:
        csv_path = args[args.index("--csv") + 1]
    if "--out" in args:
        out_path = args[args.index("--out") + 1]

    if not os.path.isfile(csv_path):
        sys.exit(f"error: {csv_path} not found — run ./run.sh first")

    rows = load_rows(csv_path)
    if not rows:
        sys.exit(f"error: {csv_path} has no data rows")

    # exec_time[sparsity_pct][configuration] = exec_time_s
    exec_time = {}
    for row in rows:
        sparsity_pct = int(float(row["sparsity_pct"]))
        exec_time.setdefault(sparsity_pct, {})[row["configuration"]] = float(row["exec_time_s"])

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150, facecolor=COLOR_SURFACE, layout="constrained")
    ax.set_facecolor(COLOR_SURFACE)

    print("Speedup vs. scf baseline, by vector-width:")
    for sparsity_pct in (10, 5, 1):
        if sparsity_pct not in exec_time or "scf" not in exec_time[sparsity_pct]:
            sys.exit(f"error: {csv_path} has no scf (baseline) row at {sparsity_pct}% sparsity")
        baseline_time = exec_time[sparsity_pct]["scf"]

        speedups = []
        for vw in VECTOR_WIDTHS:
            config = f"splyce_vw_{vw}"
            if config not in exec_time[sparsity_pct]:
                sys.exit(f"error: {csv_path} has no {config} row at {sparsity_pct}% sparsity")
            speedups.append(baseline_time / exec_time[sparsity_pct][config])

        style = SPARSITY_STYLE[sparsity_pct]
        ax.plot(VECTOR_WIDTHS, speedups, color=style["color"], linewidth=1.75,
                marker=style["marker"], markersize=7, markerfacecolor=style["color"],
                markeredgecolor=style["color"], label=style["label"], zorder=3)

        # 1% sparsity's marker is a filled circle with a cross inside it —
        # drawn as a plain 'x' scatter layered on top of that line's
        # circle markers, since matplotlib has no single built-in marker
        # for "circle containing a cross".
        if sparsity_pct == 1:
            ax.scatter(VECTOR_WIDTHS, speedups, marker="x", s=22,
                       color=COLOR_SURFACE, linewidths=1.3, zorder=4)

        print(f"  {sparsity_pct}% sparsity: " +
              "  ".join(f"vw={vw}: {s:.2f}x" for vw, s in zip(VECTOR_WIDTHS, speedups)))

    ax.set_xticks(VECTOR_WIDTHS)
    ax.set_xticklabels([str(vw) for vw in VECTOR_WIDTHS])

    ax.set_xlabel("Vectorizing Factor (n)", color=COLOR_INK_SECONDARY, fontsize=10)
    ax.set_ylabel("Speedup (×)", color=COLOR_INK_SECONDARY, fontsize=10)

    ax.grid(True, color=COLOR_GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0, colors=COLOR_INK_MUTED, labelsize=9)

    fig.suptitle("Vector-Width Scaling — spgemm", x=0.0, ha="left",
                 color=COLOR_INK_PRIMARY, fontsize=13, fontweight="bold")
    ax.legend(loc="best", frameon=False, fontsize=9, labelcolor=COLOR_INK_SECONDARY)

    fig.savefig(out_path, facecolor=COLOR_SURFACE)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
