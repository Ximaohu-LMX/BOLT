#!/usr/bin/env python3
"""Render F1 with neutral supervision names and the empirical-figure palette.

Real AUTHOR/A0 values replace the illustrative values. The original physical
canvas, typography and colors are retained. Endpoint supervision is described
in the caption rather than by separate filled markers.
The shared linear axis limits fit the real data with marker padding.
The order-only panel is the new unbounded, anchored monotonic model.
Only F1 is written; no experimental results change.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from render_layout_figures import COLORS, PAPER, plt


def render(preview=None, case_dir=None, output=None):
    # LaTeX: \begin{figure}[t]  ... \includegraphics[width=\linewidth]{figs/fig1_bimo_overview.pdf}
    # Actual saved predictions; all layout settings below retain the original.
    folder = case_dir if case_dir is not None else PAPER / "results/figure1_filled_case"
    source = folder / "nodes.csv"
    provenance = json.loads((folder / "provenance.json").read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == provenance["nodes_sha256"]
    rows = list(csv.DictReader(source.open()))
    assert len(rows) == 8
    anchors = [i for i, r in enumerate(rows) if r["is_training_anchor"] == "True"]
    assert anchors == [0, 7]
    DATA = {
        "patterns": [r["query"] for r in rows],
        "lb": [float(r["lower_bound"]) for r in rows],
        "ub": [float(r["upper_bound"]) for r in rows],
        "true": [float(r["true_count"]) for r in rows],
        "panels": [
            {"title": title, "pred": [float(r[method]) for r in rows], "anchors": anchors}
            for title, method in [
                ("(a) bound-only", "BOUND"),
                ("(b) order-only", "NO_BOUND_MONO"),
                ("(c) order+bound", "BIMO"),
            ]
        ],
    }
    assert np.all(np.array(DATA["lb"]) <= DATA["true"])
    assert np.all(np.array(DATA["true"]) <= DATA["ub"])

    labels = [f"p{i}" for i in range(1, len(DATA["patterns"]) + 1)]
    x = np.arange(len(labels), dtype=float)
    lb = np.array(DATA["lb"], dtype=float)
    ub = np.array(DATA["ub"], dtype=float)
    true = np.array(DATA["true"], dtype=float)
    # One shared linear range covers every bound, truth and plotted estimate.
    # Padding below zero keeps the near-zero node markers fully inside the axes.
    values = np.concatenate([lb, ub, true] +
                            [np.asarray(p["pred"], dtype=float) for p in DATA["panels"]])
    tick_step = 10 ** np.floor(np.log10(max(float(values.max()), 1.0)))
    tick_top = np.ceil(values.max() / tick_step) * tick_step
    pad = 0.05 * tick_top
    y_limits = (min(0.0, float(values.min())) - pad, tick_top + pad)
    y_ticks = np.arange(0, tick_top + tick_step / 2, tick_step)
    # An explicit case-specific linear viewport may reduce unused headroom.
    # Keep the default autoscale for candidate galleries and other cases.
    if "plot" in provenance:
        plot = provenance["plot"]
        assert plot.get("scale", "linear") == "linear"
        y_limits = tuple(plot["y_limits"])
        y_ticks = np.asarray(plot["y_ticks"], dtype=float)
        assert y_limits[0] < values.min() and values.max() < y_limits[1]
        assert np.all((y_ticks >= y_limits[0]) & (y_ticks <= y_limits[1]))
    line_color = COLORS["CLIQUE"]
    bounds_fill_color = "#aeb7bf"
    inactive_fill_color = "#d9d9d9"
    callout_color = COLORS["BIMO"]

    ref_ratio = 423.066 / 188.524  # figs/fig1_bimo_overview_0.pdf 的宽高比
    src_scale = 2.0
    s_pt = src_scale
    s_area = src_scale ** 2
    # fig1 生成 2x 源图；LaTeX 中 width=\linewidth 缩回后观感与单栏版一致。
    fig, axes = plt.subplots(
        1, 3, figsize=(3.33 * src_scale, (3.33 / ref_ratio) * src_scale),
        sharey=True,
        gridspec_kw={"wspace": 0.10},
    )
    for panel_idx, (ax, panel) in enumerate(zip(axes, DATA["panels"])):
        pred = np.array(panel["pred"], dtype=float)

        interval_color = inactive_fill_color if panel_idx == 1 else bounds_fill_color
        ax.fill_between(x, lb, ub, color=interval_color, alpha=0.36, zorder=1)
        ax.plot(x, lb, color=interval_color, linewidth=0.6 * s_pt, alpha=0.95, zorder=2)
        ax.plot(x, ub, color=interval_color, linewidth=0.6 * s_pt, alpha=0.95, zorder=2)
        ax.plot(x, true, color=COLORS["BOUND"],
                linewidth=0.8 * s_pt, linestyle="--", alpha=0.9, zorder=3,
                label="Ground truth")
        ax.plot(x, pred, color=line_color, linewidth=0.95 * s_pt, zorder=3)

        ax.scatter(x, pred, s=14 * s_area, facecolor="white",
                   edgecolor=line_color, linewidth=0.75 * s_pt, zorder=4)

        ax.set_xticks(x)
        ax.set_xticklabels(
            labels, rotation=42, ha="right",
            rotation_mode="anchor",
        )
        ax.tick_params(axis="both", labelsize=6.2 * s_pt, pad=0.8 * s_pt,
                       width=0.6 * s_pt, length=2.0 * s_pt)
        ax.set_title(panel["title"], fontsize=6.8 * s_pt, pad=1.5 * s_pt)
        ax.set_xlim(x[0] - 0.35, x[-1] + 0.35)
        ax.set_ylim(*y_limits)
        ax.set_yticks(y_ticks)
        ax.grid(True, linewidth=0.3 * s_pt)
        for spine in ax.spines.values():
            spine.set_linewidth(0.6 * s_pt)

    callout_arrow = dict(arrowstyle="-", color=callout_color, lw=0.5 * s_pt,
                         shrinkA=1 * s_pt, shrinkB=2 * s_pt, alpha=0.9,
                         relpos=(1.0, 0.5))
    callout_text = dict(fontsize=5.8 * s_pt, color=callout_color,
                        fontweight="bold", ha="left", va="center", zorder=6)
    # Keep the bounds label above the band, with a vertical leader into it.
    bounds_x = 3.5
    axes[0].annotate("[LB, UB] bounds", xy=(bounds_x, 135.0),
                     xytext=(bounds_x, 212.0), textcoords="data",
                     arrowprops={**callout_arrow, "relpos": (0.5, 0.0)},
                     **{**callout_text, "ha": "center"})
    prediction_y = 190.0
    prediction = DATA["panels"][1]["pred"]
    prediction_x = 6.0 + (prediction_y - prediction[6]) / (prediction[7] - prediction[6])
    axes[1].annotate("Prediction", xy=(prediction_x, prediction_y),
                     xytext=(0.55, prediction_y), textcoords="data",
                     arrowprops=callout_arrow, **callout_text)
    # In the order-only panel, the rising truth segment is separated from
    # the prediction. Point down and right to its interior, not an endpoint.
    truth_y = 66.0
    truth_x = x[-2] + (truth_y - true[-2]) / (true[-1] - true[-2])
    axes[1].annotate("True cardinality", xy=(truth_x, truth_y),
                     xytext=(0.55, 108.0), textcoords="data",
                     arrowprops={**callout_arrow, "arrowstyle": "->",
                                 "relpos": (1.0, 0.0), "shrinkB": 0.0},
                     **callout_text)

    axes[1].set_xlabel("Pattern node", fontsize=6.8 * s_pt, labelpad=2.7 * s_pt)
    axes[0].set_ylabel("Cardinality", fontsize=6.8 * s_pt, labelpad=1.0 * s_pt)

    fig.subplots_adjust(left=0.105, right=0.995, bottom=0.25, top=0.86,
                        wspace=0.10)
    fig.savefig(output if output is not None else PAPER / "figs/fig1_bimo_overview.pdf",
                metadata={"CreationDate": None, "ModDate": None})
    if preview:
        fig.savefig(preview, dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", type=Path)
    parser.add_argument("--case-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    render(args.preview, args.case_dir, args.output)
