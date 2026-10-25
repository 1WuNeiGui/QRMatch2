"""Plot the gradient cosine similarity between L_DDCL and L_cons over training epochs.

Reads the per-epoch CSV files written by grad_conflict.py and saves the figure as
SVG and PNG.

Examples
--------
Only the run with DTF:
    python plot_grad_conflict.py --run result/grad_conflict_with_dtf_epochs.csv "With DTF"

Both runs (after the run without DTF is finished):
    python plot_grad_conflict.py \
        --run result_wo_dtf/grad_conflict_without_dtf_epochs.csv "Without DTF" \
        --run result/grad_conflict_with_dtf_epochs.csv "With DTF"
"""

import argparse
import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {"without": "#4E9AF1", "with": "#6CC57C"}
FALLBACK_COLORS = ["#6CC57C", "#4E9AF1", "#F2A541", "#B07CC6"]


def read_epochs(path):
    epochs, means = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            value = float(row["cos_mean"])
            if math.isnan(value):
                continue
            epochs.append(int(row["epoch"]))
            means.append(value)
    return epochs, means


def pick_color(label, index):
    key = label.lower()
    if "without" in key or "w/o" in key:
        return COLORS["without"]
    if "with" in key:
        return COLORS["with"]
    return FALLBACK_COLORS[index % len(FALLBACK_COLORS)]


def plot_runs(runs, out, ylim=(-0.6, 0.6)):
    """Plot one or more runs. runs: list of (csv_path, label). Writes out.svg and out.png."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    all_epochs = set()
    for i, (path, label) in enumerate(runs):
        epochs, means = read_epochs(path)
        all_epochs.update(epochs)
        ax.plot(epochs, means, color=pick_color(label, i), linewidth=2, marker="o",
                markersize=7, markeredgecolor="white", markeredgewidth=1.5, label=label,
                zorder=3)

    ax.set_ylim(*ylim)
    ax.set_yticks([-0.6, -0.3, 0.0, 0.3, 0.6])
    ax.set_yticklabels(["-0.6", "-0.3", "0", "0.3", "0.6"])
    if all_epochs:
        ax.set_xticks(sorted(all_epochs))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gradient cosine similarity")

    ax.yaxis.grid(True, linestyle="--", color="#E5E5E5", linewidth=0.8, zorder=0)
    ax.xaxis.grid(False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#BBBBBB")
    ax.tick_params(axis="both", colors="#555555", length=0)
    ax.tick_params(axis="x", labelrotation=0)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=len(runs),
              frameon=False, handlelength=2.5)
    fig.tight_layout()

    out_dir = os.path.dirname(out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out + ".svg")
    fig.savefig(out + ".png", dpi=300)
    plt.close(fig)
    return out + ".svg", out + ".png"


def main():
    parser = argparse.ArgumentParser(description="Plot gradient cosine similarity curves")
    parser.add_argument("--run", nargs=2, action="append", metavar=("CSV", "LABEL"),
                        help="per-epoch CSV file and its legend label; repeat for several runs")
    parser.add_argument("--out", default="result/grad_conflict",
                        help="output path without extension (.svg and .png are written)")
    parser.add_argument("--ylim", nargs=2, type=float, default=[-0.6, 0.6])
    args = parser.parse_args()

    runs = args.run or [["result/grad_conflict_with_dtf_epochs.csv", "With DTF"]]
    svg, png = plot_runs(runs, args.out, args.ylim)
    print("Saved", svg, "and", png)


if __name__ == "__main__":
    main()
