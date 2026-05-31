"""
Ablation 2: WPD drop_ll J sweep at r=12 and r=16.
WPD baseline (no drop_ll) = J=inf.
x-axis: FID (lower=better, inverted), y-axis: mIoU and DepSSIM.

Usage:
    python plot_ablation_J_sweep.py --output outputs/ablation_J_sweep.png
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# x-axis positions: J=3,4,5 then "no drop_ll"
X_POS    = [0, 1, 2, 3]
X_LABELS = ["3", "4", "5", "no drop_ll"]

DATA = {
    "r12": [
        {"fid": 72.16, "miou": 37.00, "dep": 0.8182},  # J=3
        {"fid": 73.84, "miou": 43.50, "dep": 0.8394},  # J=4
        {"fid": 81.72, "miou": 44.81, "dep": 0.8468},  # J=5
        {"fid": 85.41, "miou": 47.22, "dep": 0.8637},  # no drop_ll (baseline r12)
    ],
    "r16": [
        {"fid": 73.61, "miou": 38.73, "dep": 0.8383},  # J=3
        {"fid": 76.70, "miou": 44.31, "dep": 0.8512},  # J=4
        {"fid": 85.99, "miou": 45.84, "dep": 0.8554},  # J=5
        {"fid": 87.87, "miou": 48.28, "dep": 0.8693},  # no drop_ll (baseline r16)
    ],
}

COLORS = {"r12": "#1f77b4", "r16": "#ff7f0e"}
MARKERS = {"r12": "o", "r16": "s"}


def plot_panel(ax, metric_key, ylabel, title):
    for radius, color, marker in [("r12", COLORS["r12"], MARKERS["r12"]),
                                   ("r16", COLORS["r16"], MARKERS["r16"])]:
        vals = [d[metric_key] for d in DATA[radius]]
        ax.plot(X_POS, vals, marker=marker, linestyle="-", color=color,
                label=f"radius={radius}", zorder=3)

    ax.set_xticks(X_POS)
    ax.set_xticklabels(X_LABELS, fontsize=10)
    ax.set_xlabel("J  (drop_ll decomposition depth)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, linestyle="--")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/ablation_J_sweep.png")
    args = parser.parse_args()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 4))
    plot_panel(ax1, "fid",  "FID↓",       "Distribution Realism")
    plot_panel(ax2, "miou", "mIoU↑ (%)", "Semantic Preservation")
    plot_panel(ax3, "dep",  "DepSSIM↑",   "Depth Preservation")
    # Only show legend on first panel
    ax2.get_legend().remove()
    ax3.get_legend().remove()
    fig.suptitle("Ablation: drop_ll decomposition depth J", fontsize=13)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
