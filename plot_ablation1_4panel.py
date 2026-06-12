"""
Ablation 1, 4-panel version: PPD (FFT) vs ψ-PD baseline vs ψ-PD J=4 drop_ll.

Top row    — x = KID (paired-reference distribution realism, inverted axis)
Bottom row — x = CLIP-Residual v7 (no-reference perceptual realism)
Columns    — y = mIoU (left), Depth-SSIM (right)

KID/mIoU/DepSSIM parsed from logs/vkitti_eval/; CLIP-Residual v7 values from
plot_ablation_clip_residual.SCORES_V7 (logs/vkitti_eval/<v>_clipres.log).

Usage:
    python plot_ablation1_4panel.py --output figures/ablation1_4panel.png --label-radii
"""
import argparse
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_ablation_baseline_vs_ppd import (
    COLORS, LABELS, MARKERS, RADII, INPUT, load_variant,
)
from plot_ablation_clip_residual import SCORES_V7, INPUT_CLIP_SETS

INPUT_CLIP = INPUT_CLIP_SETS["v7"]


def style_ax(ax):
    ax.set_facecolor("#0d0d1a")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#ffffff45")
    ax.grid(True, alpha=0.25, linestyle="--")


def plot_curves(ax, variants, x_of, y_key, label_radii):
    for key, data in variants:
        radii = [r for r in RADII
                 if x_of(key, r, data) is not None
                 and data.get(r, {}).get(y_key) is not None]
        xs = [x_of(key, r, data) for r in radii]
        ys = [data[r][y_key] for r in radii]
        ax.plot(xs, ys, marker=MARKERS[key], color=COLORS[key],
                linewidth=2, markersize=6, markerfacecolor="white",
                markeredgewidth=1.8, label=LABELS[key])
        if label_radii:
            for r, x, y in zip(radii, xs, ys):
                ax.annotate(f"r{r}", (x, y), textcoords="offset points",
                            xytext=(4, 4), fontsize=7, color=COLORS[key])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="figures/ablation1_4panel.png")
    parser.add_argument("--label-radii", action="store_true")
    args = parser.parse_args()

    variants = [
        ("ppd",      {r: load_variant("ppd", r) for r in RADII}),
        ("baseline", {r: load_variant("baseline", r) for r in RADII}),
        ("dropll",   {r: load_variant("dropll_J4", r) for r in RADII}),
    ]

    def x_kid(key, r, data):
        kid = data.get(r, {}).get("kid")
        return kid if kid is not None and kid < INPUT["kid"] else None

    def x_clip(key, r, data):
        return SCORES_V7[key].get(r)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.patch.set_facecolor("#0d0d1a")

    panels = [
        (axes[0, 0], x_kid,  "miou", "mIoU ↑ (%)",   "KID vs Semantic Preservation",  True),
        (axes[0, 1], x_kid,  "dep",  "Depth-SSIM ↑", "KID vs Depth Preservation",     True),
        (axes[1, 0], x_clip, "miou", "mIoU ↑ (%)",   "CLIP-Residual vs Semantic Preservation", False),
        (axes[1, 1], x_clip, "dep",  "Depth-SSIM ↑", "CLIP-Residual vs Depth Preservation",    False),
    ]
    for ax, x_of, y_key, ylabel, title, is_kid in panels:
        style_ax(ax)
        plot_curves(ax, variants, x_of, y_key, args.label_radii)
        if is_kid:
            ix, iy = INPUT["kid"], INPUT[y_key]
            ax.plot(ix, iy, marker="+", color="#ef4444", markersize=11,
                    markeredgewidth=2, linestyle="none", label="Input (raw sim)", zorder=5)
            ax.set_xlabel("KID ↓  (better realism →)", fontsize=11)
            ax.invert_xaxis()
        else:
            iy = INPUT["miou"] if y_key == "miou" else INPUT["dep"]
            ax.annotate(f"Input: ({INPUT_CLIP:.2f}, {iy:.2f})", xy=(0.02, 0.04),
                        xycoords="axes fraction", fontsize=8, color="#ef4444")
            ax.set_xlabel("CLIP-Residual ↑  (better realism →)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        leg = ax.legend(fontsize=8, loc="lower left")
        leg.get_frame().set_facecolor("#1a1a2e")
        leg.get_frame().set_edgecolor("#ffffff30")
        for text in leg.get_texts():
            text.set_color("white")

    fig.suptitle("Ablation 1: FFT vs DT-ℂWPT phase injection — distribution (KID) and perceptual (CLIP-Residual) realism axes",
                 fontsize=12.5, color="white")
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
