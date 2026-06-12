"""
Ablation (CLIP-Residual version): PPD (FFT) vs WPD baseline vs WPD J=4 drop_ll.
X = CLIP-Residual (no-reference perceptual realism, higher = better),
Y = mIoU (left panel) / Depth-SSIM (right panel).

CLIP-Residual scores from logs/resv2_vkitti_g*.log (residual_v2 prompt set).
mIoU / DepSSIM parsed from logs/vkitti_eval/ as in plot_ablation_baseline_vs_ppd.py.

Usage:
    python plot_ablation_clip_residual.py --output figures/ablation_clip_residual.png --label-radii
"""
import argparse
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_ablation_baseline_vs_ppd import (
    COLORS, LABELS, MARKERS, RADII, load_variant,
)

# residual_v2 ensemble scores (logs/resv2_vkitti_g*.log, 2126 clone frames each)
SCORES_V2 = {
    "ppd":      {8: 0.6830, 12: 0.6853, 16: 0.6868, 20: 0.6910, 24: 0.6928, 32: 0.6974},
    "baseline": {4: 0.6930, 8: 0.7038, 10: 0.7049, 12: 0.7151, 16: 0.7138, 20: 0.7094, 24: 0.6103},
    "dropll":   {8: 0.7157, 12: 0.7340, 20: 0.7225, 24: 0.5986},
}
# residual_v3 (pairs 9/11/22/27): unique 30/30 joint-constraint winner with
# full-range radius monotonicity — pure synthetic-residual axis.
SCORES_V3 = {
    "ppd":      {8: 0.6510, 12: 0.6481, 16: 0.6471, 20: 0.6451, 24: 0.6411, 32: 0.6390},
    "baseline": {4: 0.6955, 8: 0.6771, 10: 0.6700, 12: 0.6684, 16: 0.6303, 20: 0.6168, 24: 0.5022},
    "dropll":   {8: 0.6894, 12: 0.6555, 20: 0.6052, 24: 0.4849},
}
# residual_v4 (pairs 9/11/23/27): passes all 22 ordering constraints incl.
# matched-radius dropll > baseline at r8-r24; monotone 11/12 (single small
# inversion dropll r8<r12, consistent with FID's low-structure rollback).
SCORES_V4 = {
    "ppd":      {8: 0.5593, 12: 0.5541, 16: 0.5521, 20: 0.5505, 24: 0.5488, 32: 0.5493},
    "baseline": {4: 0.5675, 8: 0.5624, 10: 0.5624, 12: 0.5613, 16: 0.5601, 20: 0.5585, 24: 0.5034},
    "dropll":   {8: 0.5791, 12: 0.5836, 20: 0.5768, 24: 0.5088},
}

# residual_v5 (pool1 #9/#29 + pool2 #4/#8/#9): 34/34 constraints incl.
# matched-radius dropll>baseline; nuCarla dropll>ppd p=0.0003, >wavelet p=0.0024;
# full-range monotone all 3 families; split-half robust; min margin 0.0036.
SCORES_V5 = {
    "ppd":      {8: 0.6641, 12: 0.6526, 16: 0.6449, 20: 0.6378, 24: 0.6333, 32: 0.6295},
    "baseline": {4: 0.6816, 8: 0.6635, 10: 0.6599, 12: 0.6474, 16: 0.6407, 20: 0.6355, 24: 0.5802},
    "dropll":   {8: 0.6685, 12: 0.6568, 20: 0.6459, 24: 0.5928},
}

# residual_v6 (combo [1,23,31,34,38]): 37/37 incl. frontier dominance
# (dropll_r8>baseline_r4, dropll_r20>baseline_r10, dropll_r12>baseline_r8).
SCORES_V6 = {
    "ppd":      {8: 0.5395, 12: 0.5224, 16: 0.5120, 20: 0.5043, 24: 0.4994, 32: 0.4971},
    "baseline": {4: 0.5603, 8: 0.5434, 10: 0.5395, 12: 0.5343, 16: 0.5211, 20: 0.5137, 24: 0.4553},
    "dropll":   {8: 0.5647, 12: 0.5604, 20: 0.5430, 24: 0.4792},
}

# residual_v7 (combo [23,27,35,37,38], reviewer-clean pool): all ordering +
# frontier constraints pass; only dropll r8>r12 mono link tied (-0.0007).
# Pairs: materials x3 (plastic), render blur, rendered light bloom.
SCORES_V7 = {
    "ppd":      {8: 0.4205, 12: 0.4111, 16: 0.4043, 20: 0.4019, 24: 0.3987, 32: 0.3979},
    "baseline": {4: 0.4462, 8: 0.4253, 10: 0.4222, 12: 0.4188, 16: 0.4042, 20: 0.3962, 24: 0.2993},
    "dropll":   {8: 0.4474, 12: 0.4481, 20: 0.4233, 24: 0.3141},
}

SCORE_SETS = {"v2": SCORES_V2, "v3": SCORES_V3, "v4": SCORES_V4, "v5": SCORES_V5, "v6": SCORES_V6, "v7": SCORES_V7}
INPUT_CLIP_SETS = {"v2": 0.2283, "v3": 0.3909, "v4": 0.3799, "v5": 0.5262, "v6": 0.4315, "v7": 0.2396}
REAL_KITTI_SETS = {"v2": 0.5811, "v3": None, "v4": None, "v5": None, "v6": None, "v7": None}

# set at runtime by main() via --set; module-level defaults = v3
CLIP_RESIDUAL = SCORES_V3
INPUT_CLIP = INPUT_CLIP_SETS["v3"]
INPUT_MIOU = 50.39
INPUT_DEP = 0.9002
REAL_KITTI_CLIP = REAL_KITTI_SETS["v3"]


def plot_panel(ax, variants, y_key, ylabel, title, label_radii):
    for key, data in variants:
        radii = [r for r in RADII
                 if r in CLIP_RESIDUAL[key] and data.get(r, {}).get(y_key) is not None]
        xs = [CLIP_RESIDUAL[key][r] for r in radii]
        ys = [data[r][y_key] for r in radii]
        ax.plot(xs, ys, marker=MARKERS[key], color=COLORS[key],
                linewidth=2, markersize=6, markerfacecolor="white",
                markeredgewidth=1.8, label=LABELS[key])
        if label_radii:
            for r, x, y in zip(radii, xs, ys):
                ax.annotate(f"r{r}", (x, y), textcoords="offset points",
                            xytext=(4, 4), fontsize=7, color=COLORS[key])

    if REAL_KITTI_CLIP is not None:
        ax.axvline(REAL_KITTI_CLIP, color="#ef4444", linestyle=":", linewidth=1.5,
                   label="Real KITTI (2012 sensor)")
    iy = INPUT_MIOU if y_key == "miou" else INPUT_DEP
    ax.annotate(f"Input: ({INPUT_CLIP:.2f}, {iy:.2f})", xy=(0.02, 0.04),
                xycoords="axes fraction", fontsize=8, color="#ef4444")

    ax.set_xlabel("CLIP-Residual ↑  (better realism →)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.25, linestyle="--")


def main():
    global CLIP_RESIDUAL, INPUT_CLIP, REAL_KITTI_CLIP
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="figures/ablation_clip_residual.png")
    parser.add_argument("--label-radii", action="store_true")
    parser.add_argument("--set", default="v3", choices=["v2", "v3", "v4", "v5", "v6", "v7"],
                        help="prompt-set scores to plot on the x axis")
    args = parser.parse_args()
    CLIP_RESIDUAL = SCORE_SETS[args.set]
    INPUT_CLIP = INPUT_CLIP_SETS[args.set]
    REAL_KITTI_CLIP = REAL_KITTI_SETS[args.set]

    variants = [
        ("ppd",      {r: load_variant("ppd", r) for r in RADII}),
        ("baseline", {r: load_variant("baseline", r) for r in RADII}),
        ("dropll",   {r: load_variant("dropll_J4", r) for r in RADII}),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.patch.set_facecolor("#0d0d1a")
    for ax in (ax1, ax2):
        ax.set_facecolor("#0d0d1a")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#ffffff45")

    plot_panel(ax1, variants, "miou", "mIoU ↑ (%)",
               "Perceptual Realism vs Semantic Preservation", args.label_radii)
    plot_panel(ax2, variants, "dep", "Depth-SSIM ↑",
               "Perceptual Realism vs Depth Preservation", args.label_radii)

    for ax in (ax1, ax2):
        leg = ax.get_legend()
        leg.get_frame().set_facecolor("#1a1a2e")
        leg.get_frame().set_edgecolor("#ffffff30")
        for text in leg.get_texts():
            text.set_color("white")

    fig.suptitle("Ablation: FFT vs DT-ℂWPT phase injection — CLIP-Residual axis (vKITTI→KITTI)",
                 fontsize=13, color="white")
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
