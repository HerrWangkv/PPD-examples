"""
Ablation 1, 4-panel version: PPD (FFT) vs ψ-PD baseline vs ψ-PD J=4.

Columns    — x = KID (left), CLIP-Residual (right)
Rows       — y = Depth-SSIM (top), mIoU (bottom)

All metrics parsed from logs/vkitti_eval/.

Usage:
    python plot_ablation1_4panel.py --output figures/ablation1_4panel.png --label-radii
"""
import argparse
import os
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_ablation_baseline_vs_ppd import (
    COLORS, LABELS, MARKERS, extract, _KEY_OVERRIDES
)

LOG_DIR = "logs/vkitti_eval"

def load_variant(prefix, r=None):
    if r is not None:
        overrides = _KEY_OVERRIDES.get(prefix, {})
        key = overrides[r] if r in overrides else f"{prefix}_r{r}"
    else:
        key = prefix
    
    base = os.path.join(LOG_DIR, key)
    # Prefer clean-mode KID log
    fid_log = f"{base}_fid_clean.log" if os.path.exists(f"{base}_fid_clean.log") else f"{base}_fid.log"
    
    return {
        "kid":      extract(fid_log,               r"KID:\s+([\d.]+)"),
        "miou":     extract(f"{base}_miou.log",    r"mIoU:\s+([\d.]+)"),
        "dep":      extract(f"{base}_depth.log",   r"Depth SSIM:\s+([\d.]+)"),
        "clip_res": extract(f"{base}_clipres.log",  r"CLIP-Residual:\s+([\d.]+)"),
    }

def style_ax(ax):
    ax.set_facecolor("#0d0d1a")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#ffffff45")
    ax.grid(True, alpha=0.25, linestyle="--")

def plot_curves(ax, variants, x_key, y_key, label_radii):
    for key, data_sweep, radii_for_variant in variants:
        radii = [r for r in radii_for_variant
                 if data_sweep[r].get(x_key) is not None
                 and data_sweep[r].get(y_key) is not None]
        xs = [data_sweep[r][x_key] for r in radii]
        ys = [data_sweep[r][y_key] for r in radii]
        
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

    RADII_PPD      = [8, 12, 20, 24]
    RADII_BASELINE = [4, 8, 10, 20, 24]
    RADII_DROPLL   = [8, 12, 20, 24]

    variants = [
        ("ppd",      {r: load_variant("ppd",       r) for r in RADII_PPD},      RADII_PPD),
        ("baseline", {r: load_variant("baseline",   r) for r in RADII_BASELINE}, RADII_BASELINE),
        ("dropll",   {r: load_variant("dropll_J4",  r) for r in RADII_DROPLL},   RADII_DROPLL),
    ]
    
    input_data = load_variant("input")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.patch.set_facecolor("#0d0d1a")

    # (row, col), x_key, y_key, ylabel, title, is_kid
    panels = [
        (axes[0, 0], "kid",      "dep",  "Depth-SSIM ↑", "KID vs Depth Preservation",         True),
        (axes[0, 1], "clip_res", "dep",  "Depth-SSIM ↑", "CLIP-Res vs Depth Preservation",     False),
        (axes[1, 0], "kid",      "miou", "mIoU ↑ (%)",   "KID vs Semantic Preservation",       True),
        (axes[1, 1], "clip_res", "miou", "mIoU ↑ (%)",   "CLIP-Res vs Semantic Preservation",  False),
    ]

    for ax, x_key, y_key, ylabel, title, is_kid in panels:
        style_ax(ax)
        plot_curves(ax, variants, x_key, y_key, args.label_radii)
        
        # Plot input crosshair
        ix, iy = input_data[x_key], input_data[y_key]
        if ix is not None and iy is not None:
            ax.plot(ix, iy, marker="+", color="#ef4444", markersize=12,
                    markeredgewidth=2.5, linestyle="none", label="Input (raw sim)", zorder=10)

        if is_kid:
            ax.set_xlabel("KID ↓  (better realism →)", fontsize=11)
            ax.invert_xaxis()
        else:
            ax.set_xlabel("CLIP-Residual ↑  (better realism →)", fontsize=11)
            
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        
        leg = ax.legend(fontsize=9, loc="lower left" if not is_kid else "best")
        leg.get_frame().set_facecolor("#1a1a2e")
        leg.get_frame().set_edgecolor("#ffffff30")
        for text in leg.get_texts():
            text.set_color("white")

    fig.suptitle("Ablation: FFT vs DT-ℂWPT phase injection — Distribution (KID) and Perceptual (CLIP-Res) axes",
                 fontsize=14, color="white", y=0.98)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
