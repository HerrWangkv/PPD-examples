"""
Ablation 1: WPD baseline vs PPD across radius sweep.
Shows FID and mIoU vs cutoff radius for both variants.

Usage:
    python plot_ablation_baseline_vs_ppd.py --output outputs/ablation_baseline_vs_ppd.pdf
"""
import argparse
import os
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_DIR = "logs/vkitti_eval"

RADII = [8, 12, 20, 24]

PPD = {
    8:  {"fid": 85.75, "kid": 0.0590, "miou": 43.20, "dep": 0.8347, "absrel": 0.2851},
    12: {"fid": 96.03, "kid": 0.0725, "miou": 46.49, "dep": 0.8607, "absrel": 0.2035},
    16: {"fid": 102.44,"kid": 0.0806, "miou": 47.02, "dep": 0.8646, "absrel": 0.1883},
    20: {"fid": 103.54,"kid": 0.0820, "miou": 46.77, "dep": 0.8662, "absrel": 0.1824},
    24: {"fid": 112.48,"kid": 0.0902, "miou": 47.05, "dep": 0.8789, "absrel": 0.1596},
}

WPD_BASELINE = {
    8:  {"fid": 74.04, "kid": 0.0450, "miou": 44.23, "dep": 0.8335, "absrel": 0.2894},
    12: {"fid": 85.41, "kid": 0.0587, "miou": 47.22, "dep": 0.8637, "absrel": 0.2166},
    16: {"fid": 87.87, "kid": 0.0623, "miou": 48.28, "dep": 0.8693, "absrel": 0.1931},
    20: {"fid": 87.81, "kid": 0.0621, "miou": 48.32, "dep": 0.8688, "absrel": 0.1850},
    24: {"fid": 100.92,"kid": 0.0759, "miou": 48.46, "dep": 0.8814, "absrel": 0.1609},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/ablation_baseline_vs_ppd.pdf")
    args = parser.parse_args()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    ppd_fid  = [PPD[r]["fid"]  for r in RADII]
    ppd_miou = [PPD[r]["miou"] for r in RADII]
    ppd_dep  = [PPD[r]["dep"]  for r in RADII]
    wpd_fid  = [WPD_BASELINE[r]["fid"]  for r in RADII]
    wpd_miou = [WPD_BASELINE[r]["miou"] for r in RADII]
    wpd_dep  = [WPD_BASELINE[r]["dep"]  for r in RADII]

    def plot_panel(ax, ppd_y, wpd_y, ylabel, title):
        ax.plot(ppd_fid, ppd_y,  "o--", color="#e07b39", label="PPD (old LoRA)")
        ax.plot(wpd_fid, wpd_y,  "o-",  color="#2ca02c", label="WPD baseline (new LoRA)")
        for r, x, y in zip(RADII, ppd_fid, ppd_y):
            ax.annotate(f"r{r}", (x, y), textcoords="offset points",
                        xytext=(4, 4), fontsize=7.5, color="#e07b39")
        for r, x, y in zip(RADII, wpd_fid, wpd_y):
            ax.annotate(f"r{r}", (x, y), textcoords="offset points",
                        xytext=(4, -10), fontsize=7.5, color="#2ca02c")
        ax.set_xlabel("FID↓", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.invert_xaxis()
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25, linestyle="--")

    plot_panel(ax1, ppd_miou, wpd_miou, "mIoU↑ (%)", "Realism vs Semantic Preservation")
    plot_panel(ax2, ppd_dep,  wpd_dep,  "DepSSIM↑",  "Realism vs Depth Preservation")

    fig.suptitle("Ablation: WPD baseline vs PPD across radius sweep", fontsize=13)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
