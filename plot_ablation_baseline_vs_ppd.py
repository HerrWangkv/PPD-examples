"""
Ablation 1: WPD baseline vs PPD (correct FFT) vs WPD J=4 drop_ll across radius sweep.
Two panels, each with dual y-axes:
  Panel 1: x=radius, left y=FID, right y=KID
  Panel 2: x=radius, left y=mIoU, right y=DepSSIM
Reads directly from logs/vkitti_eval/.

Usage:
    python plot_ablation_baseline_vs_ppd.py --output figures/ablation_baseline_vs_ppd.png
"""
import argparse
import os
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_DIR = "logs/vkitti_eval"
RADII   = [8, 12, 16, 20, 24, 32]

COLORS = {
    "ppd":       "#e07b39",
    "baseline":  "#2ca02c",
    "dropll":    "#1f77b4",
    "dropll_j5": "#9467bd",
}
LABELS = {
    "ppd":       "PPD (FFT)",
    "baseline":  "WPD baseline",
    "dropll":    "WPD J=4 drop_ll (ours)",
    "dropll_j5": "WPD J=5 drop_ll",
}


def extract(path, pattern):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        text = f.read()
    m = re.findall(pattern, text)
    return float(m[-1]) if m else None


BASELINE_KEY  = {8: "baseline_r8", 12: "baseline_r12", 16: "baseline_newprompt",
                 20: "baseline_r20", 24: "baseline_r24"}
DROPLL_J5_KEY = {12: "dropll_J5_r12", 16: "dropll_step6000_J5"}


def load_variant(prefix, r):
    if prefix == "baseline":
        key = BASELINE_KEY.get(r, f"baseline_r{r}")
    elif prefix == "dropll_j5":
        key = DROPLL_J5_KEY.get(r)
        if key is None:
            return {"fid": None, "kid": None, "miou": None, "dep": None}
    else:
        key = f"{prefix}_r{r}"
    base = os.path.join(LOG_DIR, key)
    fid_log = f"{base}_fid_clean.log" if os.path.exists(f"{base}_fid_clean.log") else f"{base}_fid.log"
    return {
        "fid":  extract(fid_log,               r"FID:\s+([\d.]+)"),
        "kid":  extract(fid_log,               r"KID:\s+([\d.]+)"),
        "miou": extract(f"{base}_miou.log",    r"mIoU\b.*?([\d.]+)%"),
        "dep":  extract(f"{base}_depth.log",   r"Depth SSIM:\s+([\d.]+)"),
    }


def get_series(data, radii, key):
    return [data[r][key] for r in radii if data[r][key] is not None]


def valid_radii(data):
    return [r for r in RADII if all(v is not None for v in data[r].values())]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="figures/ablation_baseline_vs_ppd.png")
    parser.add_argument("--label-radii", action="store_true", help="Annotate each point with its radius value")
    args = parser.parse_args()

    ppd       = {r: load_variant("ppd",       r) for r in RADII}
    baseline  = {r: load_variant("baseline",  r) for r in RADII}
    dropll    = {r: load_variant("dropll_J4", r) for r in RADII}
    dropll_j5 = {r: load_variant("dropll_j5", r) for r in RADII}

    vr_ppd  = valid_radii(ppd)
    vr_base = valid_radii(baseline)
    vr_drop = valid_radii(dropll)
    vr_dj5  = valid_radii(dropll_j5)

    variants = [
        ("ppd",       ppd,       vr_ppd),
        ("baseline",  baseline,  vr_base),
        ("dropll",    dropll,    vr_drop),
        ("dropll_j5", dropll_j5, vr_dj5),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    def plot_panel(ax, y_key, ylabel, title):
        for key, data, vr in variants:
            kid_vals = [data[r]["kid"]  for r in vr]
            y_vals   = [data[r][y_key] for r in vr]
            ax.plot(kid_vals, y_vals, "o-", color=COLORS[key], label=LABELS[key])
            if args.label_radii:
                for r, kx, ky in zip(vr, kid_vals, y_vals):
                    ax.annotate(f"r{r}", (kx, ky), textcoords="offset points",
                                xytext=(4, 4), fontsize=7, color=COLORS[key])
        ax.set_xlabel("KID↓", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.invert_xaxis()
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25, linestyle="--")

    plot_panel(ax1, "miou", "mIoU↑ (%)", "Realism vs Semantic Preservation")
    plot_panel(ax2, "dep",  "DepSSIM↑",  "Realism vs Depth Preservation")

    fig.suptitle("Ablation: PPD vs WPD baseline vs WPD J=4 drop_ll", fontsize=13)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
