"""
Ablation 1: PPD (FFT) vs WPD baseline vs WPD J=4 drop_ll — radius sweep.
Two panels: KID (x, inverted) vs mIoU (left) and DepSSIM (right).
Mirrors Slide13_Ablation.tsx layout.

Usage:
    python plot_ablation_baseline_vs_ppd.py --output figures/ablation_baseline_vs_ppd.png
    python plot_ablation_baseline_vs_ppd.py --output figures/ablation_baseline_vs_ppd.png --label-radii
"""
import argparse
import os
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

LOG_DIR = "logs/vkitti_eval"
RADII   = [4, 8, 10, 12, 20, 24, 32]

# Match slide colours
COLORS = {
    "ppd":      "#f59e0b",   # amber  — NeuralRemaster / PPD
    "baseline": "#34d399",   # emerald — ψ-PD w/o drop_ℓ
    "dropll":   "#60a5fa",   # blue    — ψ-PD (ours)
}
LABELS = {
    "ppd":      "PPD (FFT)",
    "baseline": r"$\psi$-PD w/o drop_$\ell$",
    "dropll":   r"$\psi$-PD J=4 drop_$\ell\ell$ (ours)",
}
MARKERS = {
    "ppd":      "s",    # square
    "baseline": "o",    # circle
    "dropll":   "^",    # triangle
}

# Input (raw sim) reference
INPUT = {"kid": 0.0606, "miou": 50.39, "dep": 0.9002}

BASELINE_KEY = {
    4: "baseline_r4", 8: "baseline_r8", 10: "baseline_r10",
    12: "baseline_r12", 16: "baseline_newprompt",
    20: "baseline_r20", 24: "baseline_r24",
}
_KEY_OVERRIDES = {"baseline": BASELINE_KEY}


def extract(path, pattern):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        text = f.read()
    m = re.findall(pattern, text)
    return float(m[-1]) if m else None


def load_variant(prefix, r):
    overrides = _KEY_OVERRIDES.get(prefix, {})
    key = overrides[r] if r in overrides else f"{prefix}_r{r}"
    base = os.path.join(LOG_DIR, key)
    fid_log = f"{base}_fid_clean.log" if os.path.exists(f"{base}_fid_clean.log") else f"{base}_fid.log"
    return {
        "fid":  extract(fid_log,             r"FID:\s+([\d.]+)"),
        "kid":  extract(fid_log,             r"KID:\s+([\d.]+)"),
        "miou": extract(f"{base}_miou.log",  r"mIoU\b.*?([\d.]+)%"),
        "dep":  extract(f"{base}_depth.log", r"Depth SSIM:\s+([\d.]+)"),
    }


def valid_radii(data):
    return [r for r in RADII
            if all(v is not None for v in data[r].values())
            and data[r]["kid"] < INPUT["kid"]]


def plot_panel(ax, variants, y_key, ylabel, title, label_radii):
    for key, data, vr in variants:
        xs = [data[r]["kid"] for r in vr]
        ys = [data[r][y_key] for r in vr]
        ax.plot(xs, ys, marker=MARKERS[key], color=COLORS[key],
                linewidth=2, markersize=6, markerfacecolor="white",
                markeredgewidth=1.8, label=LABELS[key])
        if label_radii:
            for r, x, y in zip(vr, xs, ys):
                ax.annotate(f"r{r}", (x, y), textcoords="offset points",
                            xytext=(4, 4), fontsize=7, color=COLORS[key])

    # Input crosshair
    ix, iy = INPUT["kid"], INPUT[y_key]
    ax.plot(ix, iy, marker="+", color="#ef4444", markersize=11,
            markeredgewidth=2, linestyle="none", label="Input (raw sim)", zorder=5)

    ax.set_xlabel("KID ↓  (better realism →)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.invert_xaxis()
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, linestyle="--")


CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
]

def load_per_class(key):
    path = os.path.join(LOG_DIR, f"{key}_miou.log")
    if not os.path.exists(path):
        return {}
    text = open(path).read()
    result = {}
    for cls in CLASSES:
        # match e.g. "|      road     |  82.52  |"
        m = re.search(rf"\|\s*{re.escape(cls)}\s*\|\s*([\d.]+)\s*\|", text, re.IGNORECASE)
        if m:
            result[cls] = float(m.group(1))
    return result


def print_per_class_table(radius=12, small_radius=4):
    base_key       = BASELINE_KEY.get(radius, f"baseline_r{radius}")
    base_small_key = BASELINE_KEY.get(small_radius, f"baseline_r{small_radius}")
    ppd_key        = f"ppd_r{radius}"
    drop_key       = f"dropll_J4_r{radius}"

    base       = load_per_class(base_key)
    base_small = load_per_class(base_small_key)
    ppd        = load_per_class(ppd_key)
    drop       = load_per_class(drop_key)

    if not (base and base_small and ppd and drop):
        print(f"[per-class table] missing logs for r={radius} or r={small_radius}")
        return

    active = [c for c in CLASSES if base.get(c, 0) + ppd.get(c, 0) + drop.get(c, 0) > 0]
    # sort by drop_ll delta from baseline r=radius (positive → negative)
    active.sort(key=lambda c: drop.get(c, 0) - base.get(c, 0), reverse=True)

    def fmt(v, ref):
        delta = v - ref
        sign = "+" if delta >= 0 else "−"
        return f"{v:.1f} ({sign}{abs(delta):.1f})"

    # Find largest drop per column (most negative delta)
    def worst_class(data, ref_data):
        deltas = {c: data.get(c, 0) - ref_data.get(c, 0) for c in active}
        return min(deltas, key=deltas.get)

    worst_bs = worst_class(base_small, base)
    worst_p  = worst_class(ppd, base)
    worst_d  = worst_class(drop, base)

    def bold(s, condition):
        return f"*{s}*" if condition else s

    col1 = f"ψ-PD w/o drop_ℓ r={radius}"
    col2 = f"ψ-PD w/o drop_ℓ r={small_radius}"
    col3 = f"PPD (FFT) r={radius}"
    col4 = f"ψ-PD drop_ℓℓ r={radius}"
    print(f"\nPer-class mIoU  (reference: {col1})")
    print(f"{'Class':<16} {col1:>24} {col2:>26} {col3:>22} {col4:>26}")
    print("-" * 100)
    for c in active:
        b  = base.get(c, 0)
        bs = base_small.get(c, 0)
        p  = ppd.get(c, 0)
        d  = drop.get(c, 0)
        s_bs = bold(fmt(bs, b), c == worst_bs)
        s_p  = bold(fmt(p,  b), c == worst_p)
        s_d  = bold(fmt(d,  b), c == worst_d)
        print(f"{c:<16} {b:>24.1f} {s_bs:>26} {s_p:>22} {s_d:>26}")

    def _miou(d):
        vals = [d.get(c, 0) for c in CLASSES]
        present = [v for v in vals if v > 0]
        return sum(present) / len(present) if present else 0.0

    mb, mbs, mp, md = _miou(base), _miou(base_small), _miou(ppd), _miou(drop)
    print("-" * 100)
    print(f"{'mIoU':<16} {mb:>24.2f} {bold(fmt(mbs, mb), True):>26} {bold(fmt(mp, mb), True):>22} {bold(fmt(md, mb), True):>26}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="figures/ablation_baseline_vs_ppd.png")
    parser.add_argument("--label-radii", action="store_true")
    parser.add_argument("--table-radius", type=int, default=12,
                        help="Print per-class mIoU table at this radius (default: 12)")
    args = parser.parse_args()

    print_per_class_table(args.table_radius)

    ppd      = {r: load_variant("ppd",       r) for r in RADII}
    baseline = {r: load_variant("baseline",  r) for r in RADII}
    dropll   = {r: load_variant("dropll_J4", r) for r in RADII}

    variants = [
        ("ppd",      ppd,      valid_radii(ppd)),
        ("baseline", baseline, valid_radii(baseline)),
        ("dropll",   dropll,   valid_radii(dropll)),
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

    plot_panel(ax1, variants, "miou", "mIoU ↑ (%)", "Realism vs Semantic Preservation", args.label_radii)
    plot_panel(ax2, variants, "dep",  "Depth-SSIM ↑", "Realism vs Depth Preservation",   args.label_radii)

    for ax in (ax1, ax2):
        leg = ax.get_legend()
        leg.get_frame().set_facecolor("#1a1a2e")
        leg.get_frame().set_edgecolor("#ffffff30")
        for text in leg.get_texts():
            text.set_color("white")

    fig.suptitle("Ablation: FFT vs DT-ℂWPT phase injection  (vKITTI→KITTI)",
                 fontsize=13, color="white")
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
