"""
FID vs mIoU frontier plot for vKITTI → KITTI benchmark.

Usage:
    python plot_vkitti_frontier.py --output outputs/frontier.pdf
"""
import argparse
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

LOG_DIR = "logs/vkitti_eval"


def extract(path, pattern):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        text = f.read()
    m = re.findall(pattern, text)
    return float(m[-1]) if m else None


def load(v):
    base = os.path.join(LOG_DIR, v)
    return {
        "fid":  extract(f"{base}_fid.log",  r"FID:\s+([\d.]+)"),
        "miou": extract(f"{base}_miou.log", r"mIoU: ([\d.]+)"),
    }


# --- Variants to plot, grouped by category ---
GROUPS = {
    "Raw sim": {
        "marker": "D", "color": "#888888", "zorder": 2,
        "variants": {"input": "Input (raw sim)"},
    },
    "Diffusion editing": {
        "marker": "s", "color": "#e07b39", "zorder": 3,
        "variants": {
            "flowedit": "FlowEdit",
            "dnaedit":  "DNAEdit",
            "kontext":  "FLUX-Kontext",
        },
    },
    "Conditioned diffusion": {
        "marker": "^", "color": "#9467bd", "zorder": 3,
        "variants": {
            "cosmos_depth_edge_imgs": "Cosmos depth+edge",
            "cosmos_depth_seg_vis_edge_imgs": "Cosmos (best)",
        },
    },
    "WPD baseline (no drop_ll)": {
        "marker": "o", "color": "#2ca02c", "zorder": 4,
        "variants": {
            "baseline_r8":       "r8",
            "baseline_r12":      "r12",
            "baseline_newprompt":"r16",
            "baseline_r20":      "r20",
            "baseline_r24":      "r24",
        },
    },
    "WPD drop_ll J=4": {
        "marker": "o", "color": "#1f77b4", "zorder": 5,
        "variants": {
            "dropll_J4_r8":  "r8",
            "dropll_J4_r12": "r12",
            "dropll_step6000_J4": "r16",
            "dropll_J4_r20": "r20",
            "dropll_J4_r24": "r24",
        },
    },
}

ANNOTATE = {
    "input":              "offset",
    "dropll_J4_r8":       "best FID",
    "dropll_J4_r12":      "best trade-off",
    "baseline_r24":       "best mIoU",
    "cosmos_depth_edge_imgs": "offset",
}


def pareto_frontier(points):
    """Return indices of Pareto-optimal points (min FID, max mIoU)."""
    pts = np.array(points)
    n = len(pts)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if pts[j, 0] <= pts[i, 0] and pts[j, 1] >= pts[i, 1]:
                if pts[j, 0] < pts[i, 0] or pts[j, 1] > pts[i, 1]:
                    dominated[i] = True
                    break
    frontier = pts[~dominated]
    return frontier[np.argsort(frontier[:, 0])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/vkitti_frontier.pdf")
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(7, 5))

    all_points = []
    for group_name, group in GROUPS.items():
        fids, mious, labels = [], [], []
        for vkey, vlabel in group["variants"].items():
            m = load(vkey)
            if m["fid"] is None or m["miou"] is None:
                continue
            fids.append(m["fid"])
            mious.append(m["miou"])
            labels.append(vlabel)
            all_points.append((m["fid"], m["miou"], vkey))

        if not fids:
            continue

        ax.scatter(fids, mious,
                   marker=group["marker"],
                   color=group["color"],
                   s=60, zorder=group["zorder"],
                   label=group_name, edgecolors="white", linewidths=0.5)

        # Label radius sweep groups
        if "WPD" in group_name and len(fids) > 1:
            for x, y, lbl in zip(fids, mious, labels):
                ax.annotate(lbl, (x, y), textcoords="offset points",
                            xytext=(4, 4), fontsize=6.5,
                            color=group["color"], zorder=6)

    # Pareto frontier
    pts = [(f, m) for f, m, _ in all_points if f is not None and m is not None]
    if pts:
        frontier = pareto_frontier(pts)
        ax.plot(frontier[:, 0], frontier[:, 1], "k--", lw=1, alpha=0.4,
                zorder=1, label="Pareto frontier")

    # Special annotations
    for vkey, ann in ANNOTATE.items():
        m = load(vkey)
        if m["fid"] is None or m["miou"] is None:
            continue
        if ann == "offset":
            continue
        ax.annotate(ann, (m["fid"], m["miou"]),
                    textcoords="offset points", xytext=(-6, 8),
                    fontsize=8, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.8),
                    color="black", zorder=7,
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")])

    ax.set_xlabel("FID↓ (distributional realism)", fontsize=11)
    ax.set_ylabel("mIoU↑ (semantic preservation)", fontsize=11)
    ax.set_title("vKITTI → KITTI: Realism vs Structure Trade-off", fontsize=12)
    ax.legend(fontsize=8, loc="lower left", framealpha=0.9)
    ax.invert_xaxis()
    ax.grid(True, alpha=0.25, linestyle="--")

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
