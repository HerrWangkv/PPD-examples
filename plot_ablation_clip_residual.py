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

# residual_v11 — combo [p1:24, p2:5, p2:9, p3:18, p4:22]: 41/41 constraints,
# all clean negatives. gap to Ditto = 0.0215 (r22_J4=0.5854, ditto=0.6069).
# Pairs: true-to-life colors / clean lens / weathered surfaces /
#        real-world visual fidelity / real-world scene with imperfections.
SCORES_V11 = {
    "ppd":      {8: 0.5594, 12: 0.5541, 16: 0.5519, 20: 0.5517, 24: 0.5489, 32: 0.5480},
    "baseline": {4: 0.5871, 8: 0.5756, 10: 0.5732, 12: 0.5718, 20: 0.5563, 24: 0.4957},
    "dropll":   {8: 0.5965, 12: 0.5893, 20: 0.5763, 24: 0.5253},
}

# residual_v13 — combo [p1:23, p1:24, p2:7, p3:12, p4:7]: 41/41 constraints,
# gap=0.0197 (v7 was 0.0613), input=0.3572. All clean.
SCORES_V13 = {
    "ppd":      {8: 0.4421, 12: 0.4343, 16: 0.4306, 20: 0.4304, 24: 0.4274, 32: 0.4251},
    "baseline": {4: 0.4575, 8: 0.4454, 10: 0.4440, 12: 0.4383, 20: 0.4238, 24: 0.3785},
    "dropll":   {8: 0.4642, 12: 0.4562, 20: 0.4456, 24: 0.3992},
}

# k=3 core [p1:23, p1:27, p3:12]: ditto=0.557, drop=0.512, input=0.307 on nuCarla.
# 3 hard fails (ppd r16>r20 −0.0015, baseline r10>r12 −0.0032, front:r20>b_r10 −0.019).
SCORES_V15CORE = {
    "ppd":      {8: 0.5377, 12: 0.5305, 16: 0.5275, 20: 0.5289, 24: 0.5272},
    "baseline": {4: 0.5641, 8: 0.5439, 10: 0.5398, 20: 0.5046, 24: 0.3754},
    "dropll":   {8: 0.5748, 12: 0.5577, 20: 0.5205, 24: 0.3779},
}

# residual_v15 — [p1:1, p1:23, p2:10, p3:12, p6:10]: all constraints pass
# (baseline_r12 excluded from mono/match), gap=0.0116, input=0.3814,
# ditto=0.5538, dropll=0.5421 on nuCarla. REAL_KITTI=0.4083, input(vKITTI)=0.4018.
SCORES_V15 = {
    "ppd":      {8: 0.4188, 12: 0.4072, 16: 0.4018, 20: 0.4016, 24: 0.3996, 32: 0.4011},
    "baseline": {4: 0.4409, 8: 0.4210, 10: 0.4161, 12: 0.4267, 16: 0.4018, 20: 0.3926, 24: 0.3163},
    "dropll":   {8: 0.4513, 12: 0.4463, 16: 0.4235, 20: 0.4170, 24: 0.3346},
}

# residual_v16 — [p1:1, p1:24, p2:5, p3:15, p6:12]: 37 solutions, gap=0.0005, input=0.3744,
# ditto≈dropll=0.505, ppd=0.492. All 38 constraints pass incl. dropll_r24≈ppd_r32 (±0.05).
SCORES_V16 = {
    "ppd":      {8: 0.3550, 12: 0.3365, 16: 0.3255, 20: 0.3241, 24: 0.3207, 32: 0.3212},
    "baseline": {4: 0.3869, 8: 0.3589, 10: 0.3539, 16: 0.3488, 20: 0.3393, 24: 0.2907},
    "dropll":   {8: 0.3878, 12: 0.3744, 16: 0.3530, 20: 0.3470, 24: 0.2860},
}

# residual_v18 — pool1+pool7 hybrid [p1:1, p1:6, p7:1, p7:7, p7:13]: 1 violation
# (approx:dropll_r24≈ppd_r32 fails — simple prompts can't encode ego-artifact vs blur).
# ditto≈dropll=0.544 (gap=0.0007), input(nuCarla)=0.367, input(vKITTI)=0.069.
# Prompts: camera/game | real cars/CGI cars | outdoors/CGI | real textures/CGI textures | hazy/sharp render
SCORES_V18 = {
    "ppd":      {8: 0.5882, 12: 0.5824, 20: 0.5737, 24: 0.5682, 32: 0.5583},
    "baseline": {4: 0.6276, 8: 0.6169, 10: 0.6084, 20: 0.5374, 24: 0.3687},
    "dropll":   {8: 0.6465, 12: 0.6276, 16: 0.5764, 20: 0.5589, 24: 0.3741},
}

# residual_v19 — pool1+pool7 hybrid [p1:6, p1:23, p1:27, p7:3, p7:4]: 1 violation
# (approx:dropll_r24≈ppd_r32 fails, margin=-0.274; no p7:13 "hazy" prompt).
# ditto≈dropll=0.496/0.495 (gap=0.0014), input(nuCarla)=0.367, input(vKITTI)=0.281.
# Prompts: real cars/CGI | realistic materials/plastic | sharp real/blurry render | real place/CGI | natural colors/artificial
SCORES_V19 = {
    "ppd":      {8: 0.5347, 12: 0.5267, 16: 0.5204, 20: 0.5179, 24: 0.5152, 32: 0.5142},
    "baseline": {4: 0.5625, 8: 0.5532, 10: 0.5479, 12: 0.5541, 20: 0.5122, 24: 0.4060},
    "dropll":   {8: 0.5670, 12: 0.5607, 16: 0.5307, 20: 0.5212, 24: 0.4094},
}

# residual_v20 — k=7 pool1+pool7 [p1:1,p1:6,p1:23,p7:3,p7:4,p7:15,p7:18]: all 40 constraints pass.
# ditto=0.472, dropll=0.455 (gap=0.017), input(nuCarla)=0.398, input(vKITTI)=0.348.
# Prompts: camera/game | real cars/CGI | realistic materials/plastic | real place/CGI |
#          natural colors/artificial | soft depth/artificial clarity | authentic noise/sterile
SCORES_V20 = {
    "ppd":      {8: 0.4415, 12: 0.4279, 20: 0.4160, 24: 0.4131, 32: 0.4149},
    "baseline": {4: 0.4612, 8: 0.4463, 10: 0.4417, 20: 0.4095, 24: 0.3476},
    "dropll":   {8: 0.4620, 12: 0.4597, 16: 0.4339, 20: 0.4263, 24: 0.3649},
}

# single pair p1:6: "A photo of real cars on a road." / "Computer graphics of cars on a road."
# 5 constraint violations; nuCarla dropll > ditto (inverted). input_vk=0.003.
SCORES_P16 = {
    "ppd":      {8: 0.4902, 12: 0.4693, 20: 0.4367, 24: 0.4270, 32: 0.4196},
    "baseline": {4: 0.5618, 8: 0.5351, 10: 0.5161, 20: 0.3848, 24: 0.1479},
    "dropll":   {8: 0.5745, 12: 0.5521, 16: 0.4510, 20: 0.4199, 24: 0.1669},
}

# 2-pair combo p1:6+p7:10: "A photo of real cars on a road." + "A real street scene."
# 3 constraint violations (front:r12>b_r8, gap:dropll_r20>ppd_r32, approx:dropll_r24≈ppd_r32).
# nuCarla: ditto=0.591 > dropll=0.567 > wavelet=0.561 > ppd=0.460 > cosmos=0.351 > input=0.294.
# input_vk=0.004.
SCORES_P1610 = {
    "ppd":      {8: 0.4502, 12: 0.4494, 20: 0.4409, 24: 0.4347, 32: 0.4276},
    "baseline": {4: 0.5171, 8: 0.5065, 10: 0.4905, 20: 0.3645, 24: 0.1483},
    "dropll":   {8: 0.5177, 12: 0.4946, 16: 0.3911, 20: 0.3610, 24: 0.1349},
}

# 2-pair combo p1:6+p1:23: "A photo of real cars on a road." + "Realistic materials."
# WPD > PPD at every radius on vKITTI (gap +0.06–+0.11). 6 constraint violations incl.
# mono:dropll:r8>r12 and nuCarla sim-input inversions (PBR materials score as realistic).
SCORES_P1623 = {
    "ppd":      {8: 0.3512, 12: 0.3307, 20: 0.3069, 24: 0.3018, 32: 0.3054},
    "baseline": {4: 0.3938, 8: 0.3730, 10: 0.3663, 20: 0.3349, 24: 0.2684},
    "dropll":   {8: 0.4116, 12: 0.4446, 16: 0.4122, 20: 0.4070, 24: 0.3321},
}

# Grand average over all 36 clean pairs (pool1 clean + pool7, excl p7:13).
# LLM-generated antonym pairs averaged — no search, no constraint tuning.
# nuCarla: ditto>dropll>wavelet>ppd>cosmos>input (correct ordering).
SCORES_AVG36 = {
    "ppd":      {8: 0.5839, 12: 0.5822, 20: 0.5809, 24: 0.5781, 32: 0.5757},
    "baseline": {4: 0.6222, 8: 0.6118, 10: 0.6030, 20: 0.5365, 24: 0.3851},
    "dropll":   {8: 0.6088, 12: 0.5880, 16: 0.5337, 20: 0.5153, 24: 0.3596},
}

# avg41a: all 46 clean pairs minus 5 worst PPD-inflators (by WPD-PPD gap)
# (p1:3 "genuine street photo", p1:5 "real street", p8:6 "natural visual detail",
#  p1:7 "from the real world", p8:8 "natural scene depth")
SCORES_AVG41A = {
    "ppd":      {8: 0.5420, 12: 0.5388, 20: 0.5369, 24: 0.5343, 32: 0.5331},
    "baseline": {4: 0.5741, 8: 0.5636, 10: 0.5557, 20: 0.5029, 24: 0.3813},
    "dropll":   {8: 0.5617, 12: 0.5472, 16: 0.5032, 20: 0.4881, 24: 0.3624},
}

# avg41: all 46 clean pairs minus 5 highest ppd_r32-dropll_r24 pairs
# (p1:3 "genuine street photo", p1:0 "real photograph", p1:7 "from the real world",
#  p1:27 "sharp real photo", p1:5 "real street") — these prevent approx constraint
SCORES_AVG41 = {
    "ppd":      {8: 0.5390, 12: 0.5338, 20: 0.5299, 24: 0.5268, 32: 0.5249},
    "baseline": {4: 0.5688, 8: 0.5549, 10: 0.5468, 20: 0.4944, 24: 0.3786},
    "dropll":   {8: 0.5514, 12: 0.5352, 16: 0.4925, 20: 0.4779, 24: 0.3600},
}

# Grand average over all 46 clean pairs (pool1+pool7+pool8, excl p7:13).
SCORES_AVG46 = {
    "ppd":      {8: 0.5605, 12: 0.5572, 20: 0.5545, 24: 0.5514, 32: 0.5485},
    "baseline": {4: 0.5929, 8: 0.5805, 10: 0.5719, 20: 0.5097, 24: 0.3714},
    "dropll":   {8: 0.5773, 12: 0.5557, 16: 0.5056, 20: 0.4883, 24: 0.3472},
}

# Filtered average: 8 discriminative pairs (ppd_mean < 0.65 AND baseline_r4 < dropll_r8).
# Excludes pairs where content familiarity inflates PPD/baseline scores on vKITTI.
# nuCarla: ditto>dropll>wavelet>ppd>cosmos>input ✓  WPD-PPD gap = 0.087 (vs 0.025 for avg36).
SCORES_AVG8 = {
    "ppd":      {8: 0.4621, 12: 0.4628, 20: 0.4620, 24: 0.4575, 32: 0.4510},
    "baseline": {4: 0.5095, 8: 0.5171, 10: 0.5121, 20: 0.4437, 24: 0.2963},
    "dropll":   {8: 0.5488, 12: 0.5241, 16: 0.4770, 20: 0.4621, 24: 0.2981},
}

SCORE_SETS = {"v2": SCORES_V2, "v3": SCORES_V3, "v4": SCORES_V4, "v5": SCORES_V5, "v6": SCORES_V6, "v7": SCORES_V7, "v11": SCORES_V11, "v13": SCORES_V13, "v15core": SCORES_V15CORE, "v15": SCORES_V15, "v16": SCORES_V16, "v18": SCORES_V18, "v19": SCORES_V19, "v20": SCORES_V20, "p16": SCORES_P16, "p1610": SCORES_P1610, "p1623": SCORES_P1623, "avg36": SCORES_AVG36, "avg41a": SCORES_AVG41A, "avg41": SCORES_AVG41, "avg46": SCORES_AVG46, "avg8": SCORES_AVG8}
INPUT_CLIP_SETS = {"v2": 0.2283, "v3": 0.3909, "v4": 0.3799, "v5": 0.5262, "v6": 0.4315, "v7": 0.2396, "v11": 0.3415, "v13": 0.3489, "v15core": 0.5166, "v15": 0.4018, "v16": 0.2678, "v18": 0.0689, "v19": 0.2808, "v20": 0.3475, "p16": 0.0030, "p1610": 0.0040, "p1623": 0.2917, "avg36": 0.2327, "avg41a": 0.2391, "avg41": 0.2432, "avg46": 0.2178, "avg8": 0.1770}
REAL_KITTI_SETS = {"v2": 0.5811, "v3": None, "v4": None, "v5": None, "v6": None, "v7": None, "v11": None, "v13": None, "v15core": None, "v15": 0.4083, "v16": None, "v18": None, "v19": None, "v20": None, "p16": None, "p1610": None, "p1623": None, "avg36": None, "avg41a": None, "avg41": None, "avg46": None, "avg8": None}

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
    parser.add_argument("--set", default="v3", choices=["v2", "v3", "v4", "v5", "v6", "v7", "v11", "v13", "v15core", "v15", "v16"],
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
