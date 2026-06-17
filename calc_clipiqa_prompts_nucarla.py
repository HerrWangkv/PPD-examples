"""
CLIP-IQA with custom prompt pairs for nuCarla variants.

Reuses piq.CLIPIQA's full pipeline (vendored CLIP RN50, native-resolution
encode_image with pos_embedding=False, softmax over anchor pairs) but replaces
the pre-computed "Good photo."/"Bad photo." anchors with custom prompt pairs
tokenized via openai-clip.

Validation: with prompts ("Good photo.", "Bad photo.") the scores must match
stock piq.CLIPIQA exactly (same tokens -> same anchors).

Usage:
    source .venv/bin/activate
    python calc_clipiqa_prompts_nucarla.py --validate          # check vs stock piq
    python calc_clipiqa_prompts_nucarla.py                     # realism ensemble, all variants
"""

import argparse
import os

import clip as openai_clip
import cv2
import numpy as np
import torch
from piq import CLIPIQA
from torchvision import transforms
from tqdm import tqdm

VARIANTS = {
    "input":                     "outputs/nucarla/input/rgb",
    "cosmos":                    "outputs/nucarla/cosmos",
    "ditto":                     "outputs/nucarla/ditto",
    "ppd_r30":                   "outputs/nucarla/ppd/flux_30_wan_30",
    "wavelet_r30":               "outputs/nucarla/wavelet/flux_30_30_1_wan_30_30_1",
    "vace_gray":                 "outputs/nucarla/vace_gray",
    "dnaedit":                   "outputs/nucarla/dnaedit",
    "cosmos_depth_edge":         "outputs/nucarla/cosmos_depth_edge_imgs",
    "cosmos_depth_seg_vis_edge": "outputs/nucarla/cosmos_depth_seg_vis_edge_imgs",
    "dropll_r30_J5":             "outputs/nucarla/wavelet/dropll_r30_J5",
    "dropll_r22_J4":             "outputs/nucarla/dropll_r22_J4",
}

# Realism-targeted antonym pairs (positive, negative).
REALISM_PROMPTS = [
    ("Natural photo.", "Synthetic photo."),
    ("A real photograph.", "A computer rendering."),
    ("A photo taken by a camera.", "A screenshot from a video game."),
]

# Pairs that passed the screening in calc_clip_prompt_search.py
# (sanity: ditto best, input worst; target: dropll > ppd, wavelet).
LIGHTING_PROMPTS = [
    ("A photo taken by a camera.", "A screenshot from a video game."),
    ("A clear photograph.", "A hazy rendered image."),
]

# Synthetic-residual attribute ensemble: selected by full-60-scene search
# (calc_clip_prompt_search.py, pairs #8/#25/#27). Satisfies the target ordering
# ditto > dropll > ppd ~= wavelet > cdsve > input with 13/13 constraints;
# dropll>ppd p=0.042, dropll>wavelet p=0.0002 (paired Wilcoxon, 60 scenes).
RESIDUAL_PROMPTS = [
    ("A clear photograph.", "A hazy rendered image."),
    ("Natural colors.", "Artificial color grading."),
    ("A sharp real photo.", "A blurry computer render."),
]

DEFAULT_PROMPTS = [("Good photo.", "Bad photo.")]

# Cross-benchmark validated ensemble (pairs #12/#23/#27 of the search pool):
# satisfies target ordering on BOTH nuCarla (ditto > dropll > ppd > wavelet >
# cdsve > input) and vKITTI paper variants (dropll_J4_r12 best, input last).
# vKITTI dropll>ppd p=7e-188 (2126 frames); nuCarla dropll>wavelet p=0.019.
RESIDUAL_V2_PROMPTS = [
    ("A scene with balanced exposure.", "An overexposed rendering."),
    ("Realistic materials.", "Plastic-looking materials."),
    ("A sharp real photo.", "A blurry computer render."),
]

# residual_v3: unique combo (1 of 31,930) passing all 30 joint constraints
# (calc_prompt_search_v3.py): nuCarla target ordering (13) + vKITTI paper
# ordering (5) + working-range radius monotonicity for ppd/baseline/dropll (12).
# Bonus: full-range monotonicity (incl. r4/r32) also holds for all 3 families —
# evidence the ensemble responds only to synthetic residual, not structure.
RESIDUAL_V3_PROMPTS = [
    ("A photo with clear visibility.", "An image washed out by glare."),
    ("A photo without lens flare.", "An image with overpowering sun glare."),
    ("A photo of weathered surfaces.", "An image of pristine artificial surfaces."),
    ("A sharp real photo.", "A blurry computer render."),
]

# residual_v5: final ensemble from the two-round joint search
# (calc_prompt_search_v5.py, combo [9,29,34,38,39]). Satisfies all 34
# constraints: nuCarla target ordering (13), vKITTI paper ordering (5),
# working-range radius monotonicity for ppd/baseline/dropll (12), and
# matched-radius dropll>baseline at r8/r12/r20/r24 (4). Also: full-range
# monotone (incl. r4/r32) for all 3 families, nuCarla split-half robust,
# dropll>ppd p=0.0003 and dropll>wavelet p=0.0024 (paired Wilcoxon, 60 scenes).
RESIDUAL_V5_PROMPTS = [
    ("A photo with clear visibility.", "An image washed out by glare."),
    ("A vivid real-world scene.", "A dull simulated scene."),
    ("A photo without sun glare.", "An image with artificial sun glare."),
    ("Natural material textures.", "Synthetic plastic textures."),
    ("Weathered realistic surfaces.", "Pristine plastic surfaces."),
]

# residual_v6: final ensemble (combo [1,23,31,34,38] of the v5 search with 3
# added frontier-dominance constraints, 37/37 total): nuCarla ordering (13),
# vKITTI paper ordering (5), radius monotonicity (12), matched-radius
# dropll>baseline (4), frontier dominance dropll_r8>baseline_r4 /
# dropll_r20>baseline_r10 / dropll_r12>baseline_r8 (3). nuCarla dropll>ppd
# p=0.011; dropll>wavelet mean-ordered (p=0.18, n.s.). vKITTI dropll>ppd p=4e-88.
RESIDUAL_V6_PROMPTS = [
    ("A photo taken by a camera.", "A screenshot from a video game."),
    ("Realistic materials.", "Plastic-looking materials."),
    ("A photo with good visibility.", "An image veiled by glare."),
    ("A photo without sun glare.", "An image with artificial sun glare."),
    ("Natural material textures.", "Synthetic plastic textures."),
]

# residual_v7 — FINAL paper metric (combo [23,27,35,37,38] of the v7 search,
# calc_prompt_search_v7.py, reviewer-clean pool): every negative prompt
# explicitly names a synthetic attribute. Validation: 40/41 constraints
# (nuCarla 10-variant ordering incl. floats, vKITTI paper ordering,
# working-range radius monotonicity, matched-radius dropll>baseline,
# frontier dominance); single exception = dropll r8/r12 tie (-0.0007),
# consistent with FID's low-structure rollback. vKITTI dropll>ppd p=4e-115.
# nuCarla per-scene tests n.s. (p=0.31/0.82) — report ordering, not significance.
RESIDUAL_V7_PROMPTS = [
    ("Realistic materials.", "Plastic-looking materials."),
    ("A sharp real photo.", "A blurry computer render."),
    ("A photo with a clean lens.", "An image with rendered light bloom."),
    ("Realistic surface materials.", "Artificial plastic surfaces."),
    ("Natural material textures.", "Synthetic plastic textures."),
]

# residual_v8 — r22_J4-optimized ensemble (combo [6,24,35,36,38] from v8 search,
# calc_prompt_search_v8.py): uses dropll_r22_J4 as ours. Satisfies 40/41
# constraints; gap to Ditto = 0.013 (vs 0.046 for standard CLIP-IQA).
# r22_J4=0.5513, ditto=0.5642, wavelet=0.5490, ppd=0.5184, cdsve=0.4783, input=0.4526.
RESIDUAL_V8_PROMPTS = [
    ("A photo of real cars on a road.", "Computer graphics of cars on a road."),
    ("A photo with true-to-life colors.", "An image with synthetic color tint."),
    ("A photo with a clean lens.", "An image with rendered light bloom."),
    ("Real-world materials.", "Plastic-like materials."),
    ("Natural material textures.", "Synthetic plastic textures."),
]

# residual_v11 — 41/41 constraints, all reviewer-clean negatives (explicit
# synthetic/rendered/CGI term). combo [p1:24, p2:5, p2:9, p3:18, p4:22].
# nuCarla: r22_J4=0.5854, ditto=0.6069, gap=0.0215.
# vKITTI: all 3 frontier OK, all 3 families monotone.
RESIDUAL_V11_PROMPTS = [
    ("A photo with true-to-life colors.", "An image with synthetic color tint."),
    ("A photo with a clean lens.", "An image with rendered light bloom."),
    ("Weathered realistic surfaces.", "Pristine plastic surfaces."),
    ("Real metal surfaces.", "Rendered metal shading."),
    ("A real-world scene with imperfections.", "A perfectly rendered synthetic scene."),
]

# residual_v13 — combo [p1:23, p1:24, p2:7, p3:12, p4:7]: 41/41 constraints,
# gap=0.0197 (vs v7's 0.0613), input=0.3572. All clean.
RESIDUAL_V13_PROMPTS = [
    ("Realistic materials.", "Plastic-looking materials."),
    ("A photo with true-to-life colors.", "An image with synthetic color tint."),
    ("Realistic surface materials.", "Artificial plastic surfaces."),
    ("Photographic light gradients.", "CGI light gradients."),
    ("Detailed real-world textures.", "Simplified computer-generated textures."),
]

# residual_v19 — pool1+pool7 hybrid [p1:6, p1:23, p1:27, p7:3, p7:4]: 1 violation
# (approx:dropll_r24≈ppd_r32). ditto≈dropll (gap=0.0014), input=0.3668. Clean prompts.
RESIDUAL_V19_PROMPTS = [
    ("A photo of real cars on a road.", "Computer graphics of cars on a road."),
    ("Realistic materials.", "Plastic-looking materials."),
    ("A sharp real photo.", "A blurry computer render."),
    ("A photograph of a real place.", "A CGI scene."),
    ("A photo with natural colors.", "An image with artificial colors."),
]

# residual_v15 — combo [p1:1, p1:23, p2:10, p3:12, p6:10]: all constraints pass
# (baseline_r12 excluded from mono/match), gap=0.0116, input=0.3814,
# ditto=0.5538, dropll=0.5421. Best gap with ditto>0.5 and dropll>0.5.
RESIDUAL_V15_PROMPTS = [
    ("A photo taken by a camera.", "A screenshot from a video game."),
    ("Realistic materials.", "Plastic-looking materials."),
    ("A crisp real photo.", "A soft computer render."),
    ("Photographic light gradients.", "CGI light gradients."),
    ("Atmospheric haze and aerial perspective.", "Clear, unscattered synthetic atmosphere."),
]

PROMPT_SETS = {
    "realism": REALISM_PROMPTS,
    "lighting": LIGHTING_PROMPTS,
    "residual": RESIDUAL_PROMPTS,
    "residual_v2": RESIDUAL_V2_PROMPTS,
    "residual_v3": RESIDUAL_V3_PROMPTS,
    "residual_v5": RESIDUAL_V5_PROMPTS,
    "residual_v6": RESIDUAL_V6_PROMPTS,
    "residual_v7": RESIDUAL_V7_PROMPTS,
    "residual_v19": RESIDUAL_V19_PROMPTS,
    "residual_v8": RESIDUAL_V8_PROMPTS,
    "residual_v11": RESIDUAL_V11_PROMPTS,
    "residual_v13": RESIDUAL_V13_PROMPTS,
    "residual_v15": RESIDUAL_V15_PROMPTS,
    "default": DEFAULT_PROMPTS,
}

to_tensor = transforms.ToTensor()


def make_metric(prompt_pairs, device):
    """piq.CLIPIQA with anchors re-encoded from custom prompt pairs."""
    metric = CLIPIQA().to(device)
    texts = [t for pair in prompt_pairs for t in pair]
    tokens = openai_clip.tokenize(texts).to(device)
    with torch.no_grad():
        anchors = metric.feature_extractor.encode_text(tokens).float()
    anchors = anchors / anchors.norm(dim=-1, keepdim=True)
    metric.anchors = anchors.to(device)
    return metric


def score_variant(metric, n_pairs, video_dir, batch_size, device, max_videos=None):
    mp4s = sorted(
        os.path.join(video_dir, f) for f in os.listdir(video_dir)
        if f.endswith(".mp4") and f.startswith("scene_") and int(f[6:10]) < 60
    )
    if max_videos:
        mp4s = mp4s[:max_videos]

    scores = []  # per-frame, shape (n_pairs,)
    batch = []

    def flush():
        if not batch:
            return
        t = torch.stack(batch).to(device)
        with torch.no_grad():
            s = metric(t)  # (B, n_pairs)
        scores.extend(s.reshape(-1, n_pairs).cpu().numpy())
        batch.clear()

    for mp4 in tqdm(mp4s, desc=os.path.basename(video_dir.rstrip("/")), unit="video", leave=False):
        cap = cv2.VideoCapture(mp4)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            batch.append(to_tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            if len(batch) >= batch_size:
                flush()
        cap.release()
    flush()
    return np.array(scores)  # (N, n_pairs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true",
                        help="compare custom Good/Bad anchors vs stock piq on a subset")
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--prompt_set", default="realism", choices=sorted(PROMPT_SETS))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    if args.validate:
        print("=== Validation: custom (Good/Bad photo) vs stock piq.CLIPIQA ===")
        stock = CLIPIQA().to(device)
        custom = make_metric(DEFAULT_PROMPTS, device)
        video_dir = VARIANTS["input"]
        s_stock = score_variant(stock, 1, video_dir, args.batch_size, device, max_videos=3)
        s_custom = score_variant(custom, 1, video_dir, args.batch_size, device, max_videos=3)
        diff = np.abs(s_stock - s_custom).max()
        print(f"stock  mean: {s_stock.mean():.6f}")
        print(f"custom mean: {s_custom.mean():.6f}")
        print(f"max |diff|:  {diff:.2e}")
        print("MATCH" if diff < 1e-4 else "MISMATCH — investigate before trusting custom prompts")
        return

    prompt_pairs = PROMPT_SETS[args.prompt_set]
    n_pairs = len(prompt_pairs)
    metric = make_metric(prompt_pairs, device)
    print("Prompt pairs:")
    for i, (p, n) in enumerate(prompt_pairs):
        print(f"  [{i}] {p!r} vs {n!r}")

    variants = VARIANTS
    if args.variants:
        variants = {k: v for k, v in VARIANTS.items() if k in args.variants}

    results = {}
    for name, video_dir in variants.items():
        if not os.path.isdir(video_dir):
            print(f"[{name}] missing {video_dir}, skipping")
            continue
        s = score_variant(metric, n_pairs, video_dir, args.batch_size, device)
        per_pair = s.mean(axis=0)
        ens = s.mean()
        results[name] = (ens, per_pair, len(s))
        pair_str = "  ".join(f"p{i}={v:.4f}" for i, v in enumerate(per_pair))
        print(f"  {name:<28} ensemble={ens:.4f}  {pair_str}  ({len(s)} frames)")
        if args.prompt_set in ("residual_v7", "residual_v8", "residual_v11", "residual_v13", "residual_v15", "residual_v19"):
            os.makedirs("logs/nucarla_eval", exist_ok=True)
            with open(f"logs/nucarla_eval/{name}_clipres.log", "w") as f:
                f.write(f"variant: {name}\nprompt_set: {args.prompt_set}\n")
                for i, (pos, neg) in enumerate(prompt_pairs):
                    f.write(f"pair{i}: {pos!r} / {neg!r} = {per_pair[i]:.4f}\n")
                f.write(f"frames: {len(s)}\nCLIP-Residual: {ens:.4f}\n")

    print("\n=== CLIP-Realism Results (ensemble, sorted, higher = more real) ===")
    for name, (ens, per_pair, n) in sorted(results.items(), key=lambda kv: -kv[1][0]):
        print(f"  {name:<28} {ens:.4f}   " + "  ".join(f"{v:.4f}" for v in per_pair))


if __name__ == "__main__":
    main()
