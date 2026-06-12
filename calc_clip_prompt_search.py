"""
Full-set CLIP prompt-pair search for the target realism ordering.

Target: ditto > dropll_r30_J5 > wavelet_r30 ~= ppd_r30 > cosmos_dsve ~= input
(13 strict pairwise constraints).

Differences from the earlier failed screening:
  - evaluated on ALL 60 scenes (no subset-flip problem)
  - per-scene means saved to npy -> split-half validation (select on scenes
    0-29, verify on 30-59) to guard against overfitting scene noise
  - large candidate pool in one pass (image features shared across prompts)

Usage:
    CUDA_VISIBLE_DEVICES=0 python calc_clip_prompt_search.py          # score
    python calc_clip_prompt_search.py --analyze                       # analysis only
"""

import argparse
import os

import cv2
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from calc_clipiqa_prompts_nucarla import VARIANTS, make_metric

CANDIDATES = [
    # realism / domain
    ("A real photograph.", "A computer rendering."),
    ("A photo taken by a camera.", "A screenshot from a video game."),
    ("A photo from a dashboard camera.", "A frame from a driving simulator."),
    ("A genuine street photograph.", "A 3D rendered street scene."),
    ("An authentic photo.", "A CGI image."),
    ("A photograph of a real street.", "A street in a video game."),
    ("A photo of real cars on a road.", "Computer graphics of cars on a road."),
    ("A photo from the real world.", "An image from a virtual world."),
    # glare / haze / exposure
    ("A clear photograph.", "A hazy rendered image."),
    ("A photo with clear visibility.", "An image washed out by glare."),
    ("A crisp outdoor photograph.", "A foggy computer render."),
    ("A photo without lens flare.", "An image with overpowering sun glare."),
    ("A scene with balanced exposure.", "An overexposed rendering."),
    ("A photo with natural contrast.", "A washed-out synthetic image."),
    ("A well-exposed photograph.", "An image flooded with artificial light."),
    # lighting
    ("A photo with realistic lighting.", "A rendering with fake lighting."),
    ("Natural daylight.", "Artificial rendered light."),
    ("A photo with natural shadows.", "An image with fake shadows."),
    ("A realistic sky.", "A rendered sky."),
    ("A photo with coherent lighting.", "An image with inconsistent lighting."),
    # texture / material
    ("A photo with realistic textures.", "An image with plastic-looking textures."),
    ("A photo with detailed road texture.", "An image with a smooth artificial road."),
    ("A photo of weathered surfaces.", "An image of pristine artificial surfaces."),
    ("Realistic materials.", "Plastic-looking materials."),
    # color
    ("A photo with true-to-life colors.", "An image with synthetic color tint."),
    ("Natural colors.", "Artificial color grading."),
    ("A photo with realistic color balance.", "An image with a yellow haze."),
    # sharp/blur quality-realism mixes
    ("A sharp real photo.", "A blurry computer render."),
    ("A detailed photograph.", "A flat lifeless rendering."),
    ("A vivid real-world scene.", "A dull simulated scene."),
]

KEY_VARIANTS = ["ditto", "dropll_r30_J5", "wavelet_r30", "ppd_r30",
                "cosmos_depth_seg_vis_edge", "input"]
SHORT = {"ditto": "ditto", "dropll_r30_J5": "dropll", "wavelet_r30": "wavelet",
         "ppd_r30": "ppd", "cosmos_depth_seg_vis_edge": "cdsve", "input": "input"}

# 13 strict constraints (a beats b)
CONSTRAINTS = (
    [("ditto", v) for v in ["dropll", "wavelet", "ppd", "cdsve", "input"]] +
    [("dropll", v) for v in ["wavelet", "ppd", "cdsve", "input"]] +
    [("wavelet", "cdsve"), ("wavelet", "input"), ("ppd", "cdsve"), ("ppd", "input")]
)

OUT_NPY = "clip_prompt_search_scenes.npz"
to_tensor = transforms.ToTensor()


def score_per_scene(metric, n_pairs, video_dir, batch_size, device):
    """Returns (n_scenes, n_pairs) per-scene mean scores."""
    mp4s = sorted(
        os.path.join(video_dir, f) for f in os.listdir(video_dir)
        if f.endswith(".mp4") and f.startswith("scene_") and int(f[6:10]) < 60
    )
    scene_means = []
    for mp4 in tqdm(mp4s, desc=os.path.basename(video_dir.rstrip("/")), unit="video", leave=False):
        cap = cv2.VideoCapture(mp4)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(to_tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()
        scores = []
        for i in range(0, len(frames), batch_size):
            t = torch.stack(frames[i:i + batch_size]).to(device)
            with torch.no_grad():
                s = metric(t)
            scores.append(s.reshape(-1, n_pairs).cpu().numpy())
        scene_means.append(np.concatenate(scores).mean(axis=0))
    return np.stack(scene_means)  # (60, n_pairs)


def analyze(data):
    n_pairs = len(CANDIDATES)
    half = {"select": slice(0, 30), "verify": slice(30, 60), "full": slice(None)}

    def check(i, sl):
        means = {SHORT[v]: data[v][sl, i].mean() for v in KEY_VARIANTS}
        passed = sum(means[a] > means[b] for a, b in CONSTRAINTS)
        return passed, means

    print(f"{'#':<3} {'pair':<58} {'sel':>4} {'ver':>4} {'full':>4}")
    full_pass = []
    for i, (pos, neg) in enumerate(CANDIDATES):
        p_sel, _ = check(i, half["select"])
        p_ver, _ = check(i, half["verify"])
        p_full, means = check(i, half["full"])
        label = f"{pos[:28]:<29}/ {neg[:26]}"
        flag = " <-- " if p_sel == 13 and p_ver == 13 else ""
        print(f"{i:<3} {label:<58} {p_sel:>3}  {p_ver:>3}  {p_full:>3}{flag}")
        if p_full == 13:
            full_pass.append(i)

    print(f"\nPairs passing all 13 constraints on FULL set: {full_pass}")
    robust = [i for i in full_pass
              if check(i, half["select"])[0] == 13 and check(i, half["verify"])[0] == 13]
    print(f"Pairs passing 13/13 on BOTH halves independently: {robust}")

    for group, idxs in [("full-pass ensemble", full_pass), ("robust ensemble", robust)]:
        if not idxs:
            continue
        print(f"\n=== {group} ({len(idxs)} pairs) ===")
        for v in KEY_VARIANTS:
            print(f"  {SHORT[v]:<8} {data[v][:, idxs].mean():.4f}")
        from scipy.stats import wilcoxon
        d = data["dropll_r30_J5"][:, idxs].mean(axis=1)
        for rival in ["ppd_r30", "wavelet_r30"]:
            r = data[rival][:, idxs].mean(axis=1)
            stat, p = wilcoxon(d - r)
            wins = (d > r).sum()
            print(f"  dropll vs {SHORT[rival]:<8} win {wins}/60 scenes, wilcoxon p={p:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    if args.analyze:
        loaded = np.load(OUT_NPY)
        analyze({k: loaded[k] for k in KEY_VARIANTS})
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}, {len(CANDIDATES)} pairs, {len(KEY_VARIANTS)} variants x 60 scenes")
    metric = make_metric(CANDIDATES, device)

    data = {}
    for v in KEY_VARIANTS:
        data[v] = score_per_scene(metric, len(CANDIDATES), VARIANTS[v], args.batch_size, device)
        print(f"  {v}: {data[v].shape}")
    np.savez(OUT_NPY, **data)
    print(f"Saved per-scene scores -> {OUT_NPY}\n")
    analyze(data)


if __name__ == "__main__":
    main()
