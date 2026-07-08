"""
No-reference IQA metrics (pyiqa) for nuCarla variants: NIQE / MUSIQ / TOPIQ-NR.

Statistical / learned naturalness, no reference set — fully avoids the
nuScenes sensor-anchoring problem. NIQE: lower = better. MUSIQ/TOPIQ: higher = better.

Usage:
    CUDA_VISIBLE_DEVICES=2 conda run -n sim2real_eval python calc_nr_iqa_nucarla.py
"""

import argparse
import glob
import os

import cv2
import numpy as np
import pyiqa
import torch
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

METRICS = ["niqe", "musiq", "topiq_nr"]  # niqe: lower better; others: higher better
FRAME_STRIDE = 6  # 49 frames -> ~8 frames per video

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def video_frames(mp4, stride):
    cap = cv2.VideoCapture(mp4)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        idx += 1
    cap.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=None)
    args = parser.parse_args()

    metrics = {m: pyiqa.create_metric(m, device=DEVICE) for m in METRICS}
    print(f"Device: {DEVICE}, metrics: {METRICS}, stride={FRAME_STRIDE}\n")

    variants = VARIANTS
    if args.variants:
        variants = {k: v for k, v in VARIANTS.items() if k in args.variants}

    results = {}
    for name, video_dir in variants.items():
        mp4s = sorted(
            f for f in glob.glob(os.path.join(video_dir, "scene_*.mp4"))
            if int(os.path.basename(f)[6:10]) < 60
        )
        scores = {m: [] for m in METRICS}
        for mp4 in tqdm(mp4s, desc=name, unit="video", leave=False):
            for frame in video_frames(mp4, FRAME_STRIDE):
                t = torch.from_numpy(frame).permute(2, 0, 1).float().div(255).unsqueeze(0).to(DEVICE)
                for m, fn in metrics.items():
                    with torch.no_grad():
                        scores[m].append(fn(t).item())
        results[name] = {m: (np.mean(v), len(v)) for m, v in scores.items()}
        line = "  ".join(f"{m}={results[name][m][0]:.4f}" for m in METRICS)
        n = results[name][METRICS[0]][1]
        print(f"  {name:<28} {line}  ({n} frames)")

    for m in METRICS:
        rev = m == "niqe"  # lower better
        print(f"\n=== {m} ({'lower' if rev else 'higher'} = better) ===")
        for name, r in sorted(results.items(), key=lambda kv: kv[1][m][0], reverse=not rev):
            print(f"  {name:<28} {r[m][0]:.4f}")


if __name__ == "__main__":
    main()
