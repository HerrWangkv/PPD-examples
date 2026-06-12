"""
Compute CLIP-IQA for nuCarla translated variants using piq's standard implementation.

CLIP-IQA (Wang et al., AAAI 2023) measures perceptual image quality via antonymous
prompt pairs ("Good photo" / "Bad photo"). Higher = better quality.

Usage:
    source .venv/bin/activate
    python calc_clipiqa_nucarla.py
    python calc_clipiqa_nucarla.py --variants input cosmos_depth_edge_imgs cosmos_depth_seg_vis_edge_imgs
"""

import argparse
import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
from piq import CLIPIQA

VARIANTS = {
    "input":                          "outputs/nucarla/input/rgb",
    "cosmos":                         "outputs/nucarla/cosmos",
    "ditto":                          "outputs/nucarla/ditto",
    "ppd_r30":                        "outputs/nucarla/ppd/flux_30_wan_30",
    "wavelet_r30":                    "outputs/nucarla/wavelet/flux_30_30_1_wan_30_30_1",
    "vace_gray":                      "outputs/nucarla/vace_gray",
    "dnaedit":                        "outputs/nucarla/dnaedit",
    "cosmos_depth_edge_imgs":         "outputs/nucarla/cosmos_depth_edge_imgs",
    "cosmos_depth_seg_vis_edge_imgs": "outputs/nucarla/cosmos_depth_seg_vis_edge_imgs",
    "dropll_r30_J5":                  "outputs/nucarla/wavelet/dropll_r30_J5",
}

to_tensor = transforms.ToTensor()


def score_variant(name, video_dir, batch_size, device):
    mp4s = sorted([
        os.path.join(video_dir, f)
        for f in os.listdir(video_dir)
        if f.endswith(".mp4") and f.startswith("scene_")
        and int(f[6:10]) < 60
    ])
    if not mp4s:
        print(f"  [{name}] No MP4s found in {video_dir}, skipping.")
        return None

    clipiqa = CLIPIQA().to(device)
    scores = []
    batch = []

    def flush_batch():
        if not batch:
            return
        tensor = torch.stack(batch).to(device)
        with torch.no_grad():
            s = clipiqa(tensor).squeeze().cpu().numpy()
        scores.extend(np.atleast_1d(s).tolist())
        batch.clear()

    for mp4 in tqdm(mp4s, desc=name, unit="video"):
        cap = cv2.VideoCapture(mp4)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t = to_tensor(frame_rgb)
            batch.append(t)
            if len(batch) >= batch_size:
                flush_batch()
        cap.release()

    flush_batch()

    mean = float(np.mean(scores))
    std = float(np.std(scores))
    print(f"  {name:<35} CLIP-IQA: {mean:.4f} ± {std:.4f}  ({len(scores)} frames)")
    return mean, std, len(scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=None,
                        help="Subset of variant keys to evaluate (default: all)")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    variants = VARIANTS
    if args.variants:
        variants = {k: v for k, v in VARIANTS.items() if k in args.variants}

    results = {}
    for name, video_dir in variants.items():
        if not os.path.isdir(video_dir):
            print(f"  [{name}] Directory not found: {video_dir}, skipping.")
            continue
        result = score_variant(name, video_dir, args.batch_size, device)
        if result is not None:
            results[name] = result

    print("\n=== CLIP-IQA Results (nuCarla, sorted) ===")
    for name, (mean, std, n) in sorted(results.items(), key=lambda x: -x[1][0]):
        print(f"  {name:<35} {mean:.4f} ± {std:.4f}  ({n} frames)")


if __name__ == "__main__":
    main()
