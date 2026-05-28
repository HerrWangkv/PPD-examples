"""
Compute CLIP-IQA for vKITTI translated variants using piq's standard implementation.

CLIP-IQA (Wang et al., AAAI 2023) measures perceptual image quality via antonymous
prompt pairs ("Good photo" / "Bad photo"). Higher = better quality.

Uses piq.CLIPIQA which follows the official implementation.

Usage:
    source .venv/bin/activate

    # Raw sim baseline:
    python calc_clipiqa_vkitti.py --clone_baseline

    # Translated variant:
    python calc_clipiqa_vkitti.py --gen_folder outputs/vkitti/cosmos_depth_edge_imgs --clone_only
"""

import argparse
import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from piq import CLIPIQA

VKITTI_SCENES = ["0001", "0002", "0006", "0018", "0020"]
VKITTI_CLONE_ROOT = "/mrtstorage/datasets_tmp/vkitti/vkitti_1.3.1_rgb"

to_tensor = transforms.ToTensor()


def collect_clone_images():
    paths = []
    for scene in VKITTI_SCENES:
        clone_dir = os.path.join(VKITTI_CLONE_ROOT, scene, "clone")
        for fname in sorted(os.listdir(clone_dir)):
            if fname.endswith(".png"):
                paths.append(os.path.join(clone_dir, fname))
    return paths


def collect_gen_images(gen_folder, clone_only=False):
    return sorted([
        os.path.join(gen_folder, f)
        for f in os.listdir(gen_folder)
        if f.lower().endswith((".png", ".jpg"))
        and (not clone_only or "_clone_" in f)
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_folder", type=str, default=None)
    parser.add_argument("--clone_baseline", action="store_true")
    parser.add_argument("--clone_only", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    metric = CLIPIQA().to(device)

    if args.clone_baseline:
        paths = collect_clone_images()
        label = "vKITTI clone (raw sim)"
    elif args.gen_folder:
        paths = collect_gen_images(args.gen_folder, clone_only=args.clone_only)
        label = args.gen_folder
    else:
        parser.error("Provide --gen_folder or --clone_baseline")

    print(f"[Info] {len(paths)} images — {label}")

    scores = []
    for i in tqdm(range(0, len(paths), args.batch_size), desc="CLIP-IQA"):
        batch_paths = paths[i:i + args.batch_size]
        imgs = torch.stack([
            to_tensor(Image.open(p).convert("RGB")) for p in batch_paths
        ]).to(device)
        with torch.no_grad():
            s = metric(imgs)
        scores.extend(s.cpu().numpy().tolist() if s.ndim > 0 else [s.item()])

    scores = np.array(scores)
    mean, std = scores.mean(), scores.std()

    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"  Images:   {len(scores)}")
    print(f"  CLIP-IQA: {mean:.4f} ± {std:.4f}")
    print(f"{'='*50}")
    print(f"\nCLIP-IQA: {mean:.4f}")


if __name__ == "__main__":
    main()
