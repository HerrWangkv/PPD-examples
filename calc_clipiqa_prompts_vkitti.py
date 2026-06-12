"""
CLIP synthetic-residual ensemble (selected on nuCarla) applied to vKITTI variants.

Out-of-distribution validation of the prompt ensemble: prompts were searched on
nuCarla only; here they are applied unchanged to all vKITTI->KITTI variants plus
real KITTI frames as gold anchor.

Pre-registered predictions:
  1. REAL_KITTI > all translated variants (hard sanity)
  2. WPD drop_ll > WPD baseline at matched radius
  3. raw vKITTI input at/near bottom

Usage:
    source .venv/bin/activate
    CUDA_VISIBLE_DEVICES=0 python calc_clipiqa_prompts_vkitti.py --variants ppd_r12 dropll_J4_r12 REAL_KITTI
    python calc_clipiqa_prompts_vkitti.py            # all variants + REAL_KITTI
"""

import argparse
import glob
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from calc_clipiqa_prompts_nucarla import PROMPT_SETS, make_metric

VKITTI_OUT = "outputs/vkitti"
KITTI_TRACKING = "/tmp/kitti/kitti-tracking/training/image_02"
VKITTI_SCENES = ["0001", "0002", "0006", "0018", "0020"]

to_tensor = transforms.ToTensor()


def collect_paths(variant):
    if variant == "REAL_KITTI":
        paths = []
        for seq in VKITTI_SCENES:
            paths += sorted(glob.glob(os.path.join(KITTI_TRACKING, seq, "*.png")))
        return paths
    folder = os.path.join(VKITTI_OUT, variant)
    paths = sorted(
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.lower().endswith((".png", ".jpg"))
    )
    clone = [p for p in paths if "_clone_" in os.path.basename(p)]
    return clone if clone else paths


def score_variant(metric, n_pairs, variant, batch_size, device):
    paths = collect_paths(variant)
    scores = []
    batch = []

    def flush():
        if not batch:
            return
        imgs = torch.stack(batch).to(device)
        with torch.no_grad():
            s = metric(imgs)
        scores.extend(s.reshape(-1, n_pairs).cpu().numpy())
        batch.clear()

    # KITTI sequences differ in resolution -> flush whenever the size changes
    for p in tqdm(paths, desc=variant, leave=False):
        t = to_tensor(Image.open(p).convert("RGB"))
        if batch and batch[-1].shape != t.shape:
            flush()
        batch.append(t)
        if len(batch) >= batch_size:
            flush()
    flush()
    return np.array(scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=None,
                        help="variant dir names under outputs/vkitti, or REAL_KITTI")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--prompt_set", default="residual_v2", choices=sorted(PROMPT_SETS))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompt_pairs = PROMPT_SETS[args.prompt_set]
    metric = make_metric(prompt_pairs, device)
    n_pairs = len(prompt_pairs)

    variants = args.variants
    if variants is None:
        variants = ["REAL_KITTI"] + sorted(os.listdir(VKITTI_OUT))

    results = {}
    for v in variants:
        if v != "REAL_KITTI" and not os.path.isdir(os.path.join(VKITTI_OUT, v)):
            print(f"[{v}] missing, skipping")
            continue
        s = score_variant(metric, n_pairs, v, args.batch_size, device)
        results[v] = (s.mean(), s.mean(axis=0), len(s))
        pair_str = "  ".join(f"p{i}={x:.4f}" for i, x in enumerate(s.mean(axis=0)))
        print(f"  {v:<32} ensemble={s.mean():.4f}  {pair_str}  ({len(s)} imgs)")
        if args.prompt_set == "residual_v7":
            os.makedirs("logs/vkitti_eval", exist_ok=True)
            with open(f"logs/vkitti_eval/{v}_clipres.log", "w") as f:
                f.write(f"variant: {v}\nprompt_set: {args.prompt_set}\n")
                for i, (pos, neg) in enumerate(prompt_pairs):
                    f.write(f"pair{i}: {pos!r} / {neg!r} = {s.mean(axis=0)[i]:.4f}\n")
                f.write(f"images: {len(s)}\nCLIP-Residual: {s.mean():.4f}\n")

    print("\n=== CLIP-Residual vKITTI (sorted, higher = more real) ===")
    for v, (ens, _, n) in sorted(results.items(), key=lambda kv: -kv[1][0]):
        print(f"  {v:<32} {ens:.4f}")


if __name__ == "__main__":
    main()
