"""
Second-round prompt candidates: wording variants of the four axes of the
near-miss combo [9,11,23,27] (clear-visibility / lens-flare / materials /
sharp-render), plus pure-optics pairs. Scored on the same variants as pool 1;
results merged with pool 1 in the v5 joint search (uniform mean, 34 constraints).

Outputs:
  clip_pool2_vkitti_<variant>.npy        (per-image, n x len(NEW_CANDIDATES))
  clip_prompt_pool2_scenes.npz           (nuCarla per-scene, 6 key variants)

Usage:
    CUDA_VISIBLE_DEVICES=0 python calc_prompt_pool2.py --bench vkitti --variants ppd_r8 ...
    CUDA_VISIBLE_DEVICES=0 python calc_prompt_pool2.py --bench nucarla
"""

import argparse

import numpy as np
import torch

from calc_clipiqa_prompts_nucarla import VARIANTS as NUC_VARIANTS, make_metric
from calc_clip_prompt_search import score_per_scene
from calc_clipiqa_prompts_vkitti import score_variant as vk_score_variant

NEW_CANDIDATES = [
    # axis 9: visibility / glare-wash
    ("A scene with clear visibility.", "A scene washed out by haze."),
    ("A photo with good visibility.", "An image veiled by glare."),
    ("A clear unobstructed view.", "A view washed out by bright glare."),
    # axis 11: lens flare / sun glare
    ("A photo free of lens flare.", "An image dominated by lens flare."),
    ("A photo without sun glare.", "An image with artificial sun glare."),
    ("A photo with a clean lens.", "An image with rendered light bloom."),
    # axis 23: materials
    ("Real-world materials.", "Plastic-like materials."),
    ("Realistic surface materials.", "Artificial plastic surfaces."),
    ("Natural material textures.", "Synthetic plastic textures."),
    ("Weathered realistic surfaces.", "Pristine plastic surfaces."),
    # axis 27: sharpness / render blur
    ("A crisp real photo.", "A soft computer render."),
    ("A sharp photograph.", "A blurry rendering."),
    ("A photo in sharp focus.", "An out-of-focus render."),
    # pure optics / new
    ("A photo with natural depth of field.", "A render with uniform blur."),
    ("A photo with realistic exposure.", "A render with blown-out highlights."),
    ("A photo with film grain.", "A noiseless smooth render."),
]

NUC_KEY = ["ditto", "dropll_r30_J5", "wavelet_r30", "ppd_r30",
           "cosmos_depth_seg_vis_edge", "input"]

VK_VARIANTS_DEFAULT = [
    "input", "flowedit", "dnaedit", "cosmos_depth_edge_imgs",
    "ppd_r8", "ppd_r12", "ppd_r16", "ppd_r20", "ppd_r24", "ppd_r32",
    "baseline_r4", "baseline_r8", "baseline_r10", "baseline_r12",
    "baseline_newprompt", "baseline_r20", "baseline_r24",
    "dropll_J4_r8", "dropll_J4_r12", "dropll_J4_r20", "dropll_J4_r24",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", choices=["vkitti", "nucarla"], required=True)
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    metric = make_metric(NEW_CANDIDATES, device)
    n_pairs = len(NEW_CANDIDATES)
    print(f"{n_pairs} new candidate pairs, bench={args.bench}")

    if args.bench == "nucarla":
        variants = args.variants or NUC_KEY
        data = {}
        for v in variants:
            data[v] = score_per_scene(metric, n_pairs, NUC_VARIANTS[v], args.batch_size, device)
            print(f"  {v}: {data[v].shape}")
        np.savez("clip_prompt_pool2_scenes.npz", **data)
        print("saved clip_prompt_pool2_scenes.npz")
    else:
        variants = args.variants or VK_VARIANTS_DEFAULT
        for v in variants:
            s = vk_score_variant(metric, n_pairs, v, args.batch_size, device)
            np.save(f"clip_pool2_vkitti_{v}.npy", s)
            print(f"  {v:<32} saved {s.shape}")


if __name__ == "__main__":
    main()
