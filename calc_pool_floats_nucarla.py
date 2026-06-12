"""
Score ALL 46 candidate pairs (pool1 + pool2) on the 4 nuCarla variants that
were missing from the search constraints (cosmos, cosmos_depth_edge, vace_gray,
dnaedit). Saves per-scene means so future searches can constrain against them.

Output: clip_prompt_pools_floats.npz  (per variant: (60, 46))

Usage:
    CUDA_VISIBLE_DEVICES=0 python calc_pool_floats_nucarla.py --variants cosmos vace_gray
"""

import argparse

import numpy as np
import torch

from calc_clipiqa_prompts_nucarla import VARIANTS, make_metric
from calc_clip_prompt_search import CANDIDATES as POOL1, score_per_scene
from calc_prompt_pool2 import NEW_CANDIDATES as POOL2

FLOATS = ["cosmos", "cosmos_depth_edge", "vace_gray", "dnaedit"]
ALL_PAIRS = list(POOL1) + list(POOL2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=FLOATS)
    parser.add_argument("--out", default=None,
                        help="npz path (default clip_prompt_pools_floats_<first-variant>.npz)")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    metric = make_metric(ALL_PAIRS, device)
    data = {}
    for v in args.variants:
        data[v] = score_per_scene(metric, len(ALL_PAIRS), VARIANTS[v], args.batch_size, device)
        print(f"  {v}: {data[v].shape}")
    out = args.out or f"clip_prompt_pools_floats_{args.variants[0]}.npz"
    np.savez(out, **data)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
