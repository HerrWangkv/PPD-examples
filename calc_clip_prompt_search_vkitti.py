"""
Score the full 30-pair candidate pool (calc_clip_prompt_search.CANDIDATES) on
the vKITTI paper-table variants, saving per-image scores for joint search with
the nuCarla per-scene data.

Usage:
    CUDA_VISIBLE_DEVICES=0 python calc_clip_prompt_search_vkitti.py --variants ppd_r12 dropll_J4_r12
"""

import argparse

import numpy as np
import torch

from calc_clip_prompt_search import CANDIDATES
from calc_clipiqa_prompts_nucarla import make_metric
from calc_clipiqa_prompts_vkitti import score_variant

PAPER_VARIANTS = ["input", "flowedit", "dnaedit", "cosmos_depth_edge_imgs",
                  "ppd_r12", "dropll_J4_r12"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=PAPER_VARIANTS)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    metric = make_metric(CANDIDATES, device)
    n_pairs = len(CANDIDATES)

    for v in args.variants:
        s = score_variant(metric, n_pairs, v, args.batch_size, device)
        np.save(f"clip_pool_vkitti_{v}.npy", s)
        print(f"  {v:<32} saved {s.shape}, pool mean={s.mean():.4f}")


if __name__ == "__main__":
    main()
