#!/usr/bin/env python3
"""
Collect a specific frame across all translation variants for vKITTI or Hypersim.

Usage:
    python collect_vkitti_variants.py 0001_clone_00042.png
    python collect_vkitti_variants.py ai_001_001_cam00_frame0042.jpg --dataset hypersim
    python collect_vkitti_variants.py 0001_clone_00042.png --variants paper
    python collect_vkitti_variants.py 0001_clone_00042.png --out /tmp/compare/
"""

import argparse
import shutil
from pathlib import Path

DATASET_DIRS = {
    "vkitti":   Path("outputs/vkitti"),
    "hypersim": Path("outputs/hypersim"),
}

PAPER_VARIANTS = {
    "vkitti": [
        "input", "flowedit", "dnaedit", "cosmos_depth_edge_imgs",
        "ppd_r12", "dropll_J4_r12",
    ],
    "hypersim": [
        "input", "flowedit", "dnaedit", "cosmos_depth_edge_imgs",
        "ppd_r20", "dropll_J5_r24",
    ],
}


def collect(filename: str, dataset: str, out_dir: Path, variants: list[str] | None):
    base_dir = DATASET_DIRS[dataset]
    out_dir.mkdir(parents=True, exist_ok=True)

    if variants is None:
        variants = sorted(d.name for d in base_dir.iterdir() if d.is_dir())

    stem = Path(filename).stem
    found, missing = [], []
    for v in variants:
        src = None
        for ext in (Path(filename).suffix, ".jpg", ".png"):
            candidate = base_dir / v / (stem + ext)
            if candidate.exists():
                src = candidate
                break
        if src:
            dst = out_dir / f"{v}__{src.name}"
            shutil.copy2(src, dst)
            found.append(v)
        else:
            missing.append(v)

    print(f"[{dataset}] Collected {len(found)}/{len(variants)} → {out_dir}")
    for v in found:
        print(f"  {v}")
    if missing:
        print(f"Missing: {', '.join(missing)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="e.g. 0001_clone_00042.png or ai_001_001_cam00_frame0042.jpg")
    parser.add_argument("--dataset", choices=["vkitti", "hypersim"], default="vkitti")
    parser.add_argument("--out", default=None,
                        help="output dir (default: outputs/compare/<dataset>/<stem>)")
    parser.add_argument("--variants", nargs="+", default=None,
                        help="variants to collect; use 'paper' for paper set, default=all")
    args = parser.parse_args()

    stem = Path(args.filename).stem
    out_dir = Path(args.out) if args.out else Path("outputs/compare") / args.dataset / stem

    variants = args.variants
    if variants == ["paper"]:
        variants = PAPER_VARIANTS[args.dataset]

    collect(args.filename, args.dataset, out_dir, variants)
