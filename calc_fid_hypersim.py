"""
Calculate FID & KID for Hypersim sim2real experiments vs ScanNet test split.

Real reference: ScanNet test split (/mrtstorage/datasets_tmp/scannet/test/)
Gen: any flat folder of translated Hypersim images.

Usage:
    python calc_fid_hypersim.py --gen_folder outputs/hypersim/dropll_J5_r24
    python calc_fid_hypersim.py --gen_folder outputs/hypersim/input
"""

import os
import shutil
import tempfile
import argparse
from cleanfid import fid

REAL_DIR = "/mrtstorage/datasets_tmp/scannet"
SCANNET_STATS_NAME = "scannet_test_legacy"


class FlattenedFolder:
    def __init__(self, original_path):
        self.original_path = original_path
        self.temp_dir = None
        self.final_path = original_path

    def __enter__(self):
        image_files = []
        abs_path = os.path.abspath(self.original_path)
        for root, _, files in os.walk(abs_path):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(root, f))

        print(f"[Info] Found {len(image_files)} images in {abs_path}")

        root_images = [f for f in os.listdir(self.original_path)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if len(root_images) == len(image_files):
            return self.original_path

        self.temp_dir = os.path.abspath(tempfile.mkdtemp(prefix="fid_flatten_"))
        print(f"[Info] Flattening nested folders → {self.temp_dir}")
        abs_base = os.path.abspath(self.original_path)
        for src in image_files:
            rel = os.path.relpath(src, abs_base)
            flat_name = rel.replace(os.sep, "_")
            dst = os.path.join(self.temp_dir, flat_name)
            os.symlink(src, dst)

        self.final_path = self.temp_dir
        return self.final_path

    def __exit__(self, *_):
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_folder", type=str, required=True,
                        help="Folder of translated Hypersim images")
    parser.add_argument("--mode", type=str, default="legacy_pytorch",
                        choices=["clean", "legacy_pytorch"])
    parser.add_argument("--recompute_stats", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.gen_folder):
        raise FileNotFoundError(f"gen_folder not found: {args.gen_folder}")

    stats_name = f"{SCANNET_STATS_NAME}_{args.mode}"

    with FlattenedFolder(REAL_DIR) as real_flat:
        with FlattenedFolder(args.gen_folder) as gen_flat:
            print(f"\n{'='*60}")
            print(f"  Gen:  {args.gen_folder}")
            print(f"  Real: ScanNet test ({REAL_DIR})")
            print(f"  Mode: {args.mode}")
            print(f"{'='*60}\n")

            if args.recompute_stats or not fid.test_stats_exists(stats_name, mode=args.mode):
                print("[Cache] Computing ScanNet stats...")
                fid.make_custom_stats(name=stats_name, fdir=real_flat, mode=args.mode)
            else:
                print(f"[Cache] Using cached stats '{stats_name}'")

            print("Computing FID...")
            fid_score = fid.compute_fid(
                fdir1=gen_flat,
                dataset_name=stats_name,
                dataset_split="custom",
                mode=args.mode,
            )
            print("Computing KID...")
            kid_score = fid.compute_kid(
                fdir1=gen_flat,
                dataset_name=stats_name,
                dataset_split="custom",
                mode=args.mode,
            )

    print(f"\n{'='*60}")
    print(f"  Gen: {args.gen_folder}")
    print(f"  FID:  {fid_score:.4f}")
    print(f"  KID:  {kid_score:.6f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
