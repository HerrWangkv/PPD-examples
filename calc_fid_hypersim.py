"""
Calculate FID for Hypersim sim2real experiments vs ScanNet.
Compares: original Hypersim, WPD-translated, FlowEdit-translated.
"""

import os
import shutil
import tempfile
import argparse
from cleanfid import fid

REAL_DIR = "/mrtstorage/datasets_tmp/scannet"

EXPERIMENTS = {
    "hypersim_original": "/mrtstorage/datasets_tmp/hypersim",
    "hypersim_wavelet":  "/mrtstorage/users/kwang/hypersim_wavelet",
    "hypersim_flowedit": "/mrtstorage/users/kwang/hypersim_flowedit",
}


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


def compute_fid(gen_path, exp_name):
    print(f"\n{'='*60}")
    print(f" Experiment: {exp_name}")
    print(f"{'='*60}")

    with FlattenedFolder(REAL_DIR) as real_flat:
        with FlattenedFolder(gen_path) as gen_flat:
            print(f"  Gen:  {gen_flat}")
            print(f"  Real: {real_flat}")

            fid_score = fid.compute_fid(
                fdir1=gen_flat,
                fdir2=real_flat,
                mode="clean",
                use_dataparallel=False,
            )

    print(f"  FID: {fid_score:.4f}")
    return fid_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None,
                        help="Run only one experiment: original / wavelet / flowedit")
    args = parser.parse_args()

    exps = {k: v for k, v in EXPERIMENTS.items()
            if args.only is None or args.only in k}

    results = {}
    for name, path in exps.items():
        if not os.path.exists(path):
            print(f"[SKIP] {name}: path not found ({path})")
            continue
        results[name] = compute_fid(path, name)

    print(f"\n{'='*60}")
    print(f"{'Experiment':<30} {'FID':>10}")
    print(f"{'-'*60}")
    for name, f in results.items():
        print(f"{name:<30} {f:>10.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
