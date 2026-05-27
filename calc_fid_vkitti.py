"""
Calculate FID & KID for vKITTI sim2real experiments.

Real reference: KITTI tracking training image_02, sequences 0001/0002/0006/0018/0020
                2126 frames — exact 1:1 match with vKITTI clone condition (same scenes,
                same frame count, same weather).

Baseline: run with --gen_folder pointing to vKITTI clone to get the raw sim-to-real gap.

Usage:
    source .venv/bin/activate
    # Baseline (raw sim gap):
    python calc_fid_vkitti.py --gen_folder /mrtstorage/datasets_tmp/vkitti/vkitti_1.3.1_rgb --clone_baseline
    # Translated variant:
    python calc_fid_vkitti.py --gen_folder /mrtstorage/users/kwang/vkitti_translated/flowedit
"""

import argparse
import os
import shutil
import tempfile
from cleanfid import fid

KITTI_TRACKING = "/tmp/kitti/kitti-tracking/training/image_02"
VKITTI_SCENES = ["0001", "0002", "0006", "0018", "0020"]
KITTI_STATS_NAME = "kitti_tracking_vkitti_scenes"

# vKITTI clone root: {VKITTI_CLONE_ROOT}/{scene}/clone/*.png
VKITTI_CLONE_ROOT = "/mrtstorage/datasets_tmp/vkitti/vkitti_1.3.1_rgb"


class KittiTrackingFolder:
    """Flat temp dir of KITTI tracking frames for the 5 vKITTI scenes (2126 frames)."""
    def __init__(self):
        self.temp_dir = None

    def __enter__(self):
        image_files = []
        for seq in VKITTI_SCENES:
            seq_dir = os.path.join(KITTI_TRACKING, seq)
            if not os.path.isdir(seq_dir):
                raise FileNotFoundError(f"KITTI tracking sequence not found: {seq_dir}")
            for f in sorted(os.listdir(seq_dir)):
                if f.lower().endswith((".png", ".jpg")):
                    image_files.append((seq, f, os.path.join(seq_dir, f)))

        print(f"[KITTI] {len(image_files)} frames from sequences {VKITTI_SCENES}")
        self.temp_dir = os.path.abspath(tempfile.mkdtemp(prefix="kitti_tracking_flat_"))
        for seq, fname, src in image_files:
            os.symlink(src, os.path.join(self.temp_dir, f"{seq}_{fname}"))
        print(f"[KITTI] Flat dir: {self.temp_dir}")
        return self.temp_dir

    def __exit__(self, *_):
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


class VkittiCloneFolder:
    """Flat temp dir of vKITTI clone frames for the 5 scenes (2126 frames)."""
    def __init__(self):
        self.temp_dir = None

    def __enter__(self):
        image_files = []
        for scene in VKITTI_SCENES:
            clone_dir = os.path.join(VKITTI_CLONE_ROOT, scene, "clone")
            if not os.path.isdir(clone_dir):
                raise FileNotFoundError(f"vKITTI clone not found: {clone_dir}")
            for f in sorted(os.listdir(clone_dir)):
                if f.lower().endswith((".png", ".jpg")):
                    image_files.append((scene, f, os.path.join(clone_dir, f)))

        print(f"[vKITTI clone] {len(image_files)} frames from scenes {VKITTI_SCENES}")
        self.temp_dir = os.path.abspath(tempfile.mkdtemp(prefix="vkitti_clone_flat_"))
        for scene, fname, src in image_files:
            os.symlink(src, os.path.join(self.temp_dir, f"{scene}_{fname}"))
        print(f"[vKITTI clone] Flat dir: {self.temp_dir}")
        return self.temp_dir

    def __exit__(self, *_):
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


class FlattenedFolder:
    """Flat temp dir for a gen folder (handles already-flat folders without copy)."""
    def __init__(self, path, suffix=(".png", ".jpg", ".jpeg"), clone_only=False):
        self.path = path
        self.suffix = suffix
        self.clone_only = clone_only
        self.temp_dir = None

    def __enter__(self):
        image_files = []
        abs_path = os.path.abspath(self.path)
        for root, _, files in os.walk(abs_path):
            for f in files:
                if f.lower().endswith(self.suffix):
                    if not self.clone_only or '_clone_' in f:
                        image_files.append(os.path.join(root, f))
        print(f"[Gen] {len(image_files)} images in {abs_path}"
              + (" (clone only)" if self.clone_only else ""))

        if not self.clone_only:
            root_images = [f for f in os.listdir(self.path) if f.lower().endswith(self.suffix)]
            if len(root_images) == len(image_files):
                return self.path

        self.temp_dir = os.path.abspath(tempfile.mkdtemp(prefix="fid_flatten_"))
        for src in image_files:
            rel = os.path.relpath(src, abs_path)
            os.symlink(src, os.path.join(self.temp_dir, rel.replace(os.sep, "_")))
        return self.temp_dir

    def __exit__(self, *_):
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_folder", type=str,
                        default="/mrtstorage/users/kwang/vkitti_translated/flowedit")
    parser.add_argument("--mode", type=str, default="legacy_pytorch",
                        choices=["clean", "legacy_pytorch"])
    parser.add_argument("--clone_baseline", action="store_true",
                        help="Use vKITTI clone as gen (raw sim baseline)")
    parser.add_argument("--clone_only", action="store_true",
                        help="Only use clone-condition images from gen folder")
    parser.add_argument("--recompute_stats", action="store_true")
    args = parser.parse_args()

    if not os.path.isdir(KITTI_TRACKING):
        raise FileNotFoundError(
            f"KITTI tracking not found at {KITTI_TRACKING}. "
            "Is kitti.sqfs mounted at /tmp/kitti?"
        )

    stats_name = f"{KITTI_STATS_NAME}_{args.mode}"

    gen_ctx = VkittiCloneFolder() if args.clone_baseline else FlattenedFolder(args.gen_folder, clone_only=args.clone_only)
    gen_label = "vKITTI clone (baseline)" if args.clone_baseline else args.gen_folder

    with KittiTrackingFolder() as real_flat:
        with gen_ctx as gen_flat:
            print(f"\n{'='*60}")
            print(f"  Gen:  {gen_label}")
            print(f"  Real: KITTI tracking scenes {VKITTI_SCENES} (2126 frames)")
            print(f"  Mode: {args.mode}")
            print(f"{'='*60}\n")

            if args.recompute_stats or not fid.test_stats_exists(stats_name, mode=args.mode):
                print("[Cache] Computing KITTI tracking stats...")
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
    print(f"  Gen: {gen_label}")
    print(f"  FID:  {fid_score:.4f}")
    print(f"  KID:  {kid_score:.6f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
