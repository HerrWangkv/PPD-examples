"""
Download MPII Human Pose images from HuggingFace (Voxel51/MPII_Human_Pose_Dataset).
Saves JPEGs to --output_dir. No registration required.
"""

import os
import argparse
from datasets import load_dataset
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str,
                        default="/mrtstorage/datasets_tmp/mpii")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split to download (train / validation / test)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Download only first N images (for testing)")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading Voxel51/MPII_Human_Pose_Dataset ({args.split})...")
    ds = load_dataset("Voxel51/MPII_Human_Pose_Dataset", split=args.split,
                      trust_remote_code=True)

    total = len(ds) if args.limit is None else min(args.limit, len(ds))
    print(f"Total samples: {len(ds)} | Downloading: {total}")

    saved = 0
    skipped = 0
    for i, sample in enumerate(tqdm(ds.select(range(total)))):
        img = sample.get("image") or sample.get("img")
        if img is None:
            # Try first PIL-typed field
            for v in sample.values():
                try:
                    from PIL import Image as PILImage
                    if isinstance(v, PILImage.Image):
                        img = v
                        break
                except Exception:
                    pass
        if img is None:
            skipped += 1
            continue

        fname = f"{i:06d}.jpg"
        out_path = os.path.join(args.output_dir, fname)
        if os.path.exists(out_path):
            skipped += 1
            continue

        try:
            img.convert("RGB").save(out_path, quality=95)
            saved += 1
        except Exception as e:
            print(f"  ERROR saving {fname}: {e}")
            skipped += 1

    print(f"Done. Saved: {saved} | Skipped/errors: {skipped}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
