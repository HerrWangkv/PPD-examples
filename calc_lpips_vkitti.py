"""
Compute paired LPIPS between translated vKITTI images and real KITTI tracking frames.

Each translated file {scene}_clone_{frame:05d}.png is paired with the corresponding
real KITTI tracking frame: image_02/{scene}/{frame:06d}.png

Scenes: 0001, 0002, 0006, 0018, 0020 (2126 paired frames)

Usage:
    source .venv/bin/activate

    # Raw sim baseline (vKITTI clone vs KITTI real):
    python calc_lpips_vkitti.py --clone_baseline

    # Translated variant:
    python calc_lpips_vkitti.py --gen_folder /mrtstorage/users/kwang/vkitti_translated/flowedit
"""

import argparse
import os
import re
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import lpips

KITTI_TRACKING = "/tmp/kitti/kitti-tracking/training/image_02"
VKITTI_SCENES = ["0001", "0002", "0006", "0018", "0020"]
VKITTI_CLONE_ROOT = "/mrtstorage/datasets_tmp/vkitti/vkitti_1.3.1_rgb"

to_tensor = transforms.ToTensor()  # [0,1]


def load_image(path, size=None):
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize(size, Image.BILINEAR)
    t = to_tensor(img) * 2 - 1  # [0,1] → [-1,1] for lpips
    return t.unsqueeze(0)


def kitti_path(scene, frame_5digit):
    """Map vKITTI 5-digit frame to 6-digit KITTI tracking path."""
    frame_6digit = f"{int(frame_5digit):06d}.png"
    return os.path.join(KITTI_TRACKING, scene, frame_6digit)


def collect_pairs_from_gen(gen_folder):
    """Collect (gen_path, kitti_path) pairs from a translated gen folder."""
    pattern = re.compile(r"^(\d{4})_clone_(\d{5})\.(png|jpg)$")
    pairs = []
    for fname in sorted(os.listdir(gen_folder)):
        m = pattern.match(fname)
        if not m:
            continue
        scene, frame = m.group(1), m.group(2)
        if scene not in VKITTI_SCENES:
            continue
        real_path = kitti_path(scene, frame)
        if not os.path.exists(real_path):
            continue
        pairs.append((os.path.join(gen_folder, fname), real_path))
    return pairs


def collect_pairs_clone_baseline():
    """Collect (vKITTI clone path, KITTI tracking path) pairs."""
    pairs = []
    for scene in VKITTI_SCENES:
        clone_dir = os.path.join(VKITTI_CLONE_ROOT, scene, "clone")
        for fname in sorted(os.listdir(clone_dir)):
            if not fname.endswith(".png"):
                continue
            frame = os.path.splitext(fname)[0]  # e.g. '00080'
            real_path = kitti_path(scene, frame)
            if not os.path.exists(real_path):
                continue
            pairs.append((os.path.join(clone_dir, fname), real_path))
    return pairs


def compute_lpips(pairs, device, net="alex"):
    loss_fn = lpips.LPIPS(net=net).to(device)
    loss_fn.eval()

    # Use gen resolution (1280×384) — cleanly divisible by 16, avoids AlexNet size mismatch
    ref_size = (1280, 384)

    scores = []
    for gen_path, real_path in tqdm(pairs, desc="LPIPS", unit="img"):
        gen_t = load_image(gen_path, size=ref_size).to(device)
        real_t = load_image(real_path, size=ref_size).to(device)
        with torch.no_grad():
            d = loss_fn(gen_t, real_t)
        scores.append(d.item())

    return np.array(scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_folder", type=str,
                        default="/mrtstorage/users/kwang/vkitti_translated/flowedit")
    parser.add_argument("--clone_baseline", action="store_true",
                        help="Compute LPIPS for vKITTI clone vs KITTI real (raw sim baseline)")
    parser.add_argument("--net", type=str, default="alex",
                        choices=["alex", "vgg", "squeeze"],
                        help="LPIPS backbone (alex recommended)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if not os.path.isdir(KITTI_TRACKING):
        raise FileNotFoundError(
            f"KITTI tracking not found at {KITTI_TRACKING}. "
            "Is kitti.sqfs mounted at /tmp/kitti?"
        )

    if args.clone_baseline:
        pairs = collect_pairs_clone_baseline()
        label = "vKITTI clone (raw sim baseline)"
    else:
        pairs = collect_pairs_from_gen(args.gen_folder)
        label = args.gen_folder

    print(f"[Info] {len(pairs)} paired images found")
    if not pairs:
        raise RuntimeError("No pairs found — check gen_folder or clone paths.")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    scores = compute_lpips(pairs, device, net=args.net)

    print(f"\n{'='*60}")
    print(f"  Gen:  {label}")
    print(f"  Pairs: {len(scores)}")
    print(f"  LPIPS ({args.net}): {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
