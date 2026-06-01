"""
Compute CLIP-IQA for Hypersim translated variants.

Usage:
    python calc_clipiqa_hypersim.py --gen_folder outputs/hypersim/flowedit
"""

import argparse
import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from piq import CLIPIQA

to_tensor = transforms.ToTensor()


def get_image_paths(folder):
    exts = ('.jpg', '.jpeg', '.png')
    return sorted(
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.lower().endswith(exts)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_folder", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    metric = CLIPIQA().to(device)

    paths = get_image_paths(args.gen_folder)
    print(f"{len(paths)} images in {args.gen_folder}")

    scores = []
    for i in tqdm(range(0, len(paths), args.batch_size), desc="CLIP-IQA"):
        batch = []
        for p in paths[i:i+args.batch_size]:
            img = Image.open(p).convert("RGB")
            batch.append(to_tensor(img))
        imgs = torch.stack(batch).to(device)
        with torch.no_grad():
            s = metric(imgs).cpu().numpy()
        scores.extend(s.tolist() if s.ndim > 0 else [float(s)])

    mean = float(np.mean(scores))
    std  = float(np.std(scores))
    print(f"\n  CLIP-IQA: {mean:.4f} ± {std:.4f}")
    print("=" * 45)
    print(f"CLIP-IQA: {mean:.4f}")


if __name__ == "__main__":
    main()
