"""
Calculate cross-consistency Depth SSIM for SURREAL sim2real experiments.

Measures structure preservation: runs Depth Anything V2 on both the original
synthetic image and the translated image, then computes SSIM between the two
depth maps. A high score means translation preserved the body geometry.

Compares: WPD-translated vs original, FlowEdit-translated vs original.
"""

import os
import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from skimage.metrics import structural_similarity as ssim


SURREAL_DIR = "/mrtstorage/datasets_tmp/surreal"

EXPERIMENTS = {
    "surreal_wavelet":  "/mrtstorage/users/kwang/surreal_wavelet",
    "surreal_flowedit": "/mrtstorage/users/kwang/surreal_flowedit",
}


def get_image_paths(folder):
    exts = ('.png', '.jpg', '.jpeg')
    paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(exts):
                paths.append(os.path.join(root, f))
    return sorted(paths)


def predict_depth_batch(model, processor, images, device):
    """Predict relative depth for a list of PIL images. Returns list of H×W numpy arrays."""
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        depths = outputs.predicted_depth  # B×H×W

    results = []
    for i, img in enumerate(images):
        h, w = img.height, img.width
        d = torch.nn.functional.interpolate(
            depths[i].unsqueeze(0).unsqueeze(0), size=(h, w),
            mode="bicubic", align_corners=False,
        ).squeeze().cpu().numpy()
        results.append(d)
    return results


def normalize(depth):
    """Min-max normalize depth map to [0, 1] for SSIM comparison."""
    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min < 1e-6:
        return np.zeros_like(depth)
    return (depth - d_min) / (d_max - d_min)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None,
                        help="Run only one experiment: wavelet / flowedit")
    parser.add_argument("--model_id", type=str,
                        default="depth-anything/Depth-Anything-V2-Large-hf")
    parser.add_argument("--surreal_dir", type=str, default=SURREAL_DIR)
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Number of image pairs per GPU batch")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU index to use")
    parser.add_argument("--resize", type=int, default=None,
                        help="Resize shorter side before depth prediction (e.g. 384). "
                             "Default: use image as-is.")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    print(f"Loading {args.model_id} on {device}...")
    processor = AutoImageProcessor.from_pretrained(args.model_id)
    model = AutoModelForDepthEstimation.from_pretrained(args.model_id).to(device)
    model.eval()

    exps = {k: v for k, v in EXPERIMENTS.items()
            if args.only is None or args.only in k}

    all_results = {}

    for exp_name, gen_dir in exps.items():
        if not os.path.exists(gen_dir):
            print(f"[SKIP] {exp_name}: {gen_dir} not found")
            continue

        print(f"\n{'='*60}\n Experiment: {exp_name}\n{'='*60}")

        gen_paths = get_image_paths(gen_dir)
        print(f"  Found {len(gen_paths)} translated images")

        # Build valid pairs
        pairs = []
        missing = 0
        for gen_path in gen_paths:
            fname = os.path.basename(gen_path)
            orig_path = os.path.join(args.surreal_dir, fname)
            if not os.path.exists(orig_path):
                missing += 1
            else:
                pairs.append((orig_path, gen_path))

        ssim_scores = []
        bs = args.batch_size

        for i in tqdm(range(0, len(pairs), bs)):
            batch = pairs[i:i + bs]
            orig_imgs, gen_imgs = [], []
            for orig_path, gen_path in batch:
                try:
                    orig_img = Image.open(orig_path).convert("RGB")
                    gen_img  = Image.open(gen_path).convert("RGB")
                    if args.resize:
                        orig_img = orig_img.resize((args.resize, args.resize), Image.LANCZOS)
                        gen_img  = gen_img.resize((args.resize, args.resize), Image.LANCZOS)
                    orig_imgs.append(orig_img)
                    gen_imgs.append(gen_img)
                except Exception as e:
                    print(f"  ERROR loading {os.path.basename(orig_path)}: {e}")

            if not orig_imgs:
                continue

            try:
                depths_orig = [normalize(d) for d in
                               predict_depth_batch(model, processor, orig_imgs, device)]
                depths_gen  = [normalize(d) for d in
                               predict_depth_batch(model, processor, gen_imgs, device)]
                for d_orig, d_gen in zip(depths_orig, depths_gen):
                    score = ssim(d_orig, d_gen, data_range=1.0, win_size=7)
                    ssim_scores.append(score)
            except Exception as e:
                print(f"  ERROR in batch {i//bs}: {e}")

        if missing:
            print(f"  Missing originals: {missing}")

        avg_ssim = np.mean(ssim_scores) if ssim_scores else float("nan")
        all_results[exp_name] = avg_ssim
        print(f"  Depth-SSIM: {avg_ssim:.4f}  (n={len(ssim_scores)})")

    print(f"\n{'='*60}")
    print(f"{'Experiment':<30} {'Depth-SSIM':>12}")
    print(f"{'-'*60}")
    for name, score in all_results.items():
        print(f"{name:<30} {score:>12.4f}")
    print(f"{'='*60}")
    print("Higher = better structure preservation during translation.")


if __name__ == "__main__":
    main()
