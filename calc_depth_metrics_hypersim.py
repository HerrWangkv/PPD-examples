"""
Calculate depth metrics for Hypersim sim2real experiments using GT depth.

GT depth: /mrtstorage/datasets_tmp/hypersim_depth/ai_*_cam*_frame*.hdf5
  - HDF5 dataset key: "dataset", float32, values in meters
  - Invalid pixels: inf or nan (missing geometry)

Usage:
    python calc_depth_metrics_hypersim.py --gen_folder outputs/hypersim/dropll_J5_r24
    python calc_depth_metrics_hypersim.py --gen_folder outputs/hypersim/input
"""

import os
import argparse
import h5py
import torch
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from skimage.metrics import structural_similarity as ssim


DEPTH_DIR = "/mrtstorage/datasets_tmp/hypersim_depth"


def get_image_paths(folder):
    exts = ('.png', '.jpg', '.jpeg')
    paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(exts):
                paths.append(os.path.join(root, f))
    return sorted(paths)


def load_gt_depth(hdf5_path):
    """Load Hypersim GT depth (meters, float32). Returns (depth, valid_mask)."""
    with h5py.File(hdf5_path, "r") as f:
        depth = f["dataset"][:]  # H×W, float32, meters
    valid = np.isfinite(depth) & (depth > 0)
    return depth.astype(np.float32), valid


def predict_depth(model, processor, image_path, device):
    """Predict relative disparity from an image file."""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        pred = model(**inputs).predicted_depth  # 1×H×W
    h, w = image.height, image.width
    pred = torch.nn.functional.interpolate(
        pred.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
    ).squeeze().cpu().numpy()
    return pred


def align_least_squares(pred_disp, gt_disp, mask):
    """Least-squares scale+shift alignment: solve s*pred + t = gt on masked pixels."""
    p = pred_disp[mask]
    g = gt_disp[mask]
    if len(p) < 10:
        return pred_disp
    A = np.vstack([p, np.ones_like(p)]).T
    (s, t), _, _, _ = np.linalg.lstsq(A, g, rcond=None)
    return pred_disp * s + t


def compute_metrics(pred_depth, gt_depth, valid):
    p = pred_depth[valid]
    g = gt_depth[valid]
    if len(p) == 0:
        return None, None
    abs_rel = np.mean(np.abs(p - g) / (g + 1e-6))
    data_range = gt_depth[valid].max() - gt_depth[valid].min()
    if data_range < 1e-6:
        data_range = 1.0
    # Zero-fill invalid pixels so inf/nan don't propagate into SSIM
    gt_ssim = np.where(valid, gt_depth, 0.0)
    pred_ssim = np.where(valid, pred_depth, 0.0)
    ssim_val, _ = ssim(pred_ssim, gt_ssim, full=True,
                       data_range=data_range, win_size=7, channel_axis=None)
    return abs_rel, ssim_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_folder", type=str, required=True,
                        help="Folder of translated Hypersim images")
    parser.add_argument("--model_id", type=str,
                        default="depth-anything/Depth-Anything-V2-Large-hf")
    parser.add_argument("--depth_dir", type=str, default=DEPTH_DIR)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model_id} on {device}...")
    processor = AutoImageProcessor.from_pretrained(args.model_id)
    model = AutoModelForDepthEstimation.from_pretrained(args.model_id).to(device).eval()

    gen_paths = get_image_paths(args.gen_folder)
    print(f"\n{'='*60}")
    print(f"  Gen: {args.gen_folder}  ({len(gen_paths)} images)")
    print(f"{'='*60}")

    # Pre-load GT depth and filter valid items
    valid_items = []
    missing = 0
    for gen_path in gen_paths:
        stem = os.path.splitext(os.path.basename(gen_path))[0]
        depth_path = os.path.join(args.depth_dir, stem + ".hdf5")
        if not os.path.exists(depth_path):
            missing += 1
            continue
        valid_items.append((gen_path, depth_path))

    absrel_list, ssim_list = [], []
    BS = args.batch_size

    for i in tqdm(range(0, len(valid_items), BS), desc="Inference"):
        batch = valid_items[i:i+BS]
        try:
            images = [Image.open(p).convert("RGB") for p, _ in batch]
            inputs = processor(images=images, return_tensors="pt").to(device)
            with torch.no_grad():
                preds = model(**inputs).predicted_depth  # (B, H', W')

            for j, (gen_path, depth_path) in enumerate(batch):
                gt_depth, valid = load_gt_depth(depth_path)
                if valid.sum() < 100:
                    continue
                gh, gw = gt_depth.shape
                pred_disp = torch.nn.functional.interpolate(
                    preds[j:j+1].unsqueeze(1), size=(gh, gw),
                    mode="bicubic", align_corners=False
                ).squeeze().cpu().numpy()
                gt_disp = np.zeros_like(gt_depth)
                gt_disp[valid] = 1.0 / (gt_depth[valid] + 1e-6)
                aligned_disp = align_least_squares(pred_disp, gt_disp, valid)
                aligned_disp = np.clip(aligned_disp, 1e-3, None)
                pred_depth_final = 1.0 / aligned_disp
                abs_rel, ssim_val = compute_metrics(pred_depth_final, gt_depth, valid)
                if abs_rel is not None:
                    absrel_list.append(abs_rel)
                    ssim_list.append(ssim_val)
        except Exception as e:
            print(f"  ERROR batch {i}: {e}")

    avg_absrel = np.mean(absrel_list) if absrel_list else float("nan")
    avg_ssim   = np.mean(ssim_list)   if ssim_list   else float("nan")

    print(f"\n{'='*45}")
    print(f"  Gen folder:  {args.gen_folder}")
    print(f"  Images:      {len(absrel_list)}  (missing GT: {missing})")
    print(f"  Depth SSIM:  {avg_ssim:.4f}")
    print(f"  AbsRel:      {avg_absrel:.4f}")
    print(f"{'='*45}")


if __name__ == "__main__":
    main()
