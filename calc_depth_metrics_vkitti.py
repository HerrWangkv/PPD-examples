"""
Calculate Depth SSIM and AbsRel for vKITTI sim2real experiments.

GT:    vKITTI depthgt (uint16, cm, 65535=invalid)
Model: Depth Anything V2 Large (relative disparity → align via least squares)
Input: Flat folder of translated images named {scene}_{cond}_{frame}.png

Usage:
    source .venv/bin/activate
    python calc_depth_metrics_vkitti.py --gen_folder outputs/vkitti/flowedit
    python calc_depth_metrics_vkitti.py --gen_folder outputs/vkitti/baseline_newprompt
"""

import argparse
import os
import copy
import numpy as np
import torch
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

DEPTHGT_DIR = "/mrtstorage/datasets_tmp/vkitti/vkitti_1.3.1_depthgt"


def parse_filename(name):
    stem = os.path.splitext(name)[0]
    parts = stem.split('_')
    return parts[0], '_'.join(parts[1:-1]), parts[-1]


def load_gt_depth(path):
    """uint16 cm → float32 metres, valid mask (65535 = invalid)."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, None
    valid = (img > 0) & (img != 65535)
    depth_m = img.astype(np.float32) / 100.0
    return depth_m, valid


def align_least_squares(pred, target, mask):
    mp, mt = pred[mask], target[mask]
    if len(mp) == 0:
        return pred
    A = np.vstack([mp, np.ones_like(mp)]).T
    (s, t), *_ = np.linalg.lstsq(A, mt, rcond=None)
    return pred * s + t


def _load_depth_item(args):
    """Worker: load GT depth. Returns (gen_path, gt_depth, valid, gt_hw) or None."""
    fname, gen_folder, depthgt_dir = args
    stem = os.path.splitext(fname)[0]
    parts = stem.split('_')
    scene, cond, frame = parts[0], '_'.join(parts[1:-1]), parts[-1]
    gt_path = os.path.join(depthgt_dir, scene, cond, f"{frame}.png")
    if not os.path.exists(gt_path):
        return None
    gt_depth, valid = load_gt_depth(gt_path)
    if gt_depth is None:
        return None
    gen_path = os.path.join(gen_folder, fname)
    gt_hw = gt_depth.shape[:2]
    return (gen_path, gt_depth, valid, gt_hw)


def compute_metrics(pred_depth, gt_depth, valid):
    p, g = pred_depth[valid], gt_depth[valid]
    if len(p) == 0:
        return None, None
    absrel = np.mean(np.abs(p - g) / g)
    data_range = g.max() - g.min() or 1.0
    ssim_val, _ = ssim(pred_depth, gt_depth, full=True,
                        data_range=data_range, win_size=7)
    return absrel, ssim_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_folder', type=str,
                        default='/mrtstorage/users/kwang/vkitti_translated/flowedit')
    parser.add_argument('--model_id', type=str,
                        default='depth-anything/Depth-Anything-V2-Large-hf')
    parser.add_argument('--clone_only', action='store_true',
                        help='Only evaluate clone-condition images ({scene}_clone_{frame}.png)')
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading {args.model_id} on {device}...")
    processor = AutoImageProcessor.from_pretrained(args.model_id)
    model = AutoModelForDepthEstimation.from_pretrained(args.model_id).to(device).eval()

    gen_files = sorted(f for f in os.listdir(args.gen_folder)
                       if f.lower().endswith(('.png', '.jpg'))
                       and (not args.clone_only or '_clone_' in f))
    print(f"{len(gen_files)} images in {args.gen_folder}")

    # Parallel GT loading
    missing = 0
    valid_items = []
    worker_args = [(f, args.gen_folder, DEPTHGT_DIR) for f in gen_files]
    # valid_items: (gen_path, gt_depth, valid, gt_hw)  — use GT shape for pred interpolation
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_load_depth_item, a): a[0] for a in worker_args}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Loading GT"):
            result = fut.result()
            if result is None:
                missing += 1
            else:
                valid_items.append(result)
    # Restore sorted order
    fname_order = {f: i for i, f in enumerate(gen_files)}
    valid_items.sort(key=lambda x: fname_order[os.path.basename(x[0])])

    absrel_list, ssim_list = [], []
    debug_saved = False
    BS = args.batch_size

    for i in tqdm(range(0, len(valid_items), BS), desc="Inference"):
        batch = valid_items[i:i+BS]
        imgs_bgr = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
                    for p, *_ in batch]

        inputs = processor(images=imgs_bgr, return_tensors='pt').to(device)
        with torch.no_grad():
            preds = model(**inputs).predicted_depth  # (B, H', W')

        for j, (gen_path, gt_depth, valid, gt_hw) in enumerate(batch):
            gh, gw = gt_hw
            pred_disp = torch.nn.functional.interpolate(
                preds[j:j+1].unsqueeze(1), size=(gh, gw),
                mode='bicubic', align_corners=False,
            ).squeeze().cpu().numpy()

            gt_disp = np.zeros_like(gt_depth)
            gt_disp[valid] = 1.0 / (gt_depth[valid] + 1e-6)
            aligned_disp = align_least_squares(pred_disp, gt_disp, valid)
            aligned_disp = np.clip(aligned_disp, 1e-3, None)
            pred_depth = 1.0 / aligned_disp

            absrel, ssim_val = compute_metrics(pred_depth, gt_depth, valid)
            if absrel is None:
                continue
            absrel_list.append(absrel)
            ssim_list.append(ssim_val)

            if not debug_saved:
                viz_p = np.where(valid, pred_depth, np.nan)
                viz_g = np.where(valid, gt_depth, np.nan)
                vmin, vmax = np.nanmin(viz_g), np.nanpercentile(viz_g, 95)
                cmap = copy.copy(plt.cm.get_cmap('magma_r'))
                cmap.set_bad('white')
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                axes[0].imshow(viz_p, cmap=cmap, vmin=vmin, vmax=vmax)
                axes[0].set_title(f'Pred depth  AbsRel={absrel:.3f}')
                axes[1].imshow(viz_g, cmap=cmap, vmin=vmin, vmax=vmax)
                axes[1].set_title('GT depth')
                plt.tight_layout()
                plt.savefig('debug_depth_vkitti.png')
                debug_saved = True

    if not absrel_list:
        print("No valid comparisons.")
        return

    print(f"\n{'='*45}")
    print(f"  Gen folder:  {args.gen_folder}")
    print(f"  Images:      {len(absrel_list)}  (missing GT: {missing})")
    print(f"  Depth SSIM:  {np.mean(ssim_list):.4f}")
    print(f"  AbsRel:      {np.mean(absrel_list):.4f}")
    print(f"{'='*45}")


if __name__ == '__main__':
    main()
