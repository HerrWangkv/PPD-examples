import argparse
import os
import cv2
import torch
import numpy as np
import glob
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import copy

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate Depth SSIM and ABSREL (Least Squares Disparity Alignment)")
    parser.add_argument(
        "--gen_folder", 
        type=str, 
        required=True, 
        help="Path to your generated/adapted images (RGB)"
    )
    parser.add_argument(
        "--gt_folder", 
        type=str, 
        default="data/synthia",
        help="Path to SynthIA Ground Truth Depth folder"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="depth-anything/Depth-Anything-V2-Large-hf",
        help="HuggingFace model ID for depth estimation"
    )
    parser.add_argument(
        "--synthia_packed",
        action="store_true",
        help="Add this flag ONLY if GT is RGB-packed. Do not use for standard 16-bit cm format."
    )
    return parser.parse_args()

def get_image_paths(folder):
    """Recursively finds all image paths."""
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    image_paths = []
    
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_paths.append(os.path.join(root, file))
    return sorted(image_paths)

def load_gt_depth(path, use_packed=False):
    """Loads SynthIA Ground Truth depth."""
    if not os.path.exists(path): return None, None
    
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None: return None, None

    # Handle 3-channel grayscale
    if img.ndim == 3 and not use_packed:
        img = img[:, :, 0]

    if use_packed:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        R, G, B = img[:,:,0].astype(np.float32), img[:,:,1].astype(np.float32), img[:,:,2].astype(np.float32)
        depth_m = (R + G * 256.0 + B * 256.0 * 256.0) 
        valid_mask = depth_m < 1e9 
    else:
        valid_mask = (img > 0) & (img != 65535)
        depth_m = img.astype(np.float32) / 100.0

    return depth_m, valid_mask

def predict_depth(model, processor, image_path, device):
    """Predicts relative disparity (inverse depth) using Depth Anything V2."""
    image = cv2.imread(image_path)
    if image is None: return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Preprocess
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth
    
    # Interpolate to original size
    h, w = image.shape[:2]
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=(h, w),
        mode="bicubic",
        align_corners=False,
    )
    
    return prediction.squeeze().cpu().numpy()

def align_least_squares(pred, target, mask):
    """
    Aligns prediction to target using Least Squares (Standard Academic Protocol).
    Solves for scale (s) and shift (t) such that: s * pred + t = target
    Minimizes the error across all pixels.
    """
    masked_pred = pred[mask]
    masked_target = target[mask]
    
    if len(masked_pred) == 0: return pred

    # Stack for linear regression: [pred, 1]
    # We want to solve: A * [s, t]^T = target
    ones = np.ones_like(masked_pred)
    A = np.vstack([masked_pred, ones]).T
    
    # Solve linear system
    (s, t), _, _, _ = np.linalg.lstsq(A, masked_target, rcond=None)
    
    # Apply alignment
    aligned_pred = pred * s + t
    
    return aligned_pred

def compute_metrics(pred, gt, mask):
    """Calculates ABSREL and Depth SSIM."""
    pred_masked = pred[mask]
    gt_masked = gt[mask]
    
    if len(pred_masked) == 0: return None, None

    # 1. ABSREL
    abs_rel = np.mean(np.abs(pred_masked - gt_masked) / gt_masked)
    
    # 2. SSIM
    data_range = gt_masked.max() - gt_masked.min()
    if data_range == 0: data_range = 1.0
    
    # Use smaller window size for depth maps
    score, _ = ssim(pred, gt, full=True, data_range=data_range, win_size=7, channel_axis=None)
    
    return abs_rel, score

def main():
    args = parse_args()
    if os.path.exists(os.path.join(args.gen_folder, "RGB")):
        args.gen_folder = os.path.join(args.gen_folder, "RGB")
    if os.path.exists(os.path.join(args.gt_folder, "Depth")):
        args.gt_folder = os.path.join(args.gt_folder, "Depth", "Depth")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Scan Folders
    print(f"Scanning generated folder: {args.gen_folder}")
    gen_paths = get_image_paths(args.gen_folder)
    print(f"Found {len(gen_paths)} images.")

    # 2. Load Model
    print(f"Loading Depth Model ({args.model_id})...")
    processor = AutoImageProcessor.from_pretrained(args.model_id)
    model = AutoModelForDepthEstimation.from_pretrained(args.model_id).to(device)
    model.eval()

    print(f"--------------------------------------------------")
    print(f"Metric: Depth Structural Alignment")
    print(f"Strategy: Least Squares Alignment in Disparity Space")
    print(f"GT Unit: {'RGB Packed' if args.synthia_packed else 'Centimeters -> Meters'}")
    print(f"--------------------------------------------------")

    absrel_list = []
    ssim_list = []
    missing_gt_count = 0

    print("Calculating Depth Metrics...")
    for gen_path in tqdm(gen_paths):
        filename = os.path.basename(gen_path)
        if not filename.endswith(".png"):
            gt_path = os.path.join(args.gt_folder, filename.rsplit(".", 1)[0] + ".png")
        else:
            gt_path = os.path.join(args.gt_folder, filename)
        
        # 1. Load GT Depth (Meters)
        gt_depth, valid_mask = load_gt_depth(gt_path, use_packed=args.synthia_packed)
        if gt_depth is None: 
            missing_gt_count += 1
            continue

        # 2. Predict (Raw output is proportional to Disparity: High=Close)
        pred_disp_raw = predict_depth(model, processor, gen_path, device)
        if pred_disp_raw is None: continue

        # 3. Resize Prediction to match GT resolution
        gt_h, gt_w = gt_depth.shape
        if pred_disp_raw.shape != (gt_h, gt_w):
            pred_disp = cv2.resize(pred_disp_raw, (gt_w, gt_h), interpolation=cv2.INTER_CUBIC)
        else:
            pred_disp = pred_disp_raw

        # === CORE LOGIC: ALIGN IN DISPARITY SPACE ===
        
        # A. Convert GT Depth to GT Disparity
        #    GT Disparity: High Value = Close (same as Model Output)
        gt_disp = np.zeros_like(gt_depth)
        gt_disp[valid_mask] = 1.0 / (gt_depth[valid_mask] + 1e-6)

        # B. Align Raw Prediction to GT Disparity
        #    Use LEAST SQUARES for optimal fit
        aligned_disp = align_least_squares(pred_disp, gt_disp, valid_mask)

        # C. Invert back to Depth Space
        #    Clip minimum disparity to avoid exploding depth values
        min_disp = 1e-3 
        aligned_disp[aligned_disp < min_disp] = min_disp
        
        pred_depth_final = 1.0 / aligned_disp
        # ============================================

        # 4. Compute Metrics (compare Depths)
        abs_rel, ssim_val = compute_metrics(pred_depth_final, gt_depth, valid_mask)
        
        if abs_rel is not None:
            absrel_list.append(abs_rel)
            ssim_list.append(ssim_val)

        # DEBUG: Visualization (First valid image only)
        if len(absrel_list) == 1:
            viz_pred = pred_depth_final.copy()
            viz_gt = gt_depth.copy()
            viz_pred[~valid_mask] = np.nan
            viz_gt[~valid_mask] = np.nan
            
            vmin = np.nanmin(viz_gt)
            vmax = np.nanpercentile(viz_gt, 95)
            
            plt.figure(figsize=(15, 6))
            cmap = copy.copy(plt.cm.get_cmap("magma_r")) 
            cmap.set_bad(color='white')
            
            plt.subplot(1, 2, 1)
            plt.title(f"Aligned Pred (Depth)\nABSREL: {abs_rel:.3f}")
            plt.imshow(viz_pred, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
            plt.colorbar(label="Depth (m)")
            
            plt.subplot(1, 2, 2)
            plt.title("Ground Truth (Depth)")
            plt.imshow(viz_gt, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
            plt.colorbar(label="Depth (m)")
            
            plt.tight_layout()
            plt.savefig("debug_final_check.png")
            print("\n[DEBUG] Saved 'debug_final_check.png'. Check alignment!")

    # 5. Results
    if len(absrel_list) == 0:
        print("No valid comparisons made.")
        return

    avg_absrel = np.mean(absrel_list)
    avg_ssim = np.mean(ssim_list)

    print(f"\n================ RESULTS ================")
    print(f" Images Processed: {len(absrel_list)}")
    print(f" Missing GT:       {missing_gt_count}")
    print(f" ---------------------------------------")
    print(f" Average Depth SSIM: {avg_ssim:.4f} (Target: > 0.8)")
    print(f" Average ABSREL:     {avg_absrel:.4f} (Target: < 0.2)")
    print(f"=========================================")

if __name__ == "__main__":
    main()