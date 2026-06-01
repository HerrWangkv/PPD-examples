"""
Calculate mIoU for Hypersim sim2real experiments using pseudo labels.

Pseudo GT: SegFormer-B5 (ADE20K) predictions on raw Hypersim input images.
           Cached to --cache_dir on first run; reused on subsequent runs.
Pred:      SegFormer-B5 (ADE20K) predictions on translated images.
mIoU measures semantic structure preservation vs the original sim.

Usage:
    source .venv/bin/activate
    python calc_miou_hypersim.py --gen_folder outputs/hypersim/flowedit
    python calc_miou_hypersim.py --gen_folder outputs/hypersim/dropll_J5_r24 --batch_size 8
"""

import argparse
import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

INPUT_DIR  = "outputs/hypersim/input"
CACHE_DIR  = "outputs/hypersim/.pseudo_gt_cache"
MODEL_NAME = "nvidia/segformer-b5-finetuned-ade-640-640"
NUM_CLASSES = 150


def predict_batch(model, processor, images, device):
    """Run segformer on a list of PIL images, return list of H×W uint8 arrays."""
    inputs = processor(images=images, return_tensors="pt",
                       size={"height": 640, "width": 640}).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits  # B × C × H' × W'
    preds = []
    for i, img in enumerate(images):
        p = torch.nn.functional.interpolate(
            logits[i:i+1], size=(img.height, img.width),
            mode="bilinear", align_corners=False
        ).argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)
        preds.append(p)
    return preds


def compute_miou(pred_gt, pred_gen, num_classes):
    """Standard mIoU: average only over classes present in the GT prediction."""
    ious = []
    for cls in range(num_classes):
        if np.sum(pred_gt == cls) == 0:
            continue  # skip classes absent from GT
        tp = np.sum((pred_gt == cls) & (pred_gen == cls))
        fp = np.sum((pred_gt != cls) & (pred_gen == cls))
        fn = np.sum((pred_gt == cls) & (pred_gen != cls))
        denom = tp + fp + fn
        if denom > 0:
            ious.append(tp / denom)
    return float(np.mean(ious)) if ious else 0.0


def get_image_paths(folder):
    exts = ('.jpg', '.jpeg', '.png')
    return sorted(
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.lower().endswith(exts)
    )


def load_or_build_cache(input_dir, cache_dir, model, processor, device, batch_size):
    os.makedirs(cache_dir, exist_ok=True)
    paths = get_image_paths(input_dir)
    missing = [p for p in paths
               if not os.path.exists(os.path.join(cache_dir, os.path.basename(p) + ".npy"))]
    if missing:
        print(f"Building pseudo-GT cache: {len(missing)} images → {cache_dir}")
        for i in tqdm(range(0, len(missing), batch_size), desc="Caching pseudo-GT"):
            batch_paths = missing[i:i+batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            preds = predict_batch(model, processor, images, device)
            for p, pred in zip(batch_paths, preds):
                np.save(os.path.join(cache_dir, os.path.basename(p) + ".npy"), pred)
    else:
        print(f"Pseudo-GT cache complete ({len(paths)} entries), skipping.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_folder", type=str, required=True)
    parser.add_argument("--input_dir",  type=str, default=INPUT_DIR)
    parser.add_argument("--cache_dir",  type=str, default=CACHE_DIR)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
    model = SegformerForSemanticSegmentation.from_pretrained(
        MODEL_NAME, use_safetensors=True
    ).to(device).eval()

    load_or_build_cache(args.input_dir, args.cache_dir,
                        model, processor, device, args.batch_size)

    gen_paths = get_image_paths(args.gen_folder)
    print(f"{len(gen_paths)} images in {args.gen_folder}")

    ious, missing = [], 0
    for i in tqdm(range(0, len(gen_paths), args.batch_size), desc="Evaluating"):
        batch_paths = gen_paths[i:i+args.batch_size]
        images, gt_preds = [], []
        for p in batch_paths:
            cache_p = os.path.join(args.cache_dir, os.path.basename(p) + ".npy")
            if not os.path.exists(cache_p):
                missing += 1
                continue
            images.append(Image.open(p).convert("RGB"))
            gt_preds.append(np.load(cache_p))

        if not images:
            continue
        gen_preds = predict_batch(model, processor, images, device)
        for gt, gen in zip(gt_preds, gen_preds):
            ious.append(compute_miou(gt, gen, NUM_CLASSES))

    miou = float(np.mean(ious)) if ious else 0.0
    print(f"\nImages evaluated: {len(ious)}  (skipped/missing: {missing})")
    print(f"mIoU (pseudo-GT): {miou:.4f}")
    print("=" * 45)
    print(f"mIoU: {miou:.4f}")


if __name__ == "__main__":
    main()
