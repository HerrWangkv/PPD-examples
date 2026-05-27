"""
Calculate mIoU for vKITTI sim2real experiments.

Oracle: SegFormer-B5 trained on Cityscapes (same class taxonomy as KITTI semantics).
GT:     vKITTI scenegt RGB-encoded labels, mapped to Cityscapes trainIDs.
Input:  Flat folder of translated images named {scene}_{cond}_{frame}.png

Usage:
    source .venv/bin/activate
    python calc_miou_vkitti.py --gen_folder outputs/vkitti/flowedit
    python calc_miou_vkitti.py --gen_folder outputs/vkitti/baseline_newprompt
"""

import argparse
import os
import numpy as np
import torch
import cv2
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from prettytable import PrettyTable

SCENEGT_DIR = "/mrtstorage/datasets_tmp/vkitti/vkitti_1.3.1_scenegt"

CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle',
]

# vKITTI category prefix → Cityscapes trainID (255 = ignore)
VKITTI_TO_CITYSCAPES = {
    'Road':         0,
    'Building':     2,
    'GuardRail':    4,
    'Pole':         5,
    'TrafficLight': 6,
    'TrafficSign':  7,
    'Tree':         8,
    'Vegetation':   8,
    'Terrain':      9,
    'Sky':          10,
    'Car':          13,
    'Van':          13,
    'Truck':        14,
    'Misc':         255,
}


def load_encoding(scene, cond):
    """Build (r,g,b) → trainID dict from scenegt encoding txt."""
    txt = os.path.join(SCENEGT_DIR, f"{scene}_{cond}_scenegt_rgb_encoding.txt")
    if not os.path.exists(txt):
        txt = os.path.join(SCENEGT_DIR, f"{scene}_clone_scenegt_rgb_encoding.txt")
    rgb_to_tid = {}
    with open(txt) as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            cat_base = parts[0].split(':')[0]
            r, g, b = int(parts[1]), int(parts[2]), int(parts[3])
            rgb_to_tid[(r, g, b)] = VKITTI_TO_CITYSCAPES.get(cat_base, 255)
    return rgb_to_tid


def decode_semantic(seg_path, rgb_to_tid):
    """RGB-encoded semantic PNG → Cityscapes trainID mask (vectorized)."""
    img = cv2.cvtColor(cv2.imread(seg_path), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    # Pack RGB into a single uint32 key for O(1) lookup
    flat = img.reshape(-1, 3).astype(np.uint32)
    keys = (flat[:, 0] << 16) | (flat[:, 1] << 8) | flat[:, 2]
    lut = {((r << 16) | (g << 8) | b): tid for (r, g, b), tid in rgb_to_tid.items()}
    label = np.full(h * w, 255, dtype=np.uint8)
    for key, tid in lut.items():
        label[keys == key] = tid
    return label.reshape(h, w)


def _load_gt_item(args):
    """Worker: load and decode one GT mask. Returns (fname, gt_mask) or None."""
    fname, gen_folder, scenegt_dir, enc_cache_items = args
    scene, cond, frame = os.path.splitext(fname)[0].split('_', 1)[0], \
        '_'.join(os.path.splitext(fname)[0].split('_')[1:-1]), \
        os.path.splitext(fname)[0].split('_')[-1]
    gt_path = os.path.join(scenegt_dir, scene, cond, f"{frame}.png")
    if not os.path.exists(gt_path):
        return None
    # enc_cache_items passed as list of ((scene,cond), rgb_to_tid)
    rgb_to_tid = dict(enc_cache_items).get((scene, cond))
    if rgb_to_tid is None:
        return None
    gt_mask = decode_semantic(gt_path, rgb_to_tid)
    return (fname, gt_mask)


def parse_filename(name):
    """'0001_clone_00000.png' → ('0001', 'clone', '00000')"""
    stem = os.path.splitext(name)[0]
    parts = stem.split('_')
    return parts[0], '_'.join(parts[1:-1]), parts[-1]


class Evaluator:
    def __init__(self, n=19):
        self.cm = np.zeros((n, n))
        self.n = n

    def update(self, pred, gt):
        valid = (gt >= 0) & (gt < self.n)
        idx = self.n * gt[valid].astype(int) + pred[valid]
        self.cm += np.bincount(idx, minlength=self.n ** 2).reshape(self.n, self.n)

    def iou(self):
        inter = np.diag(self.cm)
        union = self.cm.sum(1) + self.cm.sum(0) - inter
        return inter / (union + 1e-10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_folder', type=str,
                        default='/mrtstorage/users/kwang/vkitti_translated/flowedit')
    parser.add_argument('--clone_only', action='store_true',
                        help='Only evaluate clone-condition images ({scene}_clone_{frame}.png)')
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading SegFormer-B5 (Cityscapes) on {device}...")
    model_name = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
    processor = SegformerImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_name, use_safetensors=True).to(device).eval()

    gen_files = sorted(f for f in os.listdir(args.gen_folder)
                       if f.lower().endswith(('.png', '.jpg'))
                       and (not args.clone_only or '_clone_' in f))
    print(f"{len(gen_files)} images in {args.gen_folder}")

    evaluator = Evaluator()
    missing = 0

    # Pre-load all encodings (one per scene/cond combination)
    enc_cache = {}
    for fname in gen_files:
        scene, cond, _ = parse_filename(fname)
        key = (scene, cond)
        if key not in enc_cache:
            enc_cache[key] = load_encoding(scene, cond)

    # Parallel GT loading
    enc_cache_items = list(enc_cache.items())
    worker_args = [(f, args.gen_folder, SCENEGT_DIR, enc_cache_items) for f in gen_files]
    valid_items = []  # (fname, gt_mask) in original order
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_load_gt_item, a): a[0] for a in worker_args}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Loading GT"):
            result = fut.result()
            if result is None:
                missing += 1
            else:
                valid_items.append(result)
    # Restore sorted order for reproducibility
    fname_order = {f: i for i, f in enumerate(gen_files)}
    valid_items.sort(key=lambda x: fname_order[x[0]])

    # Batched inference
    BS = args.batch_size
    for i in tqdm(range(0, len(valid_items), BS), desc="Inference"):
        batch = valid_items[i:i+BS]
        images = [Image.open(os.path.join(args.gen_folder, f)).convert('RGB')
                  for f, _ in batch]
        gt_masks = [gt for _, gt in batch]

        inputs = processor(images=images, return_tensors='pt').to(device)
        with torch.no_grad():
            logits = model(**inputs).logits  # (B, C, H/4, W/4)

        for j, (image, gt_mask) in enumerate(zip(images, gt_masks)):
            pred_mask = torch.nn.functional.interpolate(
                logits[j:j+1], size=image.size[::-1], mode='bilinear', align_corners=False,
            ).argmax(1)[0].cpu().numpy()

            if gt_mask.shape != pred_mask.shape:
                gt_mask = cv2.resize(gt_mask, pred_mask.shape[::-1],
                                     interpolation=cv2.INTER_NEAREST)
            evaluator.update(pred_mask, gt_mask)

    iou = evaluator.iou()
    present = iou > 0
    miou = np.mean(iou[present])

    table = PrettyTable(['Class', 'IoU (%)'])
    for i, cls in enumerate(CLASSES):
        table.add_row([cls, f'{iou[i]*100:.2f}'])
    print(table)
    print(f"\nmIoU: {miou*100:.2f}%  (missing GT: {missing})")


if __name__ == '__main__':
    main()
