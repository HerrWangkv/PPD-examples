"""
CMMD (CLIP Maximum Mean Discrepancy) for nuCarla translated variants.

Follows the official protocol (Jayasumana et al., "Rethinking FID", CVPR 2024;
reference impl: google-research/cmmd, pytorch port: sayakpaul/cmmd-pytorch):
  - CLIP ViT-L/14@336 projected image embeddings, L2-normalized
  - bicubic resize to 336x336 (no center crop)
  - MMD with Gaussian RBF kernel, sigma=10, scale=1000 (biased estimator)

Reference set: nuScenes CAM_FRONT frames (same as calc_kid_nucarla.py) so the
only change vs KID is the feature space (CLIP vs Inception). Lower = better.

Usage:
    CUDA_VISIBLE_DEVICES=2 conda run -n sim2real_eval python calc_cmmd_nucarla.py
"""

import argparse
import glob
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import CLIPVisionModelWithProjection

REAL_DIR = "/tmp/nuscenes/samples/CAM_FRONT"
REAL_CACHE = "nusc_camfront_cmmd_clip_vitl336.npy"
CLIP_NAME = "openai/clip-vit-large-patch14-336"
IMG_SIZE = 336
SIGMA = 10.0
SCALE = 1000.0

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

VARIANTS = {
    "input":                     "outputs/nucarla/input/rgb",
    "cosmos":                    "outputs/nucarla/cosmos",
    "ditto":                     "outputs/nucarla/ditto",
    "ppd_r30":                   "outputs/nucarla/ppd/flux_30_wan_30",
    "wavelet_r30":               "outputs/nucarla/wavelet/flux_30_30_1_wan_30_30_1",
    "vace_gray":                 "outputs/nucarla/vace_gray",
    "dnaedit":                   "outputs/nucarla/dnaedit",
    "cosmos_depth_edge":         "outputs/nucarla/cosmos_depth_edge_imgs",
    "cosmos_depth_seg_vis_edge": "outputs/nucarla/cosmos_depth_seg_vis_edge_imgs",
    "dropll_r30_J5":             "outputs/nucarla/wavelet/dropll_r30_J5",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def preprocess(frames_rgb_uint8):
    """frames: list of HxWx3 uint8 RGB arrays -> (B,3,336,336) normalized tensor."""
    t = torch.from_numpy(np.stack(frames_rgb_uint8)).permute(0, 3, 1, 2).float() / 255.0
    t = F.interpolate(t, size=(IMG_SIZE, IMG_SIZE), mode="bicubic", align_corners=False, antialias=True)
    return (t - CLIP_MEAN) / CLIP_STD


@torch.no_grad()
def embed_batch(model, frames):
    pixel_values = preprocess(frames).to(DEVICE)
    embs = model(pixel_values=pixel_values).image_embeds
    embs = embs / torch.linalg.norm(embs, axis=-1, keepdims=True)
    return embs.cpu().numpy()


def embed_images(model, paths, batch_size=64):
    feats = []
    batch = []
    for p in tqdm(paths, desc="real images"):
        img = cv2.imread(p)
        if img is None:
            continue
        batch.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if len(batch) >= batch_size:
            feats.append(embed_batch(model, batch))
            batch = []
    if batch:
        feats.append(embed_batch(model, batch))
    return np.concatenate(feats, axis=0)


def embed_videos(model, video_dir, batch_size=64):
    mp4s = sorted(
        f for f in glob.glob(os.path.join(video_dir, "scene_*.mp4"))
        if int(os.path.basename(f)[6:10]) < 60
    )
    feats = []
    batch = []
    for mp4 in tqdm(mp4s, desc=os.path.basename(video_dir.rstrip("/")), unit="video"):
        cap = cv2.VideoCapture(mp4)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            batch.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if len(batch) >= batch_size:
                feats.append(embed_batch(model, batch))
                batch = []
        cap.release()
    if batch:
        feats.append(embed_batch(model, batch))
    return np.concatenate(feats, axis=0)


def mmd(x, y, block=4096):
    """Biased MMD estimator with Gaussian RBF kernel (sigma=10), blockwise on GPU."""
    x = torch.from_numpy(x).to(DEVICE, torch.float32)
    y = torch.from_numpy(y).to(DEVICE, torch.float32)
    gamma = 1.0 / (2 * SIGMA**2)

    def mean_kernel(a, b):
        total = 0.0
        for i in range(0, a.shape[0], block):
            ai = a[i:i + block]
            for j in range(0, b.shape[0], block):
                bj = b[j:j + block]
                d2 = torch.cdist(ai, bj).pow(2)
                total += torch.exp(-gamma * d2).sum().item()
        return total / (a.shape[0] * b.shape[0])

    return SCALE * (mean_kernel(x, x) + mean_kernel(y, y) - 2 * mean_kernel(x, y))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    model = CLIPVisionModelWithProjection.from_pretrained(CLIP_NAME).eval().to(DEVICE)

    # Real reference embeddings (cached)
    if os.path.exists(REAL_CACHE):
        real = np.load(REAL_CACHE)
        print(f"[Cache] Loaded real embeddings: {real.shape}")
    else:
        paths = sorted(glob.glob(os.path.join(REAL_DIR, "*.jpg")))
        print(f"Embedding {len(paths)} real frames...")
        real = embed_images(model, paths, args.batch_size)
        np.save(REAL_CACHE, real)
        print(f"Saved cache: {REAL_CACHE} {real.shape}")

    variants = VARIANTS
    if args.variants:
        variants = {k: v for k, v in VARIANTS.items() if k in args.variants}

    results = {}
    for name, video_dir in variants.items():
        if not os.path.isdir(video_dir):
            print(f"[{name}] missing dir {video_dir}, skipping")
            continue
        fake = embed_videos(model, video_dir, args.batch_size)
        score = mmd(real, fake)
        results[name] = score
        print(f"  {name:<28} CMMD = {score:.4f}  ({fake.shape[0]} frames)")

    print("\n=== CMMD Results (sorted, lower = better) ===")
    for name, score in sorted(results.items(), key=lambda kv: kv[1]):
        print(f"  {name:<28} {score:.4f}")


if __name__ == "__main__":
    main()
