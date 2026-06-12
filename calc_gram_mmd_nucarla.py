"""
Gram-MMD (texture-aware realism, arXiv 2604.03064 style) for nuCarla variants.

Features: VGG16 Gram matrices at relu1_2 / relu2_2 / relu3_3 / relu4_3,
upper-triangular flattened. Linear-kernel MMD between real and fake Gram
distributions (= squared distance of feature means). Lower = more realistic
texture statistics.

Reference: nuScenes CAM_FRONT (subsampled). A held-out real split is scored
as sanity check — it must beat every synthetic variant.

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n sim2real_eval python calc_gram_mmd_nucarla.py \
        [--variants ditto input ...]
"""

import argparse
import glob
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models
from torchvision.models import VGG16_Weights
from tqdm import tqdm

REAL_DIR = "/tmp/nuscenes/samples/CAM_FRONT"
N_REAL_REF = 4000
N_REAL_HOLDOUT = 1000
TARGET_H, TARGET_W = 384, 640
FRAME_STRIDE = 6
LAYERS = {3: "relu1_2", 8: "relu2_2", 15: "relu3_3", 22: "relu4_3"}

VGG_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
VGG_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

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


class GramExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:23].eval()
        for p in self.vgg.parameters():
            p.requires_grad = False
        c = {3: 64, 8: 128, 15: 256, 22: 512}
        self.triu_idx = {k: torch.triu_indices(v, v) for k, v in c.items()}

    @torch.no_grad()
    def forward(self, x):
        """x: (B,3,H,W) normalized. Returns (B, D) concatenated triu Gram vectors."""
        feats = []
        h = x
        for i, layer in enumerate(self.vgg):
            h = layer(h)
            if i in LAYERS:
                B, C, H, W = h.shape
                f = h.reshape(B, C, H * W)
                gram = torch.bmm(f, f.transpose(1, 2)) / (C * H * W)
                iu = self.triu_idx[i]
                feats.append(gram[:, iu[0], iu[1]])
        return torch.cat(feats, dim=1)


def preprocess(frames):
    t = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0
    t = F.interpolate(t, size=(TARGET_H, TARGET_W), mode="bilinear", align_corners=False, antialias=True)
    return (t - VGG_MEAN) / VGG_STD


def gram_mean(extractor, frame_iter, total, batch_size=16, desc=""):
    """Running mean of Gram vectors (linear-kernel MMD only needs means)."""
    acc, n = None, 0
    batch = []

    def flush():
        nonlocal acc, n
        if not batch:
            return
        g = extractor(preprocess(batch).to(DEVICE)).double().sum(dim=0)
        acc = g if acc is None else acc + g
        n += len(batch)
        batch.clear()

    for frame in tqdm(frame_iter, total=total, desc=desc, unit="frame", leave=False):
        batch.append(frame)
        if len(batch) >= batch_size:
            flush()
    flush()
    return (acc / n).cpu().numpy(), n


def iter_images(paths):
    for p in paths:
        img = cv2.imread(p)
        if img is not None:
            yield cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def iter_videos(video_dir, stride):
    mp4s = sorted(
        f for f in glob.glob(os.path.join(video_dir, "scene_*.mp4"))
        if int(os.path.basename(f)[6:10]) < 60
    )
    for mp4 in mp4s:
        cap = cv2.VideoCapture(mp4)
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % stride == 0:
                yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            idx += 1
        cap.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--ref_cache", default="nusc_gram_mean.npy")
    args = parser.parse_args()

    extractor = GramExtractor().to(DEVICE)
    print(f"Device: {DEVICE}")

    rng = np.random.RandomState(0)
    all_real = sorted(glob.glob(os.path.join(REAL_DIR, "*.jpg")))
    perm = rng.permutation(len(all_real))
    ref_paths = [all_real[i] for i in perm[:N_REAL_REF]]
    holdout_paths = [all_real[i] for i in perm[N_REAL_REF:N_REAL_REF + N_REAL_HOLDOUT]]

    if os.path.exists(args.ref_cache):
        mu_ref = np.load(args.ref_cache)
        print(f"[Cache] real reference Gram mean loaded ({mu_ref.shape})")
    else:
        mu_ref, n = gram_mean(extractor, iter_images(ref_paths), len(ref_paths), desc="real ref")
        np.save(args.ref_cache, mu_ref)
        print(f"Real reference: {n} frames, cached -> {args.ref_cache}")

    results = {}

    # sanity: held-out real frames
    holdout_cache = args.ref_cache.replace(".npy", "_holdout.npy")
    if os.path.exists(holdout_cache):
        mu_h = np.load(holdout_cache)
    else:
        mu_h, _ = gram_mean(extractor, iter_images(holdout_paths), len(holdout_paths), desc="real holdout")
        np.save(holdout_cache, mu_h)
    results["REAL_HOLDOUT"] = float(np.sum((mu_ref - mu_h) ** 2)) * 1e3

    variants = VARIANTS
    if args.variants:
        variants = {k: v for k, v in VARIANTS.items() if k in args.variants}

    for name, video_dir in variants.items():
        if not os.path.isdir(video_dir):
            print(f"[{name}] missing {video_dir}, skipping")
            continue
        mu, n = gram_mean(extractor, iter_videos(video_dir, FRAME_STRIDE), None, desc=name)
        score = float(np.sum((mu_ref - mu) ** 2)) * 1e3
        results[name] = score
        print(f"  {name:<28} GramMMD = {score:.4f}  ({n} frames)")

    print("\n=== Gram-MMD x1e3 (sorted, lower = better) ===")
    for name, s in sorted(results.items(), key=lambda kv: kv[1]):
        print(f"  {name:<28} {s:.4f}")


if __name__ == "__main__":
    main()
