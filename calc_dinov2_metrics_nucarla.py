"""
DINOv2-space realism metrics for nuCarla variants:
  - FD-DINOv2 (Stein et al., NeurIPS 2023): Frechet distance on DINOv2 CLS embeddings
  - MMD-DINOv2: KID-style polynomial-kernel MMD in the same space
  - Precision / Density (Kynkaanniemi et al. 2019; Naeem et al. 2020): k-NN manifold
    realism axis, k=5

Reference: nuScenes CAM_FRONT (subsampled 10k; +1k holdout for sanity).
Lower FD/MMD = better; higher Precision/Density = better.

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n sim2real_eval python calc_dinov2_metrics_nucarla.py \
        [--variants ditto input ...]
"""

import argparse
import glob
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg
from tqdm import tqdm

REAL_DIR = "/tmp/nuscenes/samples/CAM_FRONT"
REF_CACHE = "nusc_dinov2_vitl14.npy"
HOLDOUT_CACHE = "nusc_dinov2_vitl14_holdout.npy"
EMB_DIR = "dinov2_emb"
N_REAL_REF = 10000
N_REAL_HOLDOUT = 1000
IMG_SIZE = 518
KNN_K = 5

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

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


def preprocess(frames):
    t = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0
    t = F.interpolate(t, size=(IMG_SIZE, IMG_SIZE), mode="bicubic",
                      align_corners=False, antialias=True)
    return (t - IMAGENET_MEAN) / IMAGENET_STD


@torch.no_grad()
def embed_frames(model, frame_iter, total, batch_size=32, desc=""):
    feats, batch = [], []

    def flush():
        if not batch:
            return
        x = preprocess(batch).to(DEVICE)
        feats.append(model(x).cpu().numpy())
        batch.clear()

    for frame in tqdm(frame_iter, total=total, desc=desc, unit="frame", leave=False):
        batch.append(frame)
        if len(batch) >= batch_size:
            flush()
    flush()
    return np.concatenate(feats, axis=0)


def iter_images(paths):
    for p in paths:
        img = cv2.imread(p)
        if img is not None:
            yield cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def iter_videos(video_dir):
    mp4s = sorted(
        f for f in glob.glob(os.path.join(video_dir, "scene_*.mp4"))
        if int(os.path.basename(f)[6:10]) < 60
    )
    for mp4 in mp4s:
        cap = cv2.VideoCapture(mp4)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cap.release()


def frechet(x, y):
    mu1, mu2 = x.mean(0), y.mean(0)
    s1 = np.cov(x, rowvar=False)
    s2 = np.cov(y, rowvar=False)
    covmean, _ = linalg.sqrtm(s1 @ s2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


def mmd_poly(x, y, n_subsets=100, subset_size=1000, rng=None):
    """KID-style unbiased MMD with cubic polynomial kernel (cleanfid convention)."""
    rng = rng or np.random.RandomState(0)
    d = x.shape[1]
    vals = []
    for _ in range(n_subsets):
        xi = x[rng.choice(len(x), min(subset_size, len(x)), replace=False)]
        yi = y[rng.choice(len(y), min(subset_size, len(y)), replace=False)]
        kxx = (xi @ xi.T / d + 1) ** 3
        kyy = (yi @ yi.T / d + 1) ** 3
        kxy = (xi @ yi.T / d + 1) ** 3
        m, n = len(xi), len(yi)
        vals.append(
            (kxx.sum() - np.trace(kxx)) / (m * (m - 1))
            + (kyy.sum() - np.trace(kyy)) / (n * (n - 1))
            - 2 * kxy.mean()
        )
    return float(np.mean(vals))


def precision_density(real, fake, k=KNN_K, block=4096):
    """Kynkaanniemi precision + Naeem density of fake w.r.t. real manifold (GPU)."""
    r = torch.from_numpy(real).to(DEVICE, torch.float32)
    f = torch.from_numpy(fake).to(DEVICE, torch.float32)

    # k-NN radius of each real point (distance to its k-th neighbour)
    radii = torch.empty(len(r), device=DEVICE)
    for i in range(0, len(r), block):
        d = torch.cdist(r[i:i + block], r)
        radii[i:i + block] = d.kthvalue(k + 1, dim=1).values  # +1: self
    # for each fake point: inside any real ball? / how many balls?
    inside_any = torch.zeros(len(f), dtype=torch.bool, device=DEVICE)
    inside_cnt = torch.zeros(len(f), device=DEVICE)
    for i in range(0, len(f), block):
        d = torch.cdist(f[i:i + block], r)        # (b, N_real)
        within = d <= radii.unsqueeze(0)
        inside_any[i:i + block] = within.any(dim=1)
        inside_cnt[i:i + block] = within.sum(dim=1).float()
    precision = inside_any.float().mean().item()
    density = (inside_cnt / k).mean().item()
    return precision, density


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    os.makedirs(EMB_DIR, exist_ok=True)
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14").eval().to(DEVICE)
    print(f"Device: {DEVICE}, DINOv2 ViT-L/14 loaded")

    rng = np.random.RandomState(0)
    all_real = sorted(glob.glob(os.path.join(REAL_DIR, "*.jpg")))
    perm = rng.permutation(len(all_real))

    if os.path.exists(REF_CACHE):
        ref = np.load(REF_CACHE)
    else:
        paths = [all_real[i] for i in perm[:N_REAL_REF]]
        ref = embed_frames(model, iter_images(paths), len(paths), args.batch_size, "real ref")
        np.save(REF_CACHE, ref)
    print(f"reference embeddings: {ref.shape}")

    if os.path.exists(HOLDOUT_CACHE):
        hold = np.load(HOLDOUT_CACHE)
    else:
        paths = [all_real[i] for i in perm[N_REAL_REF:N_REAL_REF + N_REAL_HOLDOUT]]
        hold = embed_frames(model, iter_images(paths), len(paths), args.batch_size, "real holdout")
        np.save(HOLDOUT_CACHE, hold)

    variants = VARIANTS
    if args.variants:
        variants = {k: v for k, v in VARIANTS.items() if k in args.variants}

    results = {}

    def report(name, emb):
        fd = frechet(ref, emb)
        mmd = mmd_poly(ref, emb)
        prec, dens = precision_density(ref, emb)
        results[name] = (fd, mmd, prec, dens)
        print(f"  {name:<28} FD={fd:.2f}  MMD={mmd:.5f}  Prec={prec:.4f}  Dens={dens:.4f}  ({len(emb)})")

    report("REAL_HOLDOUT", hold)

    for name, video_dir in variants.items():
        if not os.path.isdir(video_dir):
            print(f"[{name}] missing {video_dir}, skipping")
            continue
        emb_path = os.path.join(EMB_DIR, f"{name}.npy")
        if os.path.exists(emb_path):
            emb = np.load(emb_path)
        else:
            emb = embed_frames(model, iter_videos(video_dir), None, args.batch_size, name)
            np.save(emb_path, emb)
        report(name, emb)

    for idx, label, rev in [(0, "FD-DINOv2 (lower better)", False),
                            (1, "MMD-DINOv2 (lower better)", False),
                            (2, "Precision (higher better)", True),
                            (3, "Density (higher better)", True)]:
        print(f"\n=== {label} ===")
        for name, vals in sorted(results.items(), key=lambda kv: kv[1][idx], reverse=rev):
            print(f"  {name:<28} {vals[idx]:.5f}")


if __name__ == "__main__":
    main()
