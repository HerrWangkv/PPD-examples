import os, glob, cv2, tempfile, shutil
import numpy as np
import torch
from pathlib import Path
from cleanfid import fid
from cleanfid.fid import (build_feature_extractor, get_folder_features,
                           get_reference_statistics, frechet_distance, kernel_distance)

REAL_DIR = "/tmp/nuscenes/samples/CAM_FRONT"
STATS_NAME = "nuscenes_camfront_clean"
MODE = "clean"

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
    "dropll_r22_J4":             "outputs/nucarla/dropll_r22_J4",
}


def extract_frames(vid_dir, tmpdir):
    """Extract frames from scene_0000-0059 MP4s into tmpdir."""
    for mp4 in sorted(glob.glob(os.path.join(vid_dir, "scene_????.mp4"))):
        scene_num = int(Path(mp4).stem.split("_")[1])
        if scene_num >= 60:
            continue
        cap = cv2.VideoCapture(mp4)
        scene = Path(mp4).stem
        fidx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(os.path.join(tmpdir, f"{scene}_f{fidx:04d}.jpg"),
                        frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            fidx += 1
        cap.release()
    return len(glob.glob(os.path.join(tmpdir, "*.jpg")))


# --- Cache nuScenes reference stats once ---
if fid.test_stats_exists(STATS_NAME, mode=MODE):
    print(f"[Cache] Using cached stats '{STATS_NAME}'")
else:
    print(f"Computing nuScenes reference stats (one-time, ~20 min)...")
    fid.make_custom_stats(name=STATS_NAME, fdir=REAL_DIR, mode=MODE, batch_size=512)
    print(f"Saved stats '{STATS_NAME}'")

# Load reference stats once (reused across all variants)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feat_model = build_feature_extractor(MODE, device, use_dataparallel=False)
ref_mu, ref_sigma = get_reference_statistics(STATS_NAME, "na", mode=MODE, seed=0, split="custom", metric="FID")
ref_kid_feats = get_reference_statistics(STATS_NAME, "na", mode=MODE, seed=0, split="custom", metric="KID")

import argparse, sys
_p = argparse.ArgumentParser()
_p.add_argument("--variants", nargs="+", default=None)
_args, _ = _p.parse_known_args()

# --- Evaluate each variant ---
results = {}
_active = {k: v for k, v in VARIANTS.items() if _args.variants is None or k in _args.variants}
for name, vid_dir in _active.items():
    print(f"\n=== {name} ===")
    if not os.path.isdir(vid_dir):
        print(f"  Directory not found: {vid_dir}, skipping.")
        continue
    tmpdir = tempfile.mkdtemp(prefix=f"kid_{name}_")
    try:
        n = extract_frames(vid_dir, tmpdir)
        print(f"  Extracted {n} frames, computing FID and KID...")
        # Extract features once, use for both metrics
        np.random.seed(0)
        gen_feats = get_folder_features(tmpdir, feat_model, mode=MODE, device=device, batch_size=512, verbose=True)
        mu = np.mean(gen_feats, axis=0)
        sigma = np.cov(gen_feats, rowvar=False)
        fid_score = frechet_distance(mu, sigma, ref_mu, ref_sigma)
        kid_score = kernel_distance(ref_kid_feats, gen_feats)
        results[name] = (fid_score, kid_score)
        print(f"  FID = {fid_score:.4f}  KID = {kid_score:.6f}")
    finally:
        shutil.rmtree(tmpdir)

print("\n=== RESULTS (sorted by KID) ===")
for name, (f, k) in sorted(results.items(), key=lambda x: x[1][1]):
    print(f"  {name:<30} FID = {f:.4f}  KID = {k:.6f}")
