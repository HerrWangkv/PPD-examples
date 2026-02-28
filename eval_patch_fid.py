import os, glob, torch, faiss, argparse, cv2, shutil
import numpy as np
from PIL import Image
from torchvision import models, transforms
from torchvision.models import VGG16_Weights
from tqdm import tqdm
from cleanfid import fid
import pyiqa
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

# --- Configuration ---
PATCH_SIZE = 128
TARGET_H = 768
TARGET_W = 1280
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CACHE_DIR = "nusc_val_cache"
CACHE_FEATURES_PATH = os.path.join(CACHE_DIR, "real_features.npy")
CACHE_PATCHES_DIR = os.path.join(CACHE_DIR, "patches")

def get_transform():
    return transforms.Compose([
        transforms.Resize((TARGET_H, TARGET_W)),
        transforms.ToTensor(),
    ])

def get_val_image_paths(dataroot):
    print("Loading nuScenes database...")
    nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=False)
    val_scenes = set(create_splits_scenes()['val'])
    paths = []
    
    for scene in nusc.scene:
        if scene['name'] in val_scenes:
            sample = nusc.get('sample', scene['first_sample_token'])
            while True:
                cam_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
                paths.append(os.path.join(nusc.dataroot, cam_data['filename']))
                if sample['next'] == '': 
                    break
                sample = nusc.get('sample', sample['next'])
    return paths

def extract_and_save(paths, mode, out_dir, max_frames=None, save_patches=True):
    if save_patches:
        os.makedirs(out_dir, exist_ok=True)
        
    vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:24].eval().to(DEVICE)
    tx_vgg = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tx_base = get_transform()
    
    features, saved_paths = [], []
    patch_idx = 0
    
    for path in tqdm(paths, desc=f"Extracting {mode} patches"):
        frames = []
        if mode == "image":
            try:
                frames.append(tx_base(Image.open(path).convert("RGB")))
            except: continue
        else:
            cap = cv2.VideoCapture(path)
            f_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret or (max_frames and f_idx >= max_frames): break
                frames.append(tx_base(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))))
                f_idx += 1
            cap.release()

        if not frames: continue
        t = torch.stack(frames).to(DEVICE)
        
        patches = t.unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(3, PATCH_SIZE, PATCH_SIZE)
        B, C, Hp, Wp, PH, PW = patches.shape
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, C, PH, PW)
        
        for i in range(0, len(patches), 32):
            batch = patches[i:i+32]
            with torch.no_grad():
                features.append(vgg(tx_vgg(batch)).mean([2, 3]).cpu().numpy())
            
            if save_patches:
                for p in batch:
                    p_path = os.path.join(out_dir, f"p_{patch_idx:06d}.png")
                    transforms.ToPILImage()(p.cpu()).save(p_path)
                    saved_paths.append(p_path)
                    patch_idx += 1
                
    return np.concatenate(features, axis=0), saved_paths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_dir", default="nuscenes", help="nuScenes dataroot")
    parser.add_argument("--fake_dir", required=True)
    parser.add_argument("--max_frames", type=int, default=None)
    args = parser.parse_args()

    tmp_fake_all = "tmp_fake_all"
    d_real_fid, d_fake_fid = "tmp_fid_real", "tmp_fid_fake"
    for d in [tmp_fake_all, d_real_fid, d_fake_fid]:
        os.makedirs(d, exist_ok=True)

    # 1. Real Data (Load from Cache or Compute)
    if os.path.exists(CACHE_FEATURES_PATH) and os.path.exists(CACHE_PATCHES_DIR):
        print("\n>>> Step 1: Loading Real Data from Cache...")
        rf = np.load(CACHE_FEATURES_PATH)
        rp = sorted(glob.glob(os.path.join(CACHE_PATCHES_DIR, "*.png")))
    else:
        print("\n>>> Step 1: Extracting Real Data (Will Cache)...")
        os.makedirs(CACHE_DIR, exist_ok=True)
        val_paths = get_val_image_paths(args.real_dir)
        rf, rp = extract_and_save(val_paths, "image", CACHE_PATCHES_DIR, save_patches=True)
        np.save(CACHE_FEATURES_PATH, rf)

    # 2. Fake Data (Compute always)
    print("\n>>> Step 2: Extracting Fake Data...")
    f_paths = sorted(glob.glob(os.path.join(args.fake_dir, "*.mp4")))
    ff, fp = extract_and_save(f_paths, "video", tmp_fake_all, args.max_frames, save_patches=True)

    # 3. Matching
    print("\n>>> Step 3: FAISS Matching...")
    index = faiss.IndexFlatL2(rf.shape[1])
    index.add(rf.astype('float32'))
    _, indices = index.search(ff.astype('float32'), 1)

    # 4. Create Symlinks
    print("\n>>> Step 4: Constructing Paired Dataset...")
    for f_idx, match in enumerate(tqdm(indices, desc="Linking")):
        r_idx = match[0]
        out_name = f"pair_{f_idx:06d}.png"
        os.symlink(os.path.abspath(fp[f_idx]), os.path.join(d_fake_fid, out_name))
        os.symlink(os.path.abspath(rp[r_idx]), os.path.join(d_real_fid, out_name))

    # 5. Compute FID
    print("\n>>> Step 5: Computing FID...")
    score = fid.compute_fid(d_real_fid, d_fake_fid)

    # 6. Cleanup Temporary Fake Data
    for d in [tmp_fake_all, d_real_fid, d_fake_fid]:
        shutil.rmtree(d)

    print(f"\nScore: {score:.4f}")

if __name__ == "__main__":
    main()