import os, glob, torch, faiss
import numpy as np
from PIL import Image
from torchvision import models, transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# --- 配置区 ---
REAL_DIR = "/mrtstorage/users/kwang/navsim_test_camera_front"
FAKE_DIR = "/mrtstorage/users/kwang/my_long_tail_sim_dataset"
OUT_REAL, OUT_FAKE = "tmp/matched_real", "tmp/matched_fake"
PATCH_SIZE, TARGET_H, TARGET_W = 256, 704, 1280
BATCH_SIZE = 32  # 视显存大小调整
NUM_WORKERS = 8  # CPU 核心数
device = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUT_REAL, exist_ok=True)
os.makedirs(OUT_FAKE, exist_ok=True)

# --- 1. 数据并行加载 ---
class ImageDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths
        self.transform = transforms.Compose([
            transforms.Resize((TARGET_H, TARGET_W)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        return self.transform(Image.open(self.paths[idx]).convert("RGB")), self.paths[idx]

def get_features_fast(paths, desc="Processing"):
    # 1. Setup Model
    vgg = models.vgg16(weights="DEFAULT").features[:24].eval()
    if torch.cuda.device_count() > 1:
        vgg = torch.nn.DataParallel(vgg)
    vgg = vgg.to(device)
    
    loader = DataLoader(ImageDataset(paths), batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
    features, infos = [], []
    
    with torch.no_grad():
        for imgs, img_paths in tqdm(loader, desc=desc):
            imgs = imgs.to(device) # Shape: [B, 3, 704, 1280]
            
            # --- Key Optimization: Create a Batch of Patches ---
            batch_patches = []
            batch_infos = []
            
            for b in range(imgs.shape[0]):
                for top in range(0, TARGET_H - PATCH_SIZE + 1, PATCH_SIZE):
                    for left in range(0, TARGET_W - PATCH_SIZE + 1, PATCH_SIZE):
                        p = imgs[b, :, top:top+PATCH_SIZE, left:left+PATCH_SIZE]
                        batch_patches.append(p)
                        batch_infos.append((img_paths[b], top, left))
            
            # Convert list of patches to a single large tensor
            # Shape: [B * 10, 3, 256, 256]
            patch_tensor = torch.stack(batch_patches) 
            
            # Now DataParallel can split this large tensor across 4 GPUs!
            # e.g., if B=32, patch_tensor has 320 patches. Each GPU gets 80.
            feats = vgg(patch_tensor).mean([2, 3]) 
            
            features.append(feats.cpu().numpy())
            infos.extend(batch_infos)
            
    return np.concatenate(features, axis=0), infos

# --- 2. 并行保存逻辑 ---
def save_worker(args):
    img_path, top, left, out_path = args
    img = Image.open(img_path).convert("RGB")
    img = TF.resize(img, (TARGET_H, TARGET_W))
    TF.crop(img, top, left, PATCH_SIZE, PATCH_SIZE).save(out_path)

def main():
    print("Scanning...")
    real_paths = glob.glob(os.path.join(REAL_DIR, "*", "CAM_F0", "*.*"))
    fake_paths = glob.glob(os.path.join(FAKE_DIR, "*", "rgb", "*.*"))
    
    # 尝试加载已保存的特征和信息
    import pickle
    if os.path.exists("tmp/real_feats.npy") and os.path.exists("tmp/fake_feats.npy") and os.path.exists("infos.pkl"):
        print("Loading features and infos from disk...")
        rf = np.load("tmp/real_feats.npy")
        ff = np.load("tmp/fake_feats.npy")
        with open("infos.pkl", "rb") as f:
            ri, fi = pickle.load(f)
    else:
        # 提取特征
        rf, ri = get_features_fast(real_paths, "Real Patches")
        ff, fi = get_features_fast(fake_paths, "Fake Patches")
        np.save("tmp/real_feats.npy", rf)
        np.save("tmp/fake_feats.npy", ff)
        with open("infos.pkl", "wb") as f:
            pickle.dump((ri, fi), f)
        print("Features saved to disk. You can now run the matching script.")
    # FAISS 匹配 (如果有 GPU 则自动加速)
    print("Matching...")
    index = faiss.IndexFlatL2(rf.shape[1])
    if faiss.get_num_gpus() > 0:
        index = faiss.index_cpu_to_all_gpus(index)
    index.add(rf)
    
    # 将 ff (fake features) 分成较小的块进行搜索
    search_batch_size = 100 
    all_indices = []
    
    for i in range(0, len(ff), search_batch_size):
        batch_ff = ff[i:i + search_batch_size]
        _, indices = index.search(batch_ff, 1)
        all_indices.append(indices)
    
    indices = np.vstack(all_indices)

    # 并行保存
    print("Parallel Saving...")
    tasks = []
    for i, f_info in enumerate(fi):
        r_info = ri[indices[i][0]]
        tasks.append((f_info[0], f_info[1], f_info[2], f"{OUT_FAKE}/{i}.jpg"))
        tasks.append((r_info[0], r_info[1], r_info[2], f"{OUT_REAL}/{i}.jpg"))

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        list(tqdm(executor.map(save_worker, tasks), total=len(tasks), desc="Saving"))

if __name__ == "__main__":
    main()