import argparse
import os
import shutil
import tempfile
from cleanfid import fid

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate FID & KID (Auto-flatten folders)")
    parser.add_argument("--gen_folder", type=str, required=True, help="Path to generated images")
    parser.add_argument("--real_folder", type=str, default="data/cityscapes", help="Path to real Cityscapes images")
    parser.add_argument("--dataset_name", type=str, default="cityscapes_train", help="Cache name")
    return parser.parse_args()

class FlattenedFolder:
    """
    上下文管理器：如果文件夹是嵌套的，创建一个临时的扁平化文件夹（使用软链接）。
    如果已经是扁平的，则直接使用原路径。
    """
    def __init__(self, original_path):
        self.original_path = original_path
        self.temp_dir = None
        self.final_path = original_path

    def __enter__(self):
        # 1. 扫描所有图片
        image_files = []
        abs_original_path = os.path.abspath(self.original_path)
        for root, _, files in os.walk(abs_original_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(root, file))
        
        print(f"[Info] Found {len(image_files)} images in {abs_original_path}")
        
        # 2. 检查是否需要扁平化
        # 如果根目录下图片数量 等于 总图片数量，说明已经是扁平的，不需要操作
        root_images = [f for f in os.listdir(self.original_path) if f.lower().endswith(('.png', '.jpg'))]
        if len(root_images) == len(image_files):
            return self.original_path

        # 3. 创建临时目录并建立软链接
        self.temp_dir = tempfile.mkdtemp(prefix="fid_flatten_")
        print(f"[Info] Detected nested folders. Created temp flat dir at: {self.temp_dir}")
        self.temp_dir = os.path.abspath(self.temp_dir)
        
        for src_path in image_files:
            # 防止重名：比如 aachen/01.png 和 bochum/01.png
            # 新名字变成 aachen_01.png
            folder_name = os.path.basename(os.path.dirname(src_path))
            file_name = os.path.basename(src_path)
            new_name = f"{folder_name}_{file_name}"
            dst_path = os.path.join(self.temp_dir, new_name)
            
            os.symlink(src_path, dst_path)
            
        self.final_path = self.temp_dir
        return self.final_path

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 清理临时目录
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"[Info] Cleaned up temp dir.")

def main():
    args = parse_args()
    
    # 路径检查
    if not os.path.exists(args.gen_folder):
        raise FileNotFoundError(f"Folder not found: {args.gen_folder}")
    if os.path.exists(os.path.join(args.gen_folder, "RGB")):
        args.gen_folder = os.path.join(args.gen_folder, "RGB")
        
    if not os.path.exists(args.real_folder):
        raise FileNotFoundError(f"Folder not found: {args.real_folder}")
    if os.path.exists(os.path.join(args.real_folder, "leftImg8bit", "train")):
        args.real_folder = os.path.join(args.real_folder, "leftImg8bit", "train")

    # 使用上下文管理器处理 Real 文件夹
    with FlattenedFolder(args.real_folder) as real_flat_path:
        
        # 同样处理 Gen 文件夹 (万一你的生成结果也是分文件夹的)
        with FlattenedFolder(args.gen_folder) as gen_flat_path:
            
            print(f"--------------------------------------------------")
            print(f"Comparing:")
            print(f"  Gen Path:  {gen_flat_path}")
            print(f"  Real Path: {real_flat_path}")
            print(f"  Mode:      legacy_pytorch")
            print(f"--------------------------------------------------")

            # 1. 检查/创建 Real 数据集的缓存
            if fid.test_stats_exists(args.dataset_name, mode="legacy_pytorch"):
                print(f"[Cache] Found cached stats for '{args.dataset_name}'.")
            else:
                print(f"[Cache] Computing stats for '{args.dataset_name}' (First run only)...")
                fid.make_custom_stats(
                    name=args.dataset_name, 
                    fdir=real_flat_path, 
                    mode="legacy_pytorch"
                )

            # 2. 计算 FID
            print(f"Calculating FID...")
            fid_score = fid.compute_fid(
                fdir1=gen_flat_path, 
                dataset_name=args.dataset_name,
                dataset_split="custom",
                mode="legacy_pytorch"
            )
            
            # 3. 计算 KID
            print(f"Calculating KID...")
            kid_score = fid.compute_kid(
                fdir1=gen_flat_path, 
                dataset_name=args.dataset_name,
                dataset_split="custom",
                mode="legacy_pytorch"
            )

            print(f"\n================ RESULTS ================")
            print(f" Dataset: {args.dataset_name}")
            print(f" FID:     {fid_score:.4f}")
            print(f" KID:     {kid_score:.6f}")
            print(f"=========================================")

if __name__ == "__main__":
    main()