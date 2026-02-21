import argparse
import os
import torch
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from prettytable import PrettyTable

# Cityscapes 的 19 个类别名称
CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
    'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
    'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]

# --- 新增: Cityscapes 调色板 (RGB) ---
CITYSCAPES_PALETTE = np.array([
    [128, 64, 128],  # 0: Road (紫)
    [244, 35, 232],  # 1: Sidewalk (粉)
    [70, 70, 70],    # 2: Building (灰)
    [102, 102, 156], # 3: Wall (蓝灰)
    [190, 153, 153], # 4: Fence (淡紫)
    [153, 153, 153], # 5: Pole (灰)
    [250, 170, 30],  # 6: Traffic Light (橙)
    [220, 220, 0],   # 7: Traffic Sign (黄)
    [107, 142, 35],  # 8: Vegetation (绿)
    [152, 251, 152], # 9: Terrain (浅绿)
    [70, 130, 180],  # 10: Sky (蓝)
    [220, 20, 60],   # 11: Person (红)
    [255, 0, 0],     # 12: Rider (红)
    [0, 0, 142],     # 13: Car (深蓝)
    [0, 0, 70],      # 14: Truck (蓝灰)
    [0, 60, 100],    # 15: Bus (深青)
    [0, 80, 100],    # 16: Train (青)
    [0, 0, 230],     # 17: Motorcycle (蓝)
    [119, 11, 32],   # 18: Bicycle (深红)
], dtype=np.uint8)

def colorize_mask(mask):
    """将索引 Mask 转换为 RGB 彩色 Mask"""
    # 创建一个空的 RGB 图像
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    
    # 遍历每个类别进行上色 (只处理 0-18)
    for cls_id in range(len(CITYSCAPES_PALETTE)):
        color_mask[mask == cls_id] = CITYSCAPES_PALETTE[cls_id]
        
    return color_mask

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate mIoU using Pre-trained SegFormer")
    parser.add_argument(
        "--gen_folder", 
        type=str, 
        required=True,
        help="Folder containing your generated/stylized RGB images"
    )
    parser.add_argument(
        "--gt_folder", 
        type=str, 
        default="data/synthia",
        help="Folder containing Ground Truth Label images (Must be Cityscapes TrainIDs 0-18)"
    )
    parser.add_argument(
        "--debug_count",
        type=int,
        default=10,
        help="Number of debug images to save"
    )
    return parser.parse_args()

class Evaluator:
    def __init__(self, num_classes=19):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes))

    def update(self, pred_mask, gt_mask):
        """更新混淆矩阵"""
        pred = pred_mask.flatten()
        gt = gt_mask.flatten()
        
        # 过滤掉 GT 中标记为 255 (Ignore) 的像素
        mask = (gt >= 0) & (gt < self.num_classes)
        
        label = self.num_classes * gt[mask].astype('int') + pred[mask]
        count = np.bincount(label, minlength=self.num_classes**2)
        self.confusion_matrix += count.reshape(self.num_classes, self.num_classes)

    def compute_iou(self):
        intersection = np.diag(self.confusion_matrix)
        # Union = TP + FP + FN
        union = (
            self.confusion_matrix.sum(axis=1) + 
            self.confusion_matrix.sum(axis=0) - 
            intersection
        )
        iou = intersection / (union + 1e-10)
        return iou

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. 加载 Oracle 模型 (SegFormer)
    print(f"Loading Oracle Model on {device}...")
    model_name = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
    processor = SegformerImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_name, 
        use_safetensors=True
    ).to(device)
    model.eval()

    evaluator = Evaluator(num_classes=19)

    # 获取文件列表
    if not os.path.exists(args.gen_folder):
        raise FileNotFoundError(f"Generated images folder not found: {args.gen_folder}")
    
    # 检查是否包含 RGB 子文件夹，兼容两种路径写法
    if os.path.exists(os.path.join(args.gen_folder, "RGB")):
        args.gen_folder = os.path.join(args.gen_folder, "RGB")

    if not os.path.exists(args.gt_folder):
        raise FileNotFoundError(f"GT folder not found: {args.gt_folder}")
    
    # 检查是否包含 GT/LABELS 子文件夹
    if os.path.exists(os.path.join(args.gt_folder, "GT", "LABELS")):
        args.gt_folder = os.path.join(args.gt_folder, "GT", "LABELS")
    elif os.path.exists(os.path.join(args.gt_folder, "GT/LABELS")): # 有些系统路径写法不同
         args.gt_folder = os.path.join(args.gt_folder, "GT/LABELS")

    gen_files = sorted([f for f in os.listdir(args.gen_folder) if (f.endswith('.png') or f.endswith('.jpg'))])
    
    print(f"Found {len(gen_files)} images to evaluate.")
    print(f"Reading generated images from: {args.gen_folder}")
    print(f"Reading Ground Truth from:     {args.gt_folder}")

    # 计数器，用于限制 debug 图片数量
    saved_debug_count = 0

    for gen_name in tqdm(gen_files):
        gen_path = os.path.join(args.gen_folder, gen_name)
        
        gt_path = os.path.join(args.gt_folder, gen_name.replace(".png", "_labelTrainIds.png").replace(".jpg", "_labelTrainIds.png"))
        
        # 2. 读取生成图像 (RGB)
        image = Image.open(gen_path).convert("RGB")
        
        # 3. 模型预测
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            # 上采样 logits 到原图尺寸
            logits = torch.nn.functional.interpolate(
                outputs.logits, 
                size=image.size[::-1], # (height, width)
                mode="bilinear", 
                align_corners=False
            )
            pred_mask = logits.argmax(dim=1)[0].cpu().numpy()

        # 4. 读取 GT (Label ID)
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
        gt_mask = gt_mask.astype(np.int32)
        
        # 确保 GT 尺寸匹配 (缩放到 Image 尺寸)
        if gt_mask.shape[:2] != image.size[::-1]:
            gt_mask = cv2.resize(gt_mask, image.size, interpolation=cv2.INTER_NEAREST)

        # 处理多通道 GT 的情况
        if len(gt_mask.shape) == 3:
            gt_mask = gt_mask[:, :, 0]

        # 5. 更新统计
        evaluator.update(pred_mask, gt_mask)

        # --- 新增: Debug Visualization ---
        if saved_debug_count < args.debug_count:
            # 1. 准备原图 (numpy RGB)
            vis_img = np.array(image)
            
            # 2. 准备 GT 彩色图
            vis_gt = colorize_mask(gt_mask)
            
            # 3. 准备 预测 彩色图
            vis_pred = colorize_mask(pred_mask)
            
            # 4. 横向拼接: [原图 | GT | 预测]
            # 确保高度一致 (这里因为都resize到了image size，所以没问题)
            combined = np.concatenate([vis_img, vis_gt, vis_pred], axis=1)
            
            # 5. 添加文字标签 (可选)
            # 为了简单，直接保存
            save_name = f"debug_vis_{saved_debug_count}_{gen_name}"
            Image.fromarray(combined).save(save_name)
            saved_debug_count += 1

    # 6. 输出结果
    iou = evaluator.compute_iou()
    miou = np.nanmean(iou)

    table = PrettyTable(["Class", "IoU (%)"])
    for i, class_name in enumerate(CLASSES):
        table.add_row([class_name, f"{iou[i]*100:.2f}"])
    
    print(table)
    print(f"\nMean IoU (mIoU): {miou*100:.2f}%")
    if saved_debug_count > 0:
        print(f"Saved {saved_debug_count} debug images (debug_vis_*.png) for inspection.")

if __name__ == "__main__":
    main()