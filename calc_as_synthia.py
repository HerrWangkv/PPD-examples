import argparse
import os
import torch
import clip
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate NeuralRemaster Appearance Score (AS)")
    parser.add_argument(
        "--gen_folder", 
        type=str, 
        required=True, 
        help="Path to your generated/adapted images"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=32, 
        help="Batch size for inference"
    )
    # NeuralRemaster 论文未指定 backbone，但 ViT-B/32 是 CLIP 的默认标准
    parser.add_argument(
        "--backbone",
        type=str,
        default="ViT-B/32",
        help="CLIP backbone"
    )
    return parser.parse_args()

def get_image_paths(folder):
    """递归查找所有图片路径"""
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    image_paths = []
    
    # 使用 os.walk 支持嵌套文件夹
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_paths.append(os.path.join(root, file))
    return image_paths

def main():
    args = parse_args()
    if os.path.exists(os.path.join(args.gen_folder, "RGB")):
        args.gen_folder = os.path.join(args.gen_folder, "RGB")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 查找图片
    print(f"Scanning folder: {args.gen_folder}")
    image_paths = get_image_paths(args.gen_folder)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.gen_folder}")
    print(f"Found {len(image_paths)} images.")

    # 2. 加载 CLIP 模型
    print(f"Loading CLIP ({args.backbone})...")
    model, preprocess = clip.load(args.backbone, device=device)
    model.eval()

    # 3. 定义 NeuralRemaster 专用提示词
    # 注意：这里使用论文原文指定的"单条"提示词，而不是列表
    # Positive: t_p
    positive_prompts = [
        "Photorealistic",
        "Real-world",
        "Natural illumination",
        "Real people with natural poses"
    ]

    negative_prompts = [
        "Unrealistic",
        "Simulated",
        "Flat lighting",
        "Stiff mannequin-like characters",
        "Artifacts"
    ]

    print(f"--------------------------------------------------")
    print(f"Metric: NeuralRemaster Appearance Score (AS)")
    print("Pos Prompts:")
    for p in positive_prompts:
        print(f"  - {p}")
    print("Neg Prompts:")
    for n in negative_prompts:
        print(f"  - {n}")
    print(f"Formula:    Sim(Img, Pos) / Sim(Img, Neg)")
    print(f"--------------------------------------------------")

    # 预计算文本特征
    # shape: [2, 512] -> index 0 is Pos, index 1 is Neg
    # 预计算文本特征（prompt ensemble，仍保持 NeuralRemaster 的 AS 形式：Sim(img,tp)/Sim(img,tn)）
    pos_inputs = clip.tokenize(positive_prompts).to(device)
    neg_inputs = clip.tokenize(negative_prompts).to(device)

    with torch.no_grad():
        pos_feats = model.encode_text(pos_inputs)
        tp = pos_feats.mean(dim=0)

        neg_feats = model.encode_text(neg_inputs)
        tn = neg_feats.mean(dim=0)


    # 4. 批量计算
    scores = []
    
    print("Calculating Appearance Scores...")
    for i in tqdm(range(0, len(image_paths), args.batch_size)):
        batch_paths = image_paths[i : i + args.batch_size]
        batch_images = []
        
        valid_batch = True
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                batch_images.append(preprocess(img).unsqueeze(0))
            except Exception as e:
                print(f"Warning: Could not read {p}. Skipping.")
                valid_batch = False
        
        if not valid_batch or len(batch_images) == 0:
            continue

        image_input = torch.cat(batch_images).to(device)
        
        with torch.no_grad():
            # 提取图像特征
            image_features = model.encode_image(image_input)
            
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            tp = tp / tp.norm(dim=-1, keepdim=True)
            tn = tn / tn.norm(dim=-1, keepdim=True)
            # sim_pos: [Batch_Size]
            sim_pos = (image_features @ tp)
            # sim_neg: [Batch_Size]
            sim_neg = (image_features @ tn)
            
            # --- 核心修改：AS 公式计算 ---
            # AS = x^T * tp / x^T * tn
            
            # 安全性处理：防止分母为 0 或负数导致分数翻转
            # CLIP 的余弦相似度极少为负，但在极端合成图上可能出现
            sim_neg = torch.clamp(sim_neg, min=1e-6)
            
            batch_scores = sim_pos / sim_neg
            
            scores.extend(batch_scores.cpu().tolist())

    # 5. 统计结果
    avg_score = sum(scores) / len(scores)
    
    print(f"\n================ RESULTS ================")
    print(f" Images Processed: {len(scores)}")
    print(f" Average Appearance Score (AS): {avg_score:.4f}")
    print(f" (Baseline typically < 0.5; High Quality > 1.0)")
    print(f"=========================================")

if __name__ == "__main__":
    main()