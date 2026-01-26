import argparse
import os
import torch
import clip
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate CLIP-IQA Realism Score")
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
    parser.add_argument(
        "--backbone",
        type=str,
        default="ViT-B/32",
        help="CLIP backbone (ViT-B/32 is standard and fast)"
    )
    return parser.parse_args()

def get_image_paths(folder):
    """递归查找所有图片路径"""
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    image_paths = []
    
    # 使用 os.walk 支持嵌套文件夹 (如 synthia/RGB/...)
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

    # 3. 定义对抗提示词 (Prompt Pairs)
    # 针对 Sim2Real 驾驶场景进行了优化
    positive_prompts = [
        "a real photo of a city street",
        "a realistic photo of a driving scene",
        "a real world image",
        "clear photography"
    ]
    
    negative_prompts = [
        "a computer generated image of a city",
        "a synthetic image",
        "a screenshot from a video game like GTA5",
        "digital rendering",
        "low quality synthetic data"
    ]

    all_prompts = positive_prompts + negative_prompts
    
    # 预计算文本特征
    print("Encoding text prompts...")
    text_inputs = clip.tokenize(all_prompts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    # 4. 批量计算图片分数
    scores = []
    
    print("Calculating realism scores...")
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
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # 计算相似度 (Batch_Size, Num_Prompts)
            # 100.0 是 CLIP 的温度系数 scaling
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
            # 计算 "真实感概率"
            # 逻辑：把所有 Positive Prompts 的概率加起来
            # similarity[:, :len(positive_prompts)] 取的是前 N 个正向提示词的列
            realism_probs = similarity[:, :len(positive_prompts)].sum(dim=-1)
            
            scores.extend(realism_probs.cpu().tolist())

    # 5. 统计结果
    avg_score = sum(scores) / len(scores)
    
    print(f"\n================ RESULTS ================")
    print(f" Images Processed: {len(scores)}")
    print(f" CLIP Realism Score: {avg_score:.4f}")
    print(f" (0.00 = Synthetic/Fake, 1.00 = Real/Photo)")
    print(f"=========================================")

if __name__ == "__main__":
    main()