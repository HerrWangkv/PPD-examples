import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

from diffsynth import download_models
from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig

# 假设 structured_noise.py 在同一目录下
try:
    from structured_noise import generate_structured_noise_batch_vectorized
except ImportError:
    print("❌ 错误: 未找到 structured_noise.py，请确保 PPD 相关的噪声生成脚本在同一目录下。")
    exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Adaptive Sim2Real with Flux PPD (Gradient Method)")
    # --- 基础参数 ---
    parser.add_argument("--lora_checkpoint_path", type=str, default="models/ppd/flux1-dev_lora_color_step=266000_biased.safetensors")
    parser.add_argument("--input_image", type=str, required=True, help="输入图像路径")
    parser.add_argument("--output_name", type=str, default="output_gradient.png")
    parser.add_argument("--prompt", type=str, required=True, help="你的 Prompt")
    parser.add_argument("--negative_prompt", type=str, default="ugly, low quality, blur, low res")
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--cfg_scale", type=float, default=3.5)
    
    # --- PPD 核心参数 ---
    parser.add_argument("--radius", type=int, default=35, help="PPD 结构保留半径 (建议 30-40)")
    parser.add_argument("--sensitivity", type=float, default=1.0, help="灵敏度: 值越大，越容易丢弃背景保留主体 (建议 0.8 - 1.5)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def save_debug_mask(mask_tensor, width, height, filename="debug_mask.png"):
    """保存 Mask 用于调试 (白色=保留结构, 黑色=重绘)"""
    try:
        m = mask_tensor[0,0].float().cpu().numpy()
        m = (m * 255).astype(np.uint8)
        Image.fromarray(m).resize((width, height), 0).save(filename)
        print(f"   -> 调试 Mask 已保存至: {filename}")
    except Exception as e:
        print(f"   -> 保存调试 Mask 失败: {e}")

def get_gradient_mask(pipe, latents, prompt_embeds, pooled_prompt_embeds, text_ids, radius, sensitivity, device, dtype, guidance_scale=2.0, save_debug_mask_filename="debug_mask.png"):
    """
    使用输入显著性 (Input Saliency/Gradient) 来生成 Adaptive Mask。
    原理：
    1. 计算 Output 相对于 Input 的梯度。
    2. 高梯度 = 结构重要区域 (保留 PPD)。
    3. 低梯度 = 平坦/背景区域 (使用随机噪声)。
    """
    print("🔍 正在运行基于梯度的结构检测 (Gradient Saliency)...")
    
    # 1. 生成基础噪声源
    # noise_base: 纯随机噪声 (用于背景重绘)
    # noise_ppd: PPD 结构化噪声 (用于主体保留)
    noise_base = torch.randn_like(latents)
    noise_ppd = generate_structured_noise_batch_vectorized(
        latents, cutoff_radius=radius, input_noise=noise_base
    ).contiguous()
    
    # 2. 准备探测变量
    # 我们克隆 noise_ppd 并开启梯度追踪
    noise_probe = noise_ppd.detach().clone().to(dtype).requires_grad_(True)
    
    # 3. 运行单步推理 (Probe Step)
    # 设置时间步为 t=1.0 (初始去噪步)
    pipe.scheduler.set_timesteps(1)
    t_probe = pipe.scheduler.timesteps[0].unsqueeze(0).to(device)
    img_ids = pipe.dit.prepare_image_ids(latents)
    guidance = torch.full((latents.shape[0],), guidance_scale, device=device, dtype=dtype)
    
    pipe.dit.train()
    pred = pipe.dit(
        hidden_states=noise_probe,
        timestep=t_probe,
        prompt_emb=prompt_embeds,
        pooled_prompt_emb=pooled_prompt_embeds,
        text_ids=text_ids,
        image_ids=img_ids,
        guidance=guidance,
        use_gradient_checkpointing=True
    )
    pipe.dit.eval()
    # 4. 计算梯度 (核心逻辑)
    # 我们计算预测输出的 "能量" (Norm)，并反向传播看输入的哪些像素对这个能量贡献最大
    target_loss = pred.norm() 
    grads = torch.autograd.grad(target_loss, noise_probe)[0]
    
    # 5. 处理梯度生成 Mask
    # 取梯度的幅度 (Magnitude)
    saliency_map = -grads.abs().mean(dim=1, keepdim=True) # [B, 1, H, W]
    
    # 6. 后处理 Mask
    # 高斯模糊：连接断裂的边缘，让 Mask 成块
    saliency_map = TF.gaussian_blur(saliency_map, kernel_size=5, sigma=1.5)
    
    # 鲁棒归一化 (Robust Normalization)
    saliency_float = saliency_map.float().view(-1)
    # 假设梯度最小的 30% 是完全的背景 (可根据需要调整)
    low = saliency_float.quantile(0.3)
    # 假设梯度最大的 5% 是绝对的结构
    high = saliency_float.quantile(0.95)
    
    mask_norm = (saliency_map.float() - low) / (high - low + 1e-6)
    mask_norm = torch.clamp(mask_norm, 0, 1)
    
    # 应用 Sensitivity (灵敏度) 和 Sigmoid 增加对比度
    # Sensitivity > 1 会让 Mask 更黑 (更多背景重绘)
    # Sensitivity < 1 会让 Mask 更白 (更多结构保留)
    # 注意：这里我们反转一下逻辑，让 Mask=1 代表保留 PPD
    mask = torch.sigmoid((mask_norm - 0.5) * 10 * sensitivity)
    mask = mask.to(dtype=dtype)
    
    # 保存调试图
    save_debug_mask(mask, latents.shape[3]*8, latents.shape[2]*8, filename=save_debug_mask_filename)
    
    # 7. 噪声融合
    # Mask (白色/高梯度) -> 使用 noise_ppd (原始结构)
    # 1-Mask (黑色/低梯度) -> 使用 noise_base (随机生成)
    final_noise = mask * noise_ppd + (1 - mask) * noise_base
    
    return final_noise

if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    
    # 1. 加载模型
    # 自动下载模型 (如果需要)
    download_models(["FLUX.1-dev"])
    
    print("🚀 正在加载 Flux Pipeline...")
    pipe = FluxImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="flux1-dev.safetensors"),
            ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors"),
            ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/"),
            ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors"),
        ],
    )
    
    # 加载 LoRA (如果有)
    if args.lora_checkpoint_path:
        try:
            pipe.load_lora(pipe.dit, args.lora_checkpoint_path, alpha=1.0)
            print(f"✅ 已加载 LoRA: {args.lora_checkpoint_path}")
        except Exception as e:
            print(f"⚠️ LoRA 加载失败或未找到，将跳过: {e}")

    # 2. 图像预处理
    image_in_pil = Image.open(args.input_image).convert("RGB")
    w, h = image_in_pil.size
    
    # 调整大小
    if args.height and args.width:
        new_w, new_h = args.width, args.height
        use_original_size = False
    else:
        new_w, new_h = w//16*16, h//16*16 # 确保能被16整除
        use_original_size = True
        
    image_in_pil = image_in_pil.resize((new_w, new_h), resample=Image.LANCZOS)

    with torch.no_grad():
        # 3. 编码图像 -> Latents
        image_tensor = pipe.preprocess_image(image_in_pil).to(device=pipe.device, dtype=pipe.torch_dtype)
        input_latents = pipe.vae_encoder(image_tensor, tiled=False)

        # 4. 编码 Prompt
        print(f"📝 Prompt: {args.prompt}")
        prompt_embeds, pooled_prompt_embeds, text_ids = pipe.prompter.encode_prompt(
            prompt=args.prompt, 
            device=pipe.device,
            positive=True
        )

    # 5. 计算自适应噪声 (Gradient 模式)
    # 注意：这里需要临时开启梯度，所以不能在大 torch.no_grad() 块内
    noise = get_gradient_mask(
        pipe=pipe,
        latents=input_latents,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        text_ids=text_ids,
        radius=args.radius,
        sensitivity=args.sensitivity,
        device=pipe.device,
        dtype=pipe.torch_dtype,
        guidance_scale=args.cfg_scale,
        save_debug_mask_filename=args.output_name.replace(".png", "_debug_mask.png")
    )

    # 6. 生成最终图像
    print("🎨 正在生成最终图像...")
    with torch.no_grad():
        image = pipe(
            prompt=args.prompt, 
            negative_prompt=args.negative_prompt,
            height=new_h, 
            width=new_w,
            cfg_scale=args.cfg_scale, 
            num_inference_steps=50, 
            noise=noise # 注入融合后的噪声
        )

    # 7. 保存结果
    if use_original_size:
        image = image.resize((w, h))
    image.save(args.output_name)
    print(f"✅ 完成! 图像已保存至 {args.output_name}")