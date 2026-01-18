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

def get_gradient_mask(pipe, latents, prompt_embeds, pooled_prompt_embeds, text_ids, radius, sensitivity, device, dtype, guidance_scale=2.0, save_debug_mask_filename=None):
    print("🔍 正在运行相位一致性冲突检测 (Phase-Only Conflict Detection)...")
    
    # 1. 准备 PPD 噪声
    noise_base = torch.randn_like(latents)
    noise_ppd = generate_structured_noise_batch_vectorized(
        latents, cutoff_radius=radius, input_noise=noise_base
    ).contiguous()
    
    # 开启梯度追踪
    noise_probe = noise_ppd.detach().clone().to(dtype).requires_grad_(True)
    
    # 2. Probe Step (单步推理)
    pipe.dit.train() # 开启 Gradient Checkpointing 节省显存
    
    pipe.scheduler.set_timesteps(1)
    t_probe = pipe.scheduler.timesteps[0].unsqueeze(0).to(device)
    img_ids = pipe.dit.prepare_image_ids(latents)
    guidance = torch.full((latents.shape[0],), guidance_scale, device=device, dtype=dtype)
    
    pred_velocity = pipe.dit(
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
    
    # 3. 计算相位一致性损失 (FFT)
    fft_input = torch.fft.fft2(noise_probe.float())
    fft_pred = torch.fft.fft2(pred_velocity.float())
    
    # 归一化幅度，只保留相位方向
    phase_input = fft_input / (fft_input.abs() + 1e-6)
    phase_pred = fft_pred / (fft_pred.abs() + 1e-6)
    
    # 计算相位对齐度 (Cosine Similarity)
    phase_alignment = (phase_pred * phase_input.conj()).real.mean()
    target_loss = -phase_alignment # 目标是最大化对齐，所以 Loss 是负的对齐度
    
    # 4. 计算梯度 (Saliency)
    grads = torch.autograd.grad(target_loss, noise_probe)[0]
    conflict_map = grads.abs().mean(dim=1, keepdim=True) # [B, 1, H, W]
    
    # 平滑处理 (Blur)
    conflict_map = TF.gaussian_blur(conflict_map, kernel_size=5, sigma=1.5)
    
    # === 关键修改：自适应阈值 (Adaptive Thresholding) ===
    # 我们不再使用 Quantile 强制切分，而是使用统计学分布 (Mean + Std)
    
    map_float = conflict_map.float().view(-1)
    mean_val = map_float.mean()
    std_val = map_float.std()
    
    # 逻辑：
    # 平均值 (mean) 代表了“背景噪音”或“平均冲突水平”。
    # 真正的结构性冲突通常是异常值 (Outliers)。
    # 我们定义：超过 (均值 + 3倍标准差) 的区域才是必须重绘的剧烈冲突。
    
    low = mean_val
    # 3-Sigma 原则：覆盖 99.7% 的正常分布。只有极端的梯度会被视为冲突。
    # 这里的系数 3.0 可以根据需要微调，但在 PPD 中通常 2.5 - 3.0 是比较稳健的。
    high = mean_val + 1.0 * std_val 
    
    # 防止标准差为0导致的除零错误 (比如纯色图)
    range_val = torch.max(high - low, torch.tensor(1e-6, device=device))
    
    # 归一化
    mask_norm = (conflict_map.float() - low) / range_val
    mask_norm = torch.clamp(mask_norm, 0, 1)
    
    # === 修改结束 ===
    
    # 反转逻辑：冲突(Mask=1) -> 重绘(0)；一致(Mask=0) -> 保留(1)
    # 这里的 sensitivity 依然有效，用于调整 Sigmoid 的对比度坡度
    mask = 1.0 - torch.sigmoid((mask_norm - 0.5) * 10 * sensitivity)
    mask = mask.to(dtype=dtype)
    
    # 保存调试图
    if save_debug_mask_filename:
        save_debug_mask(mask, latents.shape[3]*8, latents.shape[2]*8, filename=save_debug_mask_filename)
    
    # 8. 融合
    final_noise = mask * noise_ppd + (1 - mask) * noise_base
    
    print(f"    📈 Phase Alignment Score: {phase_alignment.item():.4f}")
    print(f"    📊 Stats: Mean={mean_val:.5f}, Std={std_val:.5f} (Adaptive Threshold used)")
    
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