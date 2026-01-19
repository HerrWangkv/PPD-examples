import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from sklearn.mixture import GaussianMixture

from diffsynth import download_models
from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig

# 假设 structured_noise.py 在同一目录下
try:
    from structured_noise import generate_structured_noise_batch_vectorized
except ImportError:
    print("❌ 错误: 未找到 structured_noise.py，请确保 PPD 相关的噪声生成脚本在同一目录下。")
    exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Adaptive Sim2Real with Flux PPD (Semantic Divergence Mode)")
    # --- 基础参数 ---
    parser.add_argument("--lora_checkpoint_path", type=str, default="models/ppd/flux1-dev_lora_color_step=266000_biased.safetensors")
    parser.add_argument("--input_image", type=str, required=True, help="输入图像路径")
    parser.add_argument("--output_name", type=str, default="output_semantic.png")
    parser.add_argument("--old_prompt", type=str, required=True, help="描述原图内容的 Prompt (例如: a rock wall)")
    parser.add_argument("--new_prompt", type=str, required=True, help="描述新内容的 Prompt (例如: two explorers standing)")
    parser.add_argument("--negative_prompt", type=str, default="ugly, low quality, blur, low res")
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    
    # --- PPD 核心参数 ---
    parser.add_argument("--radius", type=int, default=35, help="PPD 结构保留半径 (建议 30-40)")
    parser.add_argument("--sensitivity", type=float, default=1.0, help="灵敏度: 值越大，越容易丢弃背景保留主体 (建议 0.8 - 1.5)")
    parser.add_argument("--semantic_hole_strength", type=float, default=3.0, 
                        help="语义挖孔强度: 值越大，Prompt要求修改的区域越容易被强制重绘 (建议 2.0 - 5.0)")
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

def get_structure_mask(
    pipe,
    latents,
    old_prompt_embeds,
    old_pooled_embeds,
    old_text_ids,
    new_prompt_embeds,
    new_pooled_embeds,
    new_text_ids,
    radius,
    sensitivity,
    device,
    dtype,
    semantic_hole_strength=3.0,
    guidance_scale=3.5,
    save_debug_mask_filename="debug_mask.png",
):
    """
    Structure-aware noise mixing using Spectral Saliency & Semantic Divergence.
    """

    print("🔍 Running structure detection (Spectral Gradient + Semantic Hole)...")

    # ------------------------------------------------------------------
    # 1. Prepare noises
    # ------------------------------------------------------------------
    noise_base = torch.randn_like(latents)

    noise_ppd = generate_structured_noise_batch_vectorized(
        latents,
        cutoff_radius=radius,
        input_noise=noise_base,
    ).contiguous()

    # 开启梯度追踪，用于计算 Saliency
    noise_probe = noise_ppd.detach().clone().to(dtype).requires_grad_(True)

    # 拼接 Batch: [Old, New]
    # Old 用于检测"原图有什么结构"
    # New 用于检测"我想改哪里"
    combined_noise = torch.cat([noise_probe, noise_probe], dim=0)
    combined_prompt_embeds = torch.cat([old_prompt_embeds, new_prompt_embeds], dim=0)
    combined_pooled_embeds = torch.cat([old_pooled_embeds, new_pooled_embeds], dim=0)
    combined_text_ids = torch.cat([old_text_ids, new_text_ids], dim=0)
    
    # ------------------------------------------------------------------
    # 2. Single-step forward probing
    # ------------------------------------------------------------------
    pipe.dit.train() # 允许梯度回传
    pipe.scheduler.set_timesteps(1)
    t_probe = pipe.scheduler.timesteps[0].unsqueeze(0).to(device)
    combined_t = torch.cat([t_probe, t_probe], dim=0)

    img_ids = pipe.dit.prepare_image_ids(latents)
    combined_img_ids = torch.cat([img_ids, img_ids], dim=0)
    guidance = torch.full(
        (2*latents.shape[0],),
        guidance_scale,
        device=device,
        dtype=dtype,
    )
    
    with torch.enable_grad():
        combined_pred = pipe.dit(
            hidden_states=combined_noise,
            timestep=combined_t,
            prompt_emb=combined_prompt_embeds,
            pooled_prompt_emb=combined_pooled_embeds,
            text_ids=combined_text_ids,
            image_ids=combined_img_ids,
            guidance=guidance,
            use_gradient_checkpointing=True
        )
        old_pred, new_pred = combined_pred.chunk(2)

        # ------------------------------------------------------------------
        # 3. 结构显著性计算 (Base Structure Detection)
        # ------------------------------------------------------------------
        # [关键修正] 我们只使用 old_pred (描述原图) 来寻找物理结构。
        # 这样可以客观地找出"哪里有石头"，而不受"新加的人"的干扰。
        target_probe = new_pred
        
        fft_input = torch.fft.fft2(noise_probe.float())
        fft_output = torch.fft.fft2(target_probe.float())
        
        # 相位对齐
        phase_input = fft_input / (fft_input.abs() + 1e-6)
        phase_output = fft_output / (fft_output.abs() + 1e-6)
        
        # Loss: 最大化相位对齐度
        phase_alignment = (phase_output * phase_input.conj()).real.sum()
        target_loss = -phase_alignment 
        
        grads = torch.autograd.grad(target_loss, noise_probe)[0]
    
    pipe.dit.eval() 

    # ------------------------------------------------------------------
    # 4. GMM 聚类生成基础 Mask (Base Mask)
    # ------------------------------------------------------------------
    # 梯度模长 = 结构显著性
    conflict_map = grads.abs().amax(dim=1, keepdim=True).float()

    # 预处理：模糊 + 对数变换 (处理长尾分布)
    conflict_map = TF.gaussian_blur(conflict_map, kernel_size=9, sigma=2.0)
    log_conflict_map = torch.log1p(conflict_map * 100)
    
    flat_view = log_conflict_map.view(-1, 1)
    flat_view = (flat_view - flat_view.min()) / (flat_view.max() - flat_view.min() + 1e-8)
    energy = flat_view.cpu().numpy()

    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=0)
    gmm.fit(energy)

    # [关键修正] 结构是梯度最大的区域 (High Saliency)
    # argmax: 均值最大的类 = 结构
    # argmin: 均值最小的类 = 背景
    print("   GMM means:", gmm.means_.squeeze())
    struct_idx = np.argmin(gmm.means_.squeeze()) 
    p_structure = gmm.predict_proba(energy)[:, struct_idx]

    base_mask = torch.from_numpy(p_structure).to(device=device, dtype=dtype)
    base_mask = base_mask.view(latents.shape[0], 1, latents.shape[2], latents.shape[3])

    # ------------------------------------------------------------------
    # 5. 语义挖孔 (Semantic Hole Punching)
    # ------------------------------------------------------------------
    # 计算 old_pred 和 new_pred 的差异。
    # 如果差异巨大，说明 Prompt 强烈要求改变这个区域 (例如：把石头改成得人)
    semantic_diff = (new_pred - old_pred).float().norm(dim=1, keepdim=True)
    
    # 平滑差异图，让挖孔区域连成一片
    semantic_diff = TF.gaussian_blur(semantic_diff, kernel_size=9, sigma=3.0)
    
    # 归一化差异
    diff_flat = semantic_diff.view(-1)
    diff_norm = (semantic_diff - diff_flat.min()) / (diff_flat.max() - diff_flat.min() + 1e-6)
    
    # 生成挖孔 Mask (1 = 保留, 0 = 挖掉)
    # 差异越大(diff_norm -> 1)，hole_map 越接近 0
    # semantic_hole_strength 控制挖孔的"狠"度
    hole_map = 1.0 - torch.sigmoid((diff_norm - 0.5) * semantic_hole_strength * 10.0)
    hole_map = hole_map.to(dtype)
    
    print(f"   ⛏️ Applied Semantic Hole Punching (Strength={semantic_hole_strength})")

    # ------------------------------------------------------------------
    # 6. 最终 Mask 融合
    # ------------------------------------------------------------------
    # 逻辑：保留原图结构 (base_mask)，除非 Prompt 说这里要变 (hole_map)
    final_mask = base_mask * hole_map

    # Sensitivity 调整 (对比度增强)
    if sensitivity is not None:
        final_mask = torch.sigmoid((final_mask - 0.5) * 10.0 * sensitivity)
        
    # 强制清理底噪
    final_mask[final_mask < 0.2] = 0.0

    # ------------------------------------------------------------------
    # 7. Debug visualization
    # ------------------------------------------------------------------
    save_debug_mask(base_mask, latents.shape[3]*8, latents.shape[2]*8, filename=save_debug_mask_filename.replace(".png", "_base.png"))
    save_debug_mask(hole_map.detach(), latents.shape[3]*8, latents.shape[2]*8, filename=save_debug_mask_filename.replace(".png", "_hole.png"))
    save_debug_mask(final_mask.detach(), latents.shape[3]*8, latents.shape[2]*8, filename=save_debug_mask_filename)

    # ------------------------------------------------------------------
    # 8. Noise mixing
    # ------------------------------------------------------------------
    noise_ppd = noise_ppd.detach() # 记得 detach
    final_noise = final_mask * noise_ppd + (1.0 - final_mask) * noise_base

    return final_noise

if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    
    # 1. 加载模型
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
    
    # 加载 LoRA
    if args.lora_checkpoint_path:
        try:
            pipe.load_lora(pipe.dit, args.lora_checkpoint_path, alpha=1.0)
            print(f"✅ 已加载 LoRA: {args.lora_checkpoint_path}")
        except Exception as e:
            print(f"⚠️ LoRA 加载失败或未找到，将跳过: {e}")

    # 2. 图像预处理
    image_in_pil = Image.open(args.input_image).convert("RGB")
    w, h = image_in_pil.size
    
    if args.height and args.width:
        new_w, new_h = args.width, args.height
        use_original_size = False
    else:
        new_w, new_h = w//16*16, h//16*16
        use_original_size = True
        
    image_in_pil = image_in_pil.resize((new_w, new_h), resample=Image.LANCZOS)

    with torch.no_grad():
        # 3. 编码图像 -> Latents
        image_tensor = pipe.preprocess_image(image_in_pil).to(device=pipe.device, dtype=pipe.torch_dtype)
        input_latents = pipe.vae_encoder(image_tensor, tiled=False)

        # 4. 编码 Prompt
        print(f"📝 Old Prompt: {args.old_prompt}")
        old_prompt_embeds, old_pooled_embeds, old_text_ids = pipe.prompter.encode_prompt(
            prompt=args.old_prompt, 
            device=pipe.device,
            positive=True
        )
        print(f"📝 New Prompt: {args.new_prompt}")
        new_prompt_embeds, new_pooled_embeds, new_text_ids = pipe.prompter.encode_prompt(
            prompt=args.new_prompt, 
            device=pipe.device,
            positive=True
        )

    # 5. 计算自适应噪声 (Semantic Divergence 模式)
    noise = get_structure_mask(
        pipe=pipe,
        latents=input_latents,
        old_prompt_embeds=old_prompt_embeds,
        old_pooled_embeds=old_pooled_embeds,
        old_text_ids=old_text_ids,
        new_prompt_embeds=new_prompt_embeds,
        new_pooled_embeds=new_pooled_embeds,
        new_text_ids=new_text_ids,
        radius=args.radius,
        sensitivity=args.sensitivity,
        semantic_hole_strength=args.semantic_hole_strength,
        device=pipe.device,
        dtype=pipe.torch_dtype,
        save_debug_mask_filename=args.output_name.replace(".png", "_debug_mask.png")
    )

    # 6. 生成最终图像
    print("🎨 正在生成最终图像...")
    with torch.no_grad():
        image = pipe(
            prompt=args.new_prompt, 
            negative_prompt=args.negative_prompt,
            height=new_h, 
            width=new_w,
            cfg_scale=2, 
            num_inference_steps=50, 
            noise=noise # 注入融合后的噪声
        )

    # 7. 保存结果
    if use_original_size:
        image = image.resize((w, h))
    image.save(args.output_name)
    print(f"✅ 完成! 图像已保存至 {args.output_name}")