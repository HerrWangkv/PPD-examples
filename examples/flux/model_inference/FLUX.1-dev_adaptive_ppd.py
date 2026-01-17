import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFilter
from diffsynth import download_models
from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig
# Assuming structured_noise is in the python path or same directory
from structured_noise import generate_structured_noise_batch_vectorized

def parse_args():
    parser = argparse.ArgumentParser(description="Adaptive Sim2Real with Flux PPD")
    # --- Standard Arguments ---
    parser.add_argument("--lora_checkpoint_path", type=str, default="models/ppd/flux1-dev_lora_color_step=266000_biased.safetensors")
    parser.add_argument("--input_image", type=str, required=True, help="Input image (Sim/GreenScreen)")
    parser.add_argument("--output_name", type=str, default="output_adaptive.png")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative_prompt", type=str, default="ugly, low quality, CG, Render, unreal, game, cartoon, blur, low res")
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--cfg_scale", type=float, default=3.5, help="Guidance scale for Flux (Standard Dev is 3.5)")
    
    # --- Adaptive PPD Arguments ---
    parser.add_argument("--radius", type=int, default=35, help="Radius for structure preservation")
    parser.add_argument("--sensitivity", type=float, default=1.0, help="Higher value = stricter masking (more background replacement)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def get_adaptive_noise(pipe, latents, prompt_embeds, pooled_prompt_embeds, text_ids, radius, sensitivity, device, dtype, guidance_scale=3.5):
    """
    Generates hybrid noise by detecting conflicts between PPD structure and Prompt intent.
    """
    print("🔍 Running Adaptive Conflict Detection...")
    
    # 1. Generate Dual Noise Sources
    # Source A: PPD Noise (Forces original structure everywhere)
    noise_base = torch.randn_like(latents)
    noise_ppd = generate_structured_noise_batch_vectorized(
        latents, cutoff_radius=radius, input_noise=noise_base
    ).contiguous()
    
    # Source B: Random Noise (Forces Prompt generation everywhere)
    noise_rnd = noise_base # Use same base for coherence
    
    # 2. The Probe Step (Simulate t=1.0 prediction)
    with torch.no_grad():
        # Set timesteps (remove device arg as requested by previous error)
        pipe.scheduler.set_timesteps(1)
        # Manually move timestep to device
        t_probe = pipe.scheduler.timesteps[0].unsqueeze(0).to(device)
        
        # Prepare Image IDs
        img_ids = pipe.dit.prepare_image_ids(latents)
        
        # Create Guidance Tensor [B]
        guidance = torch.full((latents.shape[0],), guidance_scale, device=device, dtype=dtype)
        
        # Predict A (PPD)
        pred_ppd = pipe.dit(
            hidden_states=noise_ppd,
            timestep=t_probe,
            prompt_emb=prompt_embeds,
            pooled_prompt_emb=pooled_prompt_embeds,
            text_ids=text_ids,
            image_ids=img_ids,
            guidance=guidance
        )
        
        # Predict B (Random)
        pred_rnd = pipe.dit(
            hidden_states=noise_rnd,
            timestep=t_probe,
            prompt_emb=prompt_embeds,
            pooled_prompt_emb=pooled_prompt_embeds,
            text_ids=text_ids,
            image_ids=img_ids,
            guidance=guidance
        )

    # 3. Compute Conflict Map
    # Difference implies the structure disagrees with the prompt
    diff = torch.abs(pred_ppd - pred_rnd).mean(dim=1, keepdim=True) # [B, 1, H, W]
    
    # --- FIX START: Cast to float32 for quantile ---
    diff_float = diff.float() 
    diff_flat = diff_float.view(-1)
    
    low = diff_flat.quantile(0.1)
    high = diff_flat.quantile(0.9)
    
    # Normalize (using float32 precision)
    diff_norm = (diff_float - low) / (high - low + 1e-6)
    diff_norm = torch.clamp(diff_norm, 0, 1)
    # --- FIX END ---
    
    # 4. Create Mask (1 = Keep PPD, 0 = Use Random)
    # Sigmoid to create a soft mask based on sensitivity
    # Low difference (Subject) -> High Mask Value -> Keep Structure
    # High difference (Background) -> Low Mask Value -> Randomize
    mask = 1.0 - torch.sigmoid((diff_norm - 0.5) * 10 * sensitivity)
    
    # Cast mask back to original dtype (bfloat16) for mixing
    mask = mask.to(dtype=dtype)
    
    # Optional: Save debug mask
    save_debug_mask(mask)

    # 5. Fuse Noise
    final_noise = mask * noise_ppd + (1 - mask) * noise_rnd
    return final_noise

def save_debug_mask(mask_tensor):
    try:
        m = mask_tensor[0,0].float().cpu().numpy()
        m = (m * 255).astype(np.uint8)
        Image.fromarray(m).resize((512, 512), 0).save("debug_adaptive_mask.png")
        print("   -> Debug mask saved to 'debug_adaptive_mask.png'")
    except Exception as e:
        print(f"   -> Could not save debug mask: {e}")

if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    
    # 1. Load Model
    download_models(["FLUX.1-dev"])
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
    
    # Optional LoRA
    if args.lora_checkpoint_path:
        try:
            pipe.load_lora(pipe.dit, args.lora_checkpoint_path, alpha=1)
            print(f"Loaded LoRA: {args.lora_checkpoint_path}")
        except:
            print("LoRA not found or failed to load, proceeding without it.")

    # 2. Prepare Image & Dimensions
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
        # 3. Encode Image
        image = pipe.preprocess_image(image_in_pil).to(device=pipe.device, dtype=pipe.torch_dtype)
        input_latents = pipe.vae_encoder(image, tiled=False)

        # 4. Encode Prompt (Corrected for DiffSynth)
        prompt_embeds, pooled_prompt_embeds, text_ids = pipe.prompter.encode_prompt(
            prompt=args.prompt, 
            device=pipe.device,
            positive=True
        )

        # 5. Compute Adaptive Noise
        noise = get_adaptive_noise(
            pipe=pipe,
            latents=input_latents,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            text_ids=text_ids,
            radius=args.radius,
            sensitivity=args.sensitivity,
            device=pipe.device,
            dtype=pipe.torch_dtype,
            guidance_scale=args.cfg_scale
        )

        # 6. Generate
        print("🎨 Generating Image...")
        image = pipe(
            prompt=args.prompt, 
            negative_prompt=args.negative_prompt,
            height=new_h, 
            width=new_w,
            cfg_scale=args.cfg_scale, 
            num_inference_steps=50, 
            noise=noise # Inject the fused adaptive noise
        )

        # 7. Save
        if use_original_size:
            image = image.resize((w, h))
        image.save(args.output_name)
        print(f"Done. Saved to {args.output_name}")