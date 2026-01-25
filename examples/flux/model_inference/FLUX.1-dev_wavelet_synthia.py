import argparse
import torch
import os
import cv2
import numpy as np
import torch.nn.functional as F
from PIL import Image
from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig
from diffsynth import download_models
from wavelet_noise import generate_wavelet_structured_noise_batch_vectorized

# SYNTHIA Dataset Mapping
# Sky is 10. We treat it specially to avoid rendering artifacts.
SKY_CLASS = 10 
# We keep target classes logic for potential future overrides
TARGET_CLASSES = [4, 5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18]

def parse_args():
    parser = argparse.ArgumentParser(description="Generate videos with trained model")
    parser.add_argument(
        "--lora_checkpoint_path",
        type=str,
        required=True,
        help="Path to lora checkpoint file"
    )
    parser.add_argument(
        "--synthia_folder",
        type=str,
        default="data/synthia",
        help="Synthia dataset folder"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="data/synthia_wavelet",
        help="Output folder"
    )
    parser.add_argument(
        "--cutoff_radius", 
        type=int, 
        default=20, 
        help="Pixel radius for near degradation (heavy noise)")
    parser.add_argument(
        "--maximal_radius", 
        type=int, 
        default=40, 
        help="Pixel radius for far protection (sharp)")
    parser.add_argument(
        "--gamma", 
        type=float, 
        default=1, 
        help="Depth curve control")
    parser.add_argument(
        "--prompt",
        type=str,
        default="A photorealistic driving scene in a European city, view from a car dashboard. Natural lighting, detailed asphalt road, urban buildings, trees, cars on the street. High resolution, cinematic, realistic textures, automotive photography.",
        help="Prompt"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="cartoon, video game, cgi, 3d render, unity engine, synthetic, low resolution, blurry, distorted, overexposed, oversaturated, painting, drawing, illustration, glitch, artifacts, deformed vehicles.",
        help="Negative prompt"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=704,
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
    )
    return parser.parse_args()

def get_related_paths(image_path):
    """
    Infers Depth and Mask paths from SYNTHIA structure.
    RGB: data/synthia/RGB/0009330.png
    Depth: data/synthia/Depth/Depth/0009330.png
    Labels: data/synthia/GT/LABELS/0009330_labelTrainIds.png
    """
    base_name = os.path.basename(image_path)
    root = image_path.split("/RGB/")[0]
    
    depth_path = os.path.join(root, "Depth", "Depth", base_name)
    mask_path = os.path.join(root, "GT", "LABELS", base_name.replace(".png", "_labelTrainIds.png"))
    
    return depth_path, mask_path

def load_and_preprocess_synthia_data(depth_path, mask_path, size, device, max_depth_limit=500.0):
    """Loads depth and handles sky/outlier depth values."""
    # Load 16-bit depth (cm)
    depth_cv2 = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_cv2 is None: raise FileNotFoundError(f"Depth not found: {depth_path}")
    
    # Handle multi-channel encoding
    if len(depth_cv2.shape) == 3:
        depth_cv2 = np.max(depth_cv2, axis=2)
    
    # Convert to meters
    depth_m = depth_cv2.astype(np.float32) / 100.0
    
    # --- Assert/Clamp Outliers ---
    # Any depth significantly beyond realistic scene limits is treated as "Far" 
    # and clamped to prevent normalization skewing.
    depth_m = np.clip(depth_m, 0.0, max_depth_limit)
    
    # Load mask and move to tensor
    depth_pt = torch.from_numpy(depth_m).to(device).view(1, 1, *depth_m.shape)
    mask_cv2 = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    mask_pt = torch.from_numpy(mask_cv2.astype(np.int32)).to(device).view(1, 1, *mask_cv2.shape)
    # --- Sky Depth Logic ---
    sky_mask = (mask_pt == SKY_CLASS)
    non_sky_mask = ~sky_mask
    
    if sky_mask.any():
        # Compute average of valid non-sky areas
        # This now benefits from the previous clamping of non-sky outliers
        avg_non_sky_depth = depth_pt[non_sky_mask].mean()
        depth_pt[sky_mask] = avg_non_sky_depth
        
    # Resize
    depth_pt = F.interpolate(depth_pt, size=size, mode='bilinear')
    mask_pt = F.interpolate(mask_pt.float(), size=size, mode='nearest').long()

    return depth_pt

if __name__ == "__main__":
    args = parse_args()
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

    embed_layers = None

    pipe.load_lora(pipe.dit, args.lora_checkpoint_path, alpha=1)

    for img_name in os.listdir(os.path.join(args.synthia_folder, "RGB")):
        input_image_path = os.path.join(args.synthia_folder, "RGB", img_name)
        depth_path, mask_path = get_related_paths(input_image_path)
        output_image_path = os.path.join(args.output_folder, "RGB", img_name)
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

        image_in_pil = Image.open(input_image_path).convert("RGB")
        w,h = image_in_pil.size
        if args.height is not None and args.width is not None:
            use_original_size = False
            new_w, new_h = args.width, args.height
        else:
            use_original_size = True
            new_w, new_h = w//16*16,h//16*16
        
        image_in_pil = image_in_pil.resize((new_w, new_h), resample=Image.LANCZOS)
        depth_map = load_and_preprocess_synthia_data(depth_path, mask_path, (new_h, new_w), device=pipe.device)
        prompt = args.prompt
        with torch.no_grad():
            image = pipe.preprocess_image(image_in_pil).to(device=pipe.device, dtype=pipe.torch_dtype)
            input_latents = pipe.vae_encoder(image, tiled=False)

            input_noise = torch.randn_like(input_latents)
            noise = generate_wavelet_structured_noise_batch_vectorized(
                image_batch=input_latents,
                depth_map=depth_map, 
                cutoff_radius=args.cutoff_radius,
                maximal_radius=args.maximal_radius,
                gamma=args.gamma,
                noise_std=1.0
            )
            noise = noise.contiguous()

            negative_prompt = args.negative_prompt

            image = pipe(
                prompt=prompt, negative_prompt=negative_prompt,
                height=new_h, width=new_w,
                cfg_scale=2, num_inference_steps=50, noise=noise
            )

            if use_original_size:
                image = image.resize((w,h))
            os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
            image.save(output_image_path)