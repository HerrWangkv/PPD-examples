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
        required=False,
        default="models/ppd/flux1-dev_lora_color_step=266000_biased.safetensors",
        help="Path to lora checkpoint file"
    )
    parser.add_argument(
        "--input_image",
        type=str,
        default="data/synthia/RGB/0000820.png", # Changed default to hint at dataset structure
        help="Input image filename. Script assumes standard SYNTHIA structure to find GT."
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="output.png",
        help="Output image filename"
    )
    parser.add_argument(
        "--radius", 
        type=int, 
        default=30, 
        help="Pixel radius for near degradation (heavy noise)")
    parser.add_argument(
        "--prompt",
        type=str,
        default="A photorealistic driving scene in a European city. Natural lighting, detailed asphalt road, urban buildings, trees, cars on the street. High resolution, cinematic, realistic textures, automotive photography."
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="ugly, low quality, CG, Render, unreal, game, cartoon, blur, low res",
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

if __name__ == "__main__":
    args = parse_args()
    device = "cuda"
    
    # 1. Download and Init Pipe
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

    # 2. Process Input Image
    image_in_pil = Image.open(args.input_image).convert("RGB")
    w, h = image_in_pil.size
    if args.height is not None and args.width is not None:
        use_original_size = False
        new_w, new_h = args.width, args.height
    else:
        use_original_size = True
        new_w, new_h = w // 16 * 16, h // 16 * 16
    
    image_in_pil = image_in_pil.resize((new_w, new_h), resample=Image.LANCZOS)

    # 3. Diffusion Process
    prompt = args.prompt
    with torch.no_grad():
        image = pipe.preprocess_image(image_in_pil).to(device=pipe.device, dtype=pipe.torch_dtype)
        input_latents = pipe.vae_encoder(image, tiled=False)

        input_noise = torch.randn_like(input_latents)
        # Generate Structured Noise guided by Depth Control Map
        noise = generate_wavelet_structured_noise_batch_vectorized(
            image_batch=input_latents,
            radius_map=args.radius,
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
            image = image.resize((w, h))
        os.makedirs(os.path.dirname(args.output_name), exist_ok=True)
        image.save(args.output_name)
        print(f"Output saved to {args.output_name}")