import argparse
import torch
import os
import cv2
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
from PIL import Image
from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig
from diffsynth import download_models
from wavelet_noise import generate_wavelet_structured_noise_batch_vectorized

# SYNTHIA Dataset Mapping
SKY_CLASS = 10 
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

def setup_distributed():
    """Initializes the distributed backend for DDP."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return rank, world_size, local_rank
    else:
        # Fallback for single GPU/CPU run
        print("Not running in distributed mode.")
        return 0, 1, 0

def get_related_paths(image_path):
    """
    Infers Depth and Mask paths from SYNTHIA structure.
    """
    base_name = os.path.basename(image_path)
    root = image_path.split("/RGB/")[0]
    
    depth_path = os.path.join(root, "Depth", "Depth", base_name)
    mask_path = os.path.join(root, "GT", "LABELS", base_name.replace(".png", "_labelTrainIds.png"))
    
    return depth_path, mask_path

def load_and_preprocess_synthia_data(depth_path, mask_path, size, device, max_depth_limit=500.0):
    """Loads depth and handles sky/outlier depth values."""
    depth_cv2 = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_cv2 is None: raise FileNotFoundError(f"Depth not found: {depth_path}")
    
    if len(depth_cv2.shape) == 3:
        depth_cv2 = np.max(depth_cv2, axis=2)
    
    depth_m = depth_cv2.astype(np.float32) / 100.0
    depth_m = np.clip(depth_m, 0.0, max_depth_limit)
    
    depth_pt = torch.from_numpy(depth_m).to(device).view(1, 1, *depth_m.shape)
    mask_cv2 = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    mask_pt = torch.from_numpy(mask_cv2.astype(np.int32)).to(device).view(1, 1, *mask_cv2.shape)
    
    sky_mask = (mask_pt == SKY_CLASS)
    non_sky_mask = ~sky_mask
    
    if sky_mask.any():
        avg_non_sky_depth = depth_pt[non_sky_mask].mean()
        depth_pt[sky_mask] = avg_non_sky_depth
        
    depth_pt = F.interpolate(depth_pt, size=size, mode='bilinear')
    mask_pt = F.interpolate(mask_pt.float(), size=size, mode='nearest').long()

    return depth_pt

if __name__ == "__main__":
    args = parse_args()
    
    # 1. Setup Distributed Environment
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"Initializing DDP: Rank {rank}/{world_size} on device {device}")
        download_models(["FLUX.1-dev"])
    
    # Wait for Rank 0 to finish downloading
    if world_size > 1:
        dist.barrier()

    # 2. Load Model
    pipe = FluxImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="flux1-dev.safetensors"),
            ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors"),
            ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/"),
            ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors"),
        ],
    )

    embed_layers = None
    pipe.load_lora(pipe.dit, args.lora_checkpoint_path, alpha=1)

    # =========================================================================
    # 3. GLOBAL FILTERING & WORKLOAD DISTRIBUTION
    # =========================================================================
    rgb_folder = os.path.join(args.synthia_folder, "RGB")
    all_files = sorted(os.listdir(rgb_folder))
    valid_images = [f for f in all_files if f.endswith(('.png', '.jpg', '.jpeg'))]

    # Filter out already processed images to support resuming
    images_to_process = []
    for img_name in valid_images:
        output_path = os.path.join(args.output_folder, "RGB", img_name)
        if not os.path.exists(output_path):
            images_to_process.append(img_name)

    total_count = len(valid_images)
    todo_count = len(images_to_process)

    if rank == 0:
        print(f"Total Dataset: {total_count}")
        print(f"Already Done:  {total_count - todo_count}")
        print(f"Remaining:     {todo_count} (Distributing these among {world_size} GPUs)")

    # Distribute remaining work
    my_images = images_to_process[rank::world_size]

    if len(my_images) == 0:
        print(f"[GPU {rank}] No work assigned. Exiting.")
        if world_size > 1: dist.destroy_process_group()
        exit(0)

    print(f"[GPU {rank}] Assigned {len(my_images)} tasks.")

    # 4. Processing Loop
    for img_name in my_images:
        input_image_path = os.path.join(args.synthia_folder, "RGB", img_name)
        depth_path, mask_path = get_related_paths(input_image_path)
        output_image_path = os.path.join(args.output_folder, "RGB", img_name)
        
        # Double check existence (rare race condition or user intervention)
        if os.path.exists(output_image_path):
            continue

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
        # Note: Pass the correct DDP device here
        depth_map = load_and_preprocess_synthia_data(depth_path, mask_path, (new_h, new_w), device=pipe.device)
        prompt = args.prompt
        
        with torch.no_grad():
            image = pipe.preprocess_image(image_in_pil).to(device=pipe.device, dtype=pipe.torch_dtype)
            input_latents = pipe.vae_encoder(image, tiled=False)

            # Generate structured noise
            noise = generate_wavelet_structured_noise_batch_vectorized(
                image_batch=input_latents,
                depth_map=depth_map, 
                cutoff_radius=args.cutoff_radius,
                maximal_radius=args.maximal_radius,
                gamma=args.gamma,
                noise_std=1.0
            )
            noise = noise.contiguous().to(device) # Ensure it's on the right device

            negative_prompt = args.negative_prompt
            image = pipe(
                prompt=prompt, negative_prompt=negative_prompt,
                height=new_h, width=new_w,
                cfg_scale=2, num_inference_steps=50, noise=noise
            )

            if use_original_size:
                image = image.resize((w,h))
            
            image.save(output_image_path)
            print(f"[GPU {rank}] Generated: {img_name}")

    if world_size > 1:
        dist.destroy_process_group()