from structured_noise import generate_structured_noise_batch_vectorized
import argparse
import torch
import os
import math
import torch.distributed as dist
from PIL import Image
from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig
from diffsynth import download_models

def parse_args():
    parser = argparse.ArgumentParser(description="Generate videos with trained model")
    parser.add_argument(
        "--timestep",
        type=float,
        required=True,
        help="t_0 in SDEdit"
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
        default="data/synthia_sdedit",
        help="Output folder"
    )
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

if __name__ == "__main__":
    args = parse_args()
    
    # 1. Setup Distributed Environment
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"Initializing DDP: Rank {rank}/{world_size} on device {device}")
        download_models(["FLUX.1-dev"])
    
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

    # =========================================================================
    # 3. GLOBAL FILTERING
    # =========================================================================
    rgb_folder = os.path.join(args.synthia_folder, "RGB")
    all_files = sorted(os.listdir(rgb_folder))
    valid_images = [f for f in all_files if f.endswith(('.png', '.jpg', '.jpeg'))]

    # Check existence BEFORE assigning to GPUs
    # This ensures that if you restart a run, the remaining 1000 images 
    # are split evenly (125 per GPU) rather than based on their original filenames.
    images_to_process = []
    
    # It is safe for all ranks to do this fs check (metadata read is fast)
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

    # 4. Distribute ONLY the remaining work
    my_images = images_to_process[rank::world_size]

    if len(my_images) == 0:
        print(f"[GPU {rank}] No work assigned. Exiting.")
        if world_size > 1: dist.destroy_process_group()
        exit(0)

    print(f"[GPU {rank}] Assigned {len(my_images)} tasks.")

    # 5. Processing Loop
    for img_name in my_images:
        input_image_path = os.path.join(args.synthia_folder, "RGB", img_name)
        output_image_path = os.path.join(args.output_folder, "RGB", img_name)
        
        # Double-check (in case of race conditions, though rare here)
        if os.path.exists(output_image_path):
            continue

        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

        # ... [Standard Image Processing Code] ...
        image_in_pil = Image.open(input_image_path).convert("RGB")
        w, h = image_in_pil.size
        
        if args.height is not None and args.width is not None:
            use_original_size = False
            new_w, new_h = args.width, args.height
        else:
            use_original_size = True
            new_w, new_h = w // 16 * 16, h // 16 * 16
            
        image_in_pil = image_in_pil.resize((new_w, new_h), resample=Image.LANCZOS)
        prompt = args.prompt
        
        with torch.no_grad():
            image = pipe.preprocess_image(image_in_pil).to(device=pipe.device, dtype=pipe.torch_dtype)
            input_latents = pipe.vae_encoder(image, tiled=False)

            input_noise = torch.randn_like(input_latents)
            noise = input_noise.contiguous().to(device)
            image = pipe(
                prompt=prompt, 
                negative_prompt=args.negative_prompt,
                height=new_h, width=new_w,
                denoising_strength=args.timestep,
                cfg_scale=2, num_inference_steps=50, noise=noise
            )

            if use_original_size:
                image = image.resize((w, h))
            
            image.save(output_image_path)
            print(f"[GPU {rank}] Generated: {img_name}")

    if world_size > 1:
        dist.destroy_process_group()