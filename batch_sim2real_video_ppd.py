import argparse
import torch
import os
import cv2
import numpy as np
import torch.nn.functional as F
from PIL import Image
import gc
import glob

# Import DiffSynth components
from diffsynth import download_models, save_video
from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig as FluxModelConfig
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig as WanModelConfig

# Import Wavelet Noise logic
from structured_noise import generate_structured_noise_batch_vectorized
import torch.distributed as dist

def init_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return True, rank, world, local_rank
    return False, 0, 1, 0

def parse_args():
    parser = argparse.ArgumentParser(description="Batch PPD Pipeline")
    
    # --- Batch Input/Output ---
    parser.add_argument("--input_dataset", type=str, required=True, help="Path to input dataset folder")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output folder for re-rendered videos")

    # --- Flux Arguments ---
    parser.add_argument("--flux_lora", type=str, default="models/ppd/flux1-dev_phipd_lora_302000.safetensors")
    parser.add_argument("--flux_cutoff_radius", type=int, default=20, help="Flux: Near degradation radius")
    parser.add_argument("--prompt", type=str, default="A photorealistic driving scene in a city, view from a car dashboard. Natural lighting, urban buildings, trees, cars on the street. High resolution, realistic textures.", help="Prompt for generation")

    # --- Wan Arguments ---
    parser.add_argument("--wan_low_lora", type=str, default="models/ppd/wan2.2-14b-low-step-12400.safetensors")
    parser.add_argument("--wan_high_lora", type=str, default="models/ppd/wan2.2-14b-high-step-12400.safetensors")
    parser.add_argument("--wan_cutoff_radius", type=int, default=40, help="Wan: Radius for structured noise")
    parser.add_argument("--n_frames", type=int, default=49)
    
    # --- General ---
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--seed", type=int, default=42)
    
    return parser.parse_args()

def load_frames(video_path, height, width, n_frames=None):
    """Loads video frames as PIL images and returns the original FPS."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get the input video's FPS
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame).resize((width, height), Image.LANCZOS)
        frames.append(img)
        if n_frames and len(frames) >= n_frames:
            break
    cap.release()
    
    if n_frames and len(frames) < n_frames and len(frames) > 0:
        frames = frames + [frames[-1]] * (n_frames - len(frames))
        
    return frames, input_fps

def flush():
    gc.collect()
    torch.cuda.empty_cache()

def process_video(args, rgb_video_path, output_video_path, flux_pipe, wan_pipe, device):
    # Load Data
    print(f"Loading {rgb_video_path}...")
    try:
        rgb_frames, input_fps = load_frames(rgb_video_path, args.height, args.width, n_frames=None)
    except Exception as e:
        print(f"Error loading {rgb_video_path}: {e}")
        return

    # 1. Flux Stage (First Frame)
    first_frame_pil = rgb_frames[0]
    generated_image = None
    
    print(f"--- Stage 1: Flux Re-rendering ---")
    # Ensure Flux models are available
    if os.path.exists(args.flux_lora):
        # Already loaded in main, just proceed
        pass
    else:
        print(f"Warning: Flux LoRA not found: {args.flux_lora}")

    with torch.no_grad():
        # Encode original image to get content structure (Phase)
        image_tensor = flux_pipe.preprocess_image(first_frame_pil).to(device=device, dtype=flux_pipe.torch_dtype)
        input_latents = flux_pipe.vae_encoder(image_tensor, tiled=False)
        
        # Generate PPD Noise
        noise = generate_structured_noise_batch_vectorized(
            image_batch=input_latents,
            cutoff_radius=args.flux_cutoff_radius,
        ).contiguous()

        # Generate
        generated_image = flux_pipe(
            prompt=args.prompt,
            negative_prompt="ugly, low quality, CG, Render, unreal, game, cartoon, blur, low res",
            height=args.height, width=args.width,
            cfg_scale=2,
            num_inference_steps=50,
            noise=noise
        )
    flush()

    # 2. Wan Stage (Full Video)
    print("--- Stage 2: Wan Video Generation ---")
    
    # Sliding Window Logic
    window_size = args.n_frames
    stride = window_size - 1
    total_frames = len(rgb_frames)
    
    final_video_frames = []
    current_condition_image = generated_image
    
    print(f"Processing {total_frames} frames in windows of {window_size} (stride {stride})...")
    
    for start_idx in range(0, total_frames, stride):
        end_idx = start_idx + window_size
        
        chunk_frames = rgb_frames[start_idx : end_idx]
        if len(chunk_frames) < window_size:
            pad_count = window_size - len(chunk_frames)
            chunk_frames = chunk_frames + [chunk_frames[-1]] * pad_count
        
        with torch.no_grad():
            wan_pipe.load_models_to_device(["vae"])
            pixel_values = wan_pipe.preprocess_video(chunk_frames).to(device=device, dtype=torch.bfloat16)
            input_latents = wan_pipe.vae.encode(pixel_values, device=device, tiled=True)
            
            latents_for_noise = input_latents[0].transpose(0, 1).float()
            input_noise_random = torch.randn_like(latents_for_noise)
            
            structured_noise = generate_structured_noise_batch_vectorized(
                image_batch=latents_for_noise,
                cutoff_radius=args.wan_cutoff_radius,
                input_noise=input_noise_random
            )
            structured_noise = structured_noise.transpose(0, 1).unsqueeze(0).to(dtype=wan_pipe.torch_dtype, device=device)

            video_chunk = wan_pipe(
                prompt=args.prompt,
                negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，卡通，渲染，游戏，CG，render, simulation, game, cartoon, 3D",
                tiled=True,
                input_image=current_condition_image, 
                input_noise=structured_noise, 
                height=args.height, width=args.width,
                num_frames=window_size,
                switch_DiT_boundary=0.9,
                cfg_scale=1,
                num_inference_steps=4,
            )
            
        if start_idx == 0:
            final_video_frames.extend(video_chunk)
        else:
            final_video_frames.extend(video_chunk[1:])

        current_condition_image = video_chunk[-1]
        
        del video_chunk
        flush()
    
    final_video_frames = final_video_frames[:total_frames]
    save_video(final_video_frames, output_video_path, fps=input_fps, quality=5)
    print(f"Saved: {output_video_path}")


if __name__ == "__main__":
    args = parse_args()
    is_ddp, rank, world, local_rank = init_distributed()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 1. Initialize Models ONCE
    print("Initializing Flux Model...")
    download_models(["FLUX.1-dev"])
    flux_pipe = FluxImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            FluxModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="flux1-dev.safetensors"),
            FluxModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors"),
            FluxModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/"),
            FluxModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors"),
        ],
    )
    # Load Flux LoRA Once
    if os.path.exists(args.flux_lora):
        print(f"Loading Flux LoRA: {args.flux_lora}")
        flux_pipe.load_lora(flux_pipe.dit, args.flux_lora, alpha=1.0)
    else:
        print(f"Warning: Flux LoRA not found: {args.flux_lora}")
    
    flux_pipe.enable_vram_management()

    print("Initializing Wan Model...")
    download_models(["Wan2.2-I2V-A14B"])
    wan_pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            WanModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors", offload_device="cpu"),
            WanModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors", offload_device="cpu"),
            WanModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
            WanModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
        ],
    )
    # Load Wan LoRAs Once
    if args.wan_high_lora and os.path.exists(args.wan_high_lora):
        print(f"Loading Wan High LoRA: {args.wan_high_lora}")
        wan_pipe.load_lora(wan_pipe.dit, args.wan_high_lora, alpha=1.0)
    if args.wan_low_lora and os.path.exists(args.wan_low_lora):
        print(f"Loading Wan Low LoRA: {args.wan_low_lora}")
        wan_pipe.load_lora(wan_pipe.dit2, args.wan_low_lora, alpha=1.0)

    wan_pipe.load_lora(wan_pipe.dit, "models/ppd/high_noise_model_converted.safetensors", alpha=8.0/64)
    wan_pipe.load_lora(wan_pipe.dit2, "models/ppd/low_noise_model_converted.safetensors", alpha=8.0/64)
    
    wan_pipe.enable_vram_management()

    # 2. Iterate Dataset
    rgb_dir = args.input_dataset
    video_files = glob.glob(os.path.join(rgb_dir, "*.mp4"))
    
    print(f"Found {len(video_files)} videos in {rgb_dir}")
    video_files = sorted(video_files)
    video_files = video_files[rank::world]   # each rank gets every Nth video
    print(f"[rank {rank}/{world}] Processing {len(video_files)} videos on {device}")

    for rgb_path in sorted(video_files):
        filename = os.path.basename(rgb_path)
        output_path = os.path.join(args.output_dir, filename)
        
        if os.path.exists(output_path):
            print(f"Skipping {filename} (already exists)")
            continue
            
        process_video(args, rgb_path, output_path, flux_pipe, wan_pipe, device)
