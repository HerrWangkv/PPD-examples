import argparse
import torch
import os
import cv2
import numpy as np
import torch.nn.functional as F
from PIL import Image
import gc

# Import DiffSynth components
from diffsynth import download_models, save_video
from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig as FluxModelConfig
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig as WanModelConfig

# Import Wavelet Noise logic
from wavelet_noise import generate_wavelet_structured_noise_batch_vectorized

def parse_args():
    parser = argparse.ArgumentParser(description="PPD Pipeline: Flux Re-render -> Wan Video Generation")
    
    # --- Input/Output ---
    parser.add_argument("--rgb_video", type=str, required=True, help="Path to input RGB video (for structure/motion)")
    parser.add_argument("--depth_video", type=str, required=True, help="Path to input Depth/Disparity video (for structure)")
    parser.add_argument("--output_video", type=str, default="output_wpd.mp4", help="Path for final output video")
    parser.add_argument("--prompt", type=str, default="A photorealistic driving scene in a city, view from a car dashboard. Natural lighting, urban buildings, trees, cars on the street. High resolution, realistic textures.", help="Prompt for generation")
    parser.add_argument("--image-only", action="store_true", help="If set, only runs the Flux stage and saves the first frame image.")

    # --- Flux Arguments ---
    parser.add_argument("--flux_lora", type=str, default="models/ppd/flux1-dev_phipd_lora_302000.safetensors")
    parser.add_argument("--flux_cutoff_radius", type=int, default=20, help="Flux: Near degradation radius")
    parser.add_argument("--flux_maximal_radius", type=int, default=40, help="Flux: Far preservation radius")
    parser.add_argument("--flux_gamma", type=float, default=10, help="Flux: Depth curve control")

    # --- Wan Arguments ---
    parser.add_argument("--wan_low_lora", type=str, default="models/ppd/wan2.2-14b-low-step-12400.safetensors")
    parser.add_argument("--wan_high_lora", type=str, default="models/ppd/wan2.2-14b-high-step-12400.safetensors")
    parser.add_argument("--wan_cutoff_radius", type=int, default=40, help="Wan: Radius for structured noise")
    parser.add_argument("--wan_maximal_radius", type=int, default=40, help="Wan: Max Radius (usually > cutoff)")
    parser.add_argument("--wan_gamma", type=float, default=1)
    parser.add_argument("--n_frames", type=int, default=49)
    parser.add_argument("--fps", type=int, default=10)
    
    # --- General ---
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true", help="If set, only run the first window.")
    
    return parser.parse_args()

def load_frames(video_path, height, width, n_frames=None):
    """Loads video frames as PIL images."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
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
    
    # Loop if too short
    if n_frames and len(frames) < n_frames and len(frames) > 0:
        frames = frames + [frames[-1]] * (n_frames - len(frames))
        
    return frames

def load_depth_frames_as_tensor(video_path, height, width, n_frames, device):
    """
    Loads depth video frames and converts to Disparity Tensor (T, 1, H, W).
    Assumes Input Video is [0, 255] where 255 (White) = Near (High Disparity).
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Read as grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_LINEAR)
        frames.append(gray)
        if n_frames and len(frames) >= n_frames:
            break
    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames found in depth video: {video_path}")
        
    # Loop if too short
    if n_frames and len(frames) < n_frames:
        frames = frames + [frames[-1]] * (n_frames - len(frames))
    
    # Stack and Normalize
    # Shape: (T, H, W)
    depth_stack = np.stack(frames[:n_frames], axis=0).astype(np.float32)
    
    # Normalize 0-255 -> 0.0-1.0
    # We treat this as DISPARITY (White=Near) to match common visualization and PPD training logic
    disparity_norm = depth_stack / 255.0 
    
    # Convert to Tensor (T, 1, H, W)
    # Keep on CPU to save VRAM for long videos
    disparity_tensor = torch.from_numpy(disparity_norm).unsqueeze(1)
    
    return disparity_tensor

def flush():
    gc.collect()
    torch.cuda.empty_cache()

def run_flux_stage(args, first_frame_pil, first_frame_disparity, device):
    """
    Re-renders the first frame using Flux WPD.
    first_frame_disparity: (1, 1, H, W) tensor
    """
    print(f"--- Stage 1: Flux Re-rendering ---")
    
    download_models(["FLUX.1-dev"])
    pipe = FluxImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            FluxModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="flux1-dev.safetensors"),
            FluxModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors"),
            FluxModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/"),
            FluxModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors"),
        ],
    )
    
    if os.path.exists(args.flux_lora):
        pipe.load_lora(pipe.dit, args.flux_lora, alpha=1.0)
    else:
        print(f"Warning: Flux LoRA not found: {args.flux_lora}")

    with torch.no_grad():
        # Encode original image to get content structure (Phase)
        image_tensor = pipe.preprocess_image(first_frame_pil).to(device=device, dtype=pipe.torch_dtype)
        input_latents = pipe.vae_encoder(image_tensor, tiled=False)
        
        # Generate Wavelet Noise
        # Using Disparity directly (no inversion needed if input is Disparity)
        noise = generate_wavelet_structured_noise_batch_vectorized(
            image_batch=input_latents,
            disparity_map=first_frame_disparity, # Passed as disparity
            cutoff_radius=args.flux_cutoff_radius,
            maximal_radius=args.flux_maximal_radius,
            gamma=args.flux_gamma,
            noise_std=1.0
        ).contiguous()

        # Generate
        generated_image = pipe(
            prompt=args.prompt,
            negative_prompt="ugly, low quality, CG, Render, unreal, game, cartoon, blur, low res",
            height=args.height, width=args.width,
            cfg_scale=2,
            num_inference_steps=50,
            noise=noise
        )
    
    # Cleanup
    del pipe
    flush()
    
    # Save debug
    debug_path = args.output_video.replace(".mp4", "_flux_first_frame.png")
    generated_image.save(debug_path)
    print(f"Flux output saved to {debug_path}")
    
    return generated_image

def run_wan_stage(args, first_frame_gen, rgb_frames_pil, disparity_video_tensor, device):
    """
    Generates video using Wan2.2 WPD.
    rgb_frames_pil: List of PIL images (Original Video)
    disparity_video_tensor: (T, 1, H, W)
    """
    print("--- Stage 2: Wan Video Generation ---")
    
    download_models(["Wan2.2-I2V-A14B"])
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            WanModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors", offload_device="cpu"),
            WanModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors", offload_device="cpu"),
            WanModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
            WanModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
        ],
    )
    
    # Load LoRAs
    if args.wan_high_lora and os.path.exists(args.wan_high_lora):
        pipe.load_lora(pipe.dit, args.wan_high_lora, alpha=1.0)
    if args.wan_low_lora and os.path.exists(args.wan_low_lora):
        pipe.load_lora(pipe.dit2, args.wan_low_lora, alpha=1.0)

    pipe.load_lora(pipe.dit, "models/ppd/high_noise_model_converted.safetensors", alpha=8.0/64)
    pipe.load_lora(pipe.dit2, "models/ppd/low_noise_model_converted.safetensors", alpha=8.0/64)
    pipe.enable_vram_management()
    
    # Sliding Window params
    window_size = args.n_frames
    stride = window_size - 1
    total_frames = len(rgb_frames_pil)
    
    final_video_frames = []
    current_condition_image = first_frame_gen
    
    print(f"Processing {total_frames} frames in windows of {window_size} (stride {stride})...")
    
    for start_idx in range(0, total_frames, stride):
        end_idx = start_idx + window_size
        print(f"Generating window: {start_idx} to {end_idx}")
        
        # Prepare Inputs for chunk
        # Slice RGB Frames
        chunk_frames = rgb_frames_pil[start_idx : end_idx]
        
        # Slice Disparity
        # disparity_video_tensor: (Total, 1, H, W)
        # Note: Disparity tensor might be shorter or longer than rgb if issues, but assuming matched.
        # Need to handle padding for disparity too.
        
        if len(chunk_frames) < window_size:
            pad_count = window_size - len(chunk_frames)
            # Pad List
            chunk_frames = chunk_frames + [chunk_frames[-1]] * pad_count
            
        # Handle Disparity Slice
        # Safely slice.
        chunk_disparity = disparity_video_tensor[start_idx : end_idx]
        if chunk_disparity.shape[0] < window_size:
            pad_count_disp = window_size - chunk_disparity.shape[0]
            # Pad via repetition of last frame
            last_frame = chunk_disparity[-1:] # (1, 1, H, W)
            padding = last_frame.repeat(pad_count_disp, 1, 1, 1)
            chunk_disparity = torch.cat([chunk_disparity, padding], dim=0)

        # Move chunk to device for processing
        chunk_disparity = chunk_disparity.to(device)

        # Preprocess video for VAE
        # Wan pipeline expects list of PIL images
        with torch.no_grad():
            pipe.load_models_to_device(["vae"])
            
            # Encode Original Video -> Latents
            # Input to VAE should be (B, C, T, H, W) or list logic handled by pipeline
            # pipe.preprocess_video returns tensor (1, C, T, H, W)
            pixel_values = pipe.preprocess_video(chunk_frames).to(device=device, dtype=torch.bfloat16)
            input_latents = pipe.vae.encode(pixel_values, device=device, tiled=True)
            # input_latents shape: (1, 16, T_lat, H_lat, W_lat) usually
            
            # 2. Prepare Disparity for Noise Generation
            # Disparity shape (T_chunk, 1, H, W). Need to interpolate to latent T and Spatial size.
            _, C, T_lat, H_lat, W_lat = input_latents.shape
            
            # Interpolate Disparity to (T_lat, 1, H_lat, W_lat)
            # Permute to (1, 1, T, H, W) for interpolation
            target_disparity = chunk_disparity.permute(1, 0, 2, 3).unsqueeze(0).float()
            target_disparity = F.interpolate(
                target_disparity, 
                size=(T_lat, H_lat, W_lat), 
                mode="trilinear", 
                align_corners=False
            )
            # Back to (T_lat, 1, H_lat, W_lat)
            target_disparity = target_disparity.squeeze(0).permute(1, 0, 2, 3)
            
            # Sky Mask Logic (from original code)
            sky_mask = (target_disparity < 0.01).bool()
            if (~sky_mask).any():
                target_disparity[sky_mask] = target_disparity[~sky_mask].mean()

            # 3. Generate WPD Noise
            # input_latents[0] is (C, T, H, W). 
            # Wavelet function expects (Batch, C, H, W). We treat T as Batch.
            # Transpose to (T, C, H, W)
            latents_for_noise = input_latents[0].transpose(0, 1).float()
            
            input_noise_random = torch.randn_like(latents_for_noise)
            
            structured_noise = generate_wavelet_structured_noise_batch_vectorized(
                image_batch=latents_for_noise,
                disparity_map=target_disparity,
                cutoff_radius=args.wan_cutoff_radius,
                maximal_radius=args.wan_maximal_radius,
                gamma=args.wan_gamma,
                input_noise=input_noise_random
            )
            # Transpose back to (1, C, T, H, W) for pipeline
            structured_noise = structured_noise.transpose(0, 1).unsqueeze(0).to(dtype=pipe.torch_dtype, device=device)

            # 4. Generate Video
            video_chunk = pipe(
                prompt=args.prompt,
                negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，卡通，渲染，游戏，CG，render, simulation, game, cartoon, 3D",
                tiled=True,
                input_image=current_condition_image, # Condition on re-rendered first frame (or last of prev)
                input_noise=structured_noise, # Preserves structure of original video
                height=args.height, width=args.width,
                num_frames=window_size,
                switch_DiT_boundary=0.9,
                cfg_scale=1,
                num_inference_steps=4,
            )
            
        # Accumulate
        if start_idx == 0:
            final_video_frames.extend(video_chunk)
        else:
            final_video_frames.extend(video_chunk[1:])
            
        current_condition_image = video_chunk[-1]
        
        del video_chunk
        flush()
        if args.debug:
            print("Debug mode: stopping after first window.")
            break
    
    # Trim
    final_video_frames = final_video_frames[:total_frames]
        
    save_video(final_video_frames, args.output_video, fps=args.fps, quality=5)
    print(f"Final video saved to {args.output_video}")

if __name__ == "__main__":
    args = parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load Data
    print("Loading data...")
    rgb_frames = load_frames(args.rgb_video, args.height, args.width, n_frames=None)
    disparity_tensor = load_depth_frames_as_tensor(args.depth_video, args.height, args.width, n_frames=None, device=device)
    
    # 1. Flux Stage (First Frame)
    first_frame_pil = rgb_frames[0]
    # Use first frame of disparity
    first_frame_disp = disparity_tensor[0].unsqueeze(0) # (1, 1, H, W)
    
    flux_output = run_flux_stage(args, first_frame_pil, first_frame_disp, device)
    
    # 2. Wan Stage (Full Video)
    if args.image_only:
        print("Image-only flag set; skipping Wan video generation.")
    else:
        run_wan_stage(args, flux_output, rgb_frames, disparity_tensor, device)