import torch
import cv2
import argparse
from wavelet_noise import generate_wavelet_structured_noise_batch_vectorized
from PIL import Image
from diffsynth import save_video, download_models
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Generate videos with trained model")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--n_frames", type=int, default=49, help="Number of frames to generate")
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument(
        "--lora_low_checkpoint_path",
        type=str,
        default="models/ppd/wan2.2-i2v-a14b_color_low_step=3600_biased.safetensors",
    )
    parser.add_argument(
        "--lora_high_checkpoint_path",
        type=str,
        default="models/ppd/wan2.2-i2v-a14b_color_high_step=3600_biased.safetensors",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=35,
        help="Wavelet noise radius",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A photorealistic driving scene in a city, view from a car dashboard. Natural lighting, urban buildings, trees, cars on the street. High resolution, realistic textures.",
    )
    parser.add_argument("--input_image", type=str, default="output.png", help="Input conditioning image")
    parser.add_argument("--input_video", type=str, default="input.mp4", help="Input video for structure")
    parser.add_argument("--output_name", type=str, default="output.mp4")
    return parser.parse_args()

def load_video_to_pil_images(video_path, height, width):
    cap = cv2.VideoCapture(video_path)
    images = []

    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb).resize((width, height), resample=Image.LANCZOS)
        images.append(pil_image)

    cap.release()
    return images


if __name__ == "__main__":
    args = parse_args()
    download_models(["Wan2.2-I2V-A14B"])
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda:0",
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors", offload_device="cpu"),
            ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors", offload_device="cpu"),
            ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
            ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
        ],
    )
    if args.lora_high_checkpoint_path and args.lora_low_checkpoint_path:
        pipe.load_lora(pipe.dit, args.lora_high_checkpoint_path, alpha=1)
        pipe.load_lora(pipe.dit2, args.lora_low_checkpoint_path, alpha=1)
    else:
        print("No checkpoint provided, use original model")
    pipe.load_lora(pipe.dit, "models/ppd/high_noise_model_converted.safetensors", alpha=8.0/64)
    pipe.load_lora(pipe.dit2, "models/ppd/low_noise_model_converted.safetensors", alpha=8.0/64)

    pipe.enable_vram_management()
    height = args.height
    width = args.width
    n_frame = args.n_frames

    frames = load_video_to_pil_images(args.input_video, height=height, width=width)[:n_frame]
    for i, img in enumerate(frames):
        frames[i] = img.convert("RGB")
    if len(frames) < n_frame:
        frames = frames + [frames[-1]] * (n_frame - len(frames))
    frames = frames[:n_frame]

    pipe.load_models_to_device(["vae"])
    with torch.no_grad():
        input_video = pipe.preprocess_video(frames)
        input_latents = pipe.vae.encode(input_video, device=pipe.device, tiled=True)

        # (T, C, H, W) — treat temporal frames as batch
        latents_for_noise = input_latents[0].transpose(0, 1).cpu().float()

        noise = generate_wavelet_structured_noise_batch_vectorized(
            image_batch=latents_for_noise,
            radius_map=args.radius,
            input_noise=torch.randn_like(latents_for_noise),
        )
        # Back to (1, C, T, H, W)
        noise = noise.transpose(0, 1).unsqueeze(0).contiguous()
        noise = noise.to(dtype=pipe.torch_dtype, device=pipe.device)

        input_image = Image.open(args.input_image).resize((width, height), resample=Image.LANCZOS)
        negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，卡通，渲染，游戏，CG，render, simulation, game, cartoon, 3D"
        video = pipe(
            prompt=args.prompt,
            negative_prompt=negative_prompt,
            tiled=True,
            input_image=input_image,
            switch_DiT_boundary=0.9,
            height=height, width=width,
            num_frames=n_frame,
            input_noise=noise,
            cfg_scale=1,
            num_inference_steps=4,
        )
    save_video(video, args.output_name, fps=args.fps, quality=5)
