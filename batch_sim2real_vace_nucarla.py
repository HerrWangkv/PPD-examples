"""
Batch VACE sim2real translation for 60 nuCarla scenes.

Uses Wan2.1-VACE-14B zero-shot: the synthetic CARLA video is passed as
vace_video (condition), no LoRA, no wavelet noise injection.

Usage (inside Docker):
    PYTHONPATH=. python batch_sim2real_vace_nucarla.py \
        --input_dir /mrtstorage/users/kwang/nucarla_videos/rgb \
        --output_dir outputs/nucarla/vace \
        --scene_start 0 --scene_end 59
"""
import argparse
import os
import gc
import glob
import cv2
import torch
from PIL import Image

from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig

PROMPT = (
    "A photorealistic driving scene filmed from a moving vehicle. "
    "Natural lighting, urban buildings, trees, cars on the street. "
    "High resolution, realistic textures."
)
NEGATIVE_PROMPT = (
    "ugly, low quality, CG, render, unreal, game, cartoon, blur, low res, "
    "dashboard, steering wheel, windshield frame, car interior, lens artifacts"
)
HEIGHT, WIDTH = 704, 1280
N_FRAMES = 49
SEED = 42


def load_frames_gray(video_path):
    """Load frames as grayscale-3ch for VACE gray control mode."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        frames.append(Image.fromarray(gray_rgb).resize((WIDTH, HEIGHT), Image.LANCZOS))
        if len(frames) >= N_FRAMES:
            break
    cap.release()
    while len(frames) < N_FRAMES:
        frames.append(frames[-1])
    return frames, fps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="/mrtstorage/users/kwang/nucarla_videos/rgb")
    parser.add_argument("--output_dir", default="outputs/nucarla/vace_gray")
    parser.add_argument("--scene_start", type=int, default=0)
    parser.add_argument("--scene_end", type=int, default=59)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Collect scenes to process
    scenes = []
    for i in range(args.scene_start, args.scene_end + 1):
        name = f"scene_{i:04d}"
        inp = os.path.join(args.input_dir, f"{name}.mp4")
        out = os.path.join(args.output_dir, f"{name}.mp4")
        if not os.path.exists(inp):
            print(f"[skip] {name}: input not found")
            continue
        if os.path.exists(out):
            print(f"[skip] {name}: already done")
            continue
        scenes.append((name, inp, out))

    if not scenes:
        print("All scenes done.")
        return

    print(f"{len(scenes)} scenes to process.")

    # Load model once
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda:0",
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B",
                        origin_file_pattern="diffusion_pytorch_model*.safetensors",
                        offload_device="cpu"),
            ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B",
                        origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
                        offload_device="cpu"),
            ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B",
                        origin_file_pattern="Wan2.1_VAE.pth",
                        offload_device="cpu"),
        ],
    )
    pipe.enable_vram_management()

    for name, inp, out in scenes:
        print(f"\n--- {name} ---")
        frames, fps = load_frames_gray(inp)
        video = pipe(
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            vace_video=frames,
            height=HEIGHT,
            width=WIDTH,
            num_frames=N_FRAMES,
            seed=SEED,
            tiled=True,
        )
        save_video(video, out, fps=fps, quality=5)
        print(f"Saved: {out}")
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
