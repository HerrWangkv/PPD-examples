"""T2V translation test on a single scene — no FLUX stage, no first-frame reference."""
import torch, os, gc
import numpy as np
from PIL import Image

from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig as WanModelConfig
from wavelet_noise import generate_wavelet_structured_noise_batch_vectorized
from sim2real_video_wavelet import load_frames

INPUT_VIDEO   = "outputs/nucarla/input/rgb/scene_0032.mp4"
OUTPUT_VIDEO  = "outputs/nucarla/t2v_test/scene_0032_t2v_dropll_J5_r8.mp4"
WAN_HIGH_LORA = "models/train/Wan2.2-I2V-A14B_high_lora_wpd_dropll/step-200.safetensors"
WAN_LOW_LORA  = "models/train/Wan2.2-I2V-A14B_low_lora_wpd_dropll/step-400.safetensors"
PROMPT = "A photo-realistic street scene captured from a driving car, real-world urban environment, natural lighting, high quality."
CUTOFF_RADIUS = 8
DROP_LL = True
J = 5
N_FRAMES = 49
HEIGHT, WIDTH = 704, 1280
FPS = 10

os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)
device = "cuda:0"

rgb_frames, _ = load_frames(INPUT_VIDEO, HEIGHT, WIDTH, n_frames=N_FRAMES)
print(f"Loaded {len(rgb_frames)} frames")

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device=device,
    model_configs=[
        WanModelConfig(model_id="Wan-AI/Wan2.2-T2V-A14B", origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        WanModelConfig(model_id="Wan-AI/Wan2.2-T2V-A14B", origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        WanModelConfig(model_id="Wan-AI/Wan2.2-T2V-A14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        WanModelConfig(model_id="Wan-AI/Wan2.2-T2V-A14B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
    ],
)
pipe.load_lora(pipe.dit,  WAN_HIGH_LORA, alpha=1.0)
pipe.load_lora(pipe.dit2, WAN_LOW_LORA,  alpha=1.0)
pipe.load_lora(pipe.dit,  "models/ppd/high_noise_model_converted.safetensors", alpha=8.0/64)
pipe.load_lora(pipe.dit2, "models/ppd/low_noise_model_converted.safetensors",  alpha=8.0/64)
pipe.enable_vram_management()

with torch.no_grad():
    pipe.load_models_to_device(["vae"])
    pixel_values = pipe.preprocess_video(rgb_frames).to(device=device, dtype=torch.bfloat16)
    input_latents = pipe.vae.encode(pixel_values, device=device, tiled=True)

    latents_for_noise = input_latents[0].transpose(0, 1).float()
    structured_noise = generate_wavelet_structured_noise_batch_vectorized(
        image_batch=latents_for_noise,
        radius_map=CUTOFF_RADIUS,
        drop_ll=DROP_LL,
        J=J,
        input_noise=torch.randn_like(latents_for_noise),
    )
    structured_noise = structured_noise.transpose(0, 1).unsqueeze(0).to(dtype=pipe.torch_dtype, device=device)

    video_frames = pipe(
        prompt=PROMPT,
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，卡通，渲染，游戏，CG，render, simulation, game, cartoon, 3D",
        tiled=True,
        input_noise=structured_noise,
        height=HEIGHT, width=WIDTH,
        num_frames=N_FRAMES,
        switch_DiT_boundary=0.9,
        cfg_scale=1,
        num_inference_steps=4,
    )

save_video(video_frames, OUTPUT_VIDEO, fps=FPS, quality=5)
print(f"Saved to {OUTPUT_VIDEO}")
