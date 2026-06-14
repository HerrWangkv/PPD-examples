"""
Night prompt comparison: WPD-base r12 vs WPD-drop_ll J=4 r12, N samples.

Usage (2 GPUs in parallel):
  CUDA_VISIBLE_DEVICES=0 python exp_weather_night.py --method_idx 0  # WPD-base
  CUDA_VISIBLE_DEVICES=1 python exp_weather_night.py --method_idx 1  # WPD-drop_ll

Images saved to figures/weather_night/<method>_img<i>.png
"""
import argparse, torch, random, glob, os
import numpy as np
from PIL import Image

from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig
from wavelet_noise import generate_wavelet_structured_noise_batch_vectorized

parser = argparse.ArgumentParser()
parser.add_argument("--method_idx", type=int, required=True, help="0=WPD-base, 1=WPD-drop_ll")
parser.add_argument("--n_samples", type=int, default=8)
args = parser.parse_args()

SEED  = 42
H, W  = 384, 1280
STEPS = 50
CFG   = 2
J     = 4
R     = 12

METHODS = [
    dict(name="WPD-base",
         lora="flux.safetensors",
         drop_ll=False),
    dict(name="WPD-drop_ll",
         lora="models/train/FLUX.1-dev_lora_wpd_dropll/step-6000.safetensors",
         drop_ll=True),
]

PROMPT = ("A photorealistic photo from a forward-facing vehicle camera, "
          "nighttime, dark scene, streetlights, headlight beams on wet road.")
NEG    = ("blurry, low quality, cartoon, cg render, unrealistic, "
          "dashboard, steering wheel, windshield frame")

method = METHODS[args.method_idx]
mname  = method["name"]
print(f"Method: {mname}")

os.makedirs("figures/weather_night", exist_ok=True)

all_imgs = sorted(glob.glob(
    "/mrtstorage/datasets_tmp/vkitti/vkitti_1.3.1_rgb/*/clone/*.png"))
random.seed(SEED)
samples = random.sample(all_imgs, args.n_samples)
print(f"Samples ({args.n_samples}):")
for p in samples: print(f"  {p}")

inputs_pil = [Image.open(p).convert("RGB").resize((W, H), Image.LANCZOS) for p in samples]

# Save inputs (idempotent)
for i, img in enumerate(inputs_pil):
    img.save(f"figures/weather_night/input_img{i}.png")

MODEL_CONFIGS = [
    ModelConfig(model_id="black-forest-labs/FLUX.1-dev",
                origin_file_pattern="flux1-dev.safetensors"),
    ModelConfig(model_id="black-forest-labs/FLUX.1-dev",
                origin_file_pattern="text_encoder/model.safetensors"),
    ModelConfig(model_id="black-forest-labs/FLUX.1-dev",
                origin_file_pattern="text_encoder_2/"),
    ModelConfig(model_id="black-forest-labs/FLUX.1-dev",
                origin_file_pattern="ae.safetensors"),
]

pipe = FluxImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16, device="cuda",
    model_configs=MODEL_CONFIGS,
)
pipe.load_lora(pipe.dit, method["lora"], alpha=1)

for i, img_pil in enumerate(inputs_pil):
    with torch.no_grad():
        img_t = pipe.preprocess_image(img_pil).to(pipe.device, dtype=pipe.torch_dtype)
        z_sim = pipe.vae_encoder(img_t, tiled=False)

    torch.manual_seed(SEED + i)
    input_noise = torch.randn(z_sim.shape, dtype=torch.float32, device="cpu")
    eps_struct = generate_wavelet_structured_noise_batch_vectorized(
        z_sim.float().cpu(), radius_map=R,
        input_noise=input_noise,
        drop_ll=method["drop_ll"], J=J,
    ).to(device=pipe.device, dtype=pipe.torch_dtype).contiguous()

    with torch.no_grad():
        out = pipe(
            prompt=PROMPT, negative_prompt=NEG,
            height=H, width=W,
            cfg_scale=CFG, num_inference_steps=STEPS,
            noise=eps_struct,
        )
    out_path = f"figures/weather_night/{mname}_img{i}.png"
    out.save(out_path)
    print(f"  [{i+1}/{args.n_samples}] saved: {out_path}")

print(f"Done: {mname}")
