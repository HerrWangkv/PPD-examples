"""
Weather/illumination controllability experiment — radius sweep, distributed.

Usage (one process per GPU, split by method):
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python exp_weather_prompts.py --method_idx 0
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 python exp_weather_prompts.py --method_idx 1
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=2 python exp_weather_prompts.py --method_idx 2

Images saved immediately to figures/weather/<method>_r<r>_img<i>_<prompt>.png
Compose grid after all finish: python plot_weather_prompts.py
"""
import argparse, torch, random, glob, os, gc
import numpy as np
from PIL import Image

from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig
from structured_noise import generate_structured_noise_batch_vectorized
from wavelet_noise import generate_wavelet_structured_noise_batch_vectorized

parser = argparse.ArgumentParser()
parser.add_argument("--method_idx", type=int, required=True,
                    help="0=PPD, 1=WPD-base, 2=WPD-drop_ll")
args = parser.parse_args()

SEED   = 42
N_IMG  = 1
H, W   = 384, 1280
STEPS  = 50
CFG    = 2
J      = 4
RADII  = [8, 12, 20, 24]

METHODS = [
    dict(name="PPD",
         lora="models/ppd/flux1-dev_phipd_lora_302000.safetensors",
         noise_type="ppd",
         drop_ll=False),
    dict(name="WPD-base",
         lora="flux.safetensors",
         noise_type="wpd",
         drop_ll=False),
    dict(name="WPD-drop_ll",
         lora="models/train/FLUX.1-dev_lora_wpd_dropll/step-6000.safetensors",
         noise_type="wpd",
         drop_ll=True),
]

NEG = ("blurry, low quality, cartoon, cg render, unrealistic, "
       "dashboard, steering wheel, windshield frame")

PROMPTS = {
    "Day (clear)":  ("A photorealistic photo from a forward-facing vehicle camera, "
                     "clear sunny day, bright natural lighting, vivid colors, blue sky."),
    "Overcast":     ("A photorealistic photo from a forward-facing vehicle camera, "
                     "overcast sky, soft diffuse gray lighting, no shadows, muted colors."),
    "Rain":         ("A photorealistic photo from a forward-facing vehicle camera, "
                     "heavy rain, wet glistening road, rain streaks, dark sky, headlights on."),
    "Night":        ("A photorealistic photo from a forward-facing vehicle camera, "
                     "nighttime, dark scene, streetlights, headlight beams on wet road."),
    "Fog":          ("A photorealistic photo from a forward-facing vehicle camera, "
                     "dense fog, low visibility, misty atmosphere, washed-out distant objects."),
}

PROMPT_LABELS = list(PROMPTS.keys())
method = METHODS[args.method_idx]
mname  = method["name"]
print(f"Method: {mname}  (idx={args.method_idx})")

os.makedirs("figures/weather", exist_ok=True)

# ── Sample images ─────────────────────────────────────────────────────────────
all_imgs = sorted(glob.glob(
    "/mrtstorage/datasets_tmp/vkitti/vkitti_1.3.1_rgb/*/clone/*.png"))
random.seed(SEED)
_two = random.sample(all_imgs, 2)
samples = [_two[1]]   # use only the second sample (img1)
print(f"Input image: {samples[0]}")

inputs_pil = [Image.open(p).convert("RGB").resize((W, H), Image.LANCZOS) for p in samples]

# Save input image (idempotent across workers)
inputs_pil[0].save(f"figures/weather/input_img0.png")

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

total = len(RADII) * N_IMG * len(PROMPTS)
done  = 0

for r in RADII:
    print(f"\n  radius={r}")
    for img_idx, img_pil in enumerate(inputs_pil):
        with torch.no_grad():
            img_t = pipe.preprocess_image(img_pil).to(pipe.device, dtype=pipe.torch_dtype)
            z_sim = pipe.vae_encoder(img_t, tiled=False)

        torch.manual_seed(SEED + img_idx)
        input_noise = torch.randn(z_sim.shape, dtype=torch.float32, device="cpu")
        if method["noise_type"] == "ppd":
            eps_struct = generate_structured_noise_batch_vectorized(
                z_sim.float().cpu(), cutoff_radius=r,
                input_noise=input_noise,
            ).to(device=pipe.device, dtype=pipe.torch_dtype).contiguous()
        else:
            eps_struct = generate_wavelet_structured_noise_batch_vectorized(
                z_sim.float().cpu(), radius_map=r,
                input_noise=input_noise,
                drop_ll=method["drop_ll"], J=J,
            ).to(device=pipe.device, dtype=pipe.torch_dtype).contiguous()

        for p_idx, (plabel, prompt) in enumerate(PROMPTS.items()):
            slug = plabel.lower().replace(" ", "_").replace("(", "").replace(")", "")
            out_path = f"figures/weather/{mname}_r{r}_img{img_idx}_{slug}.png"

            with torch.no_grad():
                out_pil = pipe(
                    prompt=prompt, negative_prompt=NEG,
                    height=H, width=W,
                    cfg_scale=CFG, num_inference_steps=STEPS,
                    noise=eps_struct,
                )
            out_pil.save(out_path)
            done += 1
            print(f"  [{done}/{total}] saved: {out_path}")

del pipe
torch.cuda.empty_cache()
gc.collect()
print(f"\nDone: {mname}")
