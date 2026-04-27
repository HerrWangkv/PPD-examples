"""
Validate how DINO features of z_t change along the diffusion trajectory.

Monkeypatches the scheduler step to intercept latents at every denoising step,
decodes them with the VAE, and measures DINO cosine distance to the original z0.
Produces a distance-vs-step plot and saves decoded frames at sampled steps.

Usage:
    PYTHONPATH=. python validate_dino_trajectory.py \
        --input_image models/ppd/test1.jpg \
        --prompt "$(cat models/ppd/test1.txt)" \
        --output_dir outputs/dino_validate
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image

from diffsynth import download_models
from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig
from dino_noise import find_dino_preserving_noise, latent_to_dino, load_dino


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", type=str, default="models/ppd/test1.jpg")
    parser.add_argument(
        "--noise_mode",
        type=str,
        choices=["dino", "ppd", "wavelet", "gaussian"],
        default="dino",
        help="Noise initialization. dino=Adam-optimized DINO-preserving (DinoPD). "
             "ppd=FFT structured (PPD). wavelet=DTCWT structured (WPD). gaussian=plain randn baseline.",
    )
    parser.add_argument(
        "--lora_checkpoint",
        type=str,
        default=None,
        help="LoRA path. Defaults: dino→latest DinoPD; ppd/wavelet→PPD color LoRA; gaussian→none.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A photorealistic scene. High resolution, realistic textures.",
    )
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=1.0,
                        help="v7-α/c default: cfg=1 matches DinoPD training. PPD/WPD published default is 2.0.")
    parser.add_argument("--embedded_guidance", type=float, default=3.5,
                        help="FLUX-dev distilled guidance scalar (fed as a token).")
    parser.add_argument("--dino_opt_steps", type=int, default=300,
                        help="Adam steps for DINO noise opt (only used when noise_mode=dino).")
    parser.add_argument("--dino_loss_layers", type=int, nargs="+", default=None,
                        help="0-indexed ViT block indices for the DINO opt loss. None = original "
                             "(final-layer only, equivalent to [23] for ViT-L). Try [4] for spatial; "
                             "[4,11,23] for multi-scale.")
    parser.add_argument("--dino_model_name", type=str, default="dinov2_vitl14_reg")
    parser.add_argument("--ppd_radius", type=int, default=35,
                        help="FFT cutoff radius for PPD noise (latent pixels).")
    parser.add_argument("--wpd_radius", type=int, default=30,
                        help="Wavelet cutoff radius for WPD noise (latent pixels).")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the input randn used by all noise modes.")
    parser.add_argument(
        "--save_frames_at",
        type=int,
        nargs="+",
        default=[0, 5, 10, 20, 30, 40, 49],
        help="Step indices at which to save a decoded frame",
    )
    parser.add_argument("--output_dir", type=str, default="outputs/dino_validate")
    return parser.parse_args()


PPD_DEFAULT_LORA = "models/ppd/flux1-dev_phipd_lora_302000.safetensors"
WPD_DEFAULT_LORA = "models/train/FLUX.1-dev_lora_wpd/step-20000.safetensors"
DINO_DEFAULT_LORA = "models/train/FLUX.1-dev_lora_dino_pd_v7c/step-600.safetensors"


def resolve_lora(noise_mode: str, lora_arg) -> str | None:
    if lora_arg is not None:
        return lora_arg
    if noise_mode == "dino":
        return DINO_DEFAULT_LORA
    if noise_mode == "ppd":
        return PPD_DEFAULT_LORA
    if noise_mode == "wavelet":
        return WPD_DEFAULT_LORA
    return None  # gaussian: no LoRA


@torch.no_grad()
def decode_to_pil(pipe, z: torch.Tensor) -> Image.Image:
    img = pipe.vae_decoder(z.to(pipe.torch_dtype), tiled=False).float()
    img = (img.clamp(-1, 1) + 1) / 2
    img = img.squeeze(0).permute(1, 2, 0)
    return Image.fromarray((img * 255).byte().cpu().numpy())


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda"

    # ── 1. Pipeline + LoRA ──────────────────────────────────────────────────
    print("Loading pipeline...")
    download_models(["FLUX.1-dev"])
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
    lora_path = resolve_lora(args.noise_mode, args.lora_checkpoint)
    if lora_path is not None:
        pipe.load_lora(pipe.dit, lora_path, alpha=1)
        print(f"LoRA loaded: {lora_path}")
    else:
        print("No LoRA loaded (gaussian baseline).")

    # ── 2. DINOv2 ────────────────────────────────────────────────────────────
    print("Loading DINOv2...")
    dino = load_dino(model_name=args.dino_model_name, device=device)

    # ── 3. Encode input → z0 ─────────────────────────────────────────────────
    img_pil = Image.open(args.input_image).convert("RGB")
    img_pil = img_pil.resize((args.width, args.height), resample=Image.LANCZOS)
    img_pil.save(os.path.join(args.output_dir, "input.png"))

    with torch.no_grad():
        img_t = pipe.preprocess_image(img_pil).to(device=device, dtype=pipe.torch_dtype)
        z0 = pipe.vae_encoder(img_t, tiled=False)
    z0_f32 = z0.float()

    with torch.no_grad():
        target_feats = latent_to_dino(pipe.vae_decoder, dino, z0_f32)
        target_feats = {k: v.detach() for k, v in target_feats.items()}

    # ── 4. Build noise according to noise_mode ───────────────────────────────
    g = torch.Generator(device=device).manual_seed(args.seed)
    base_randn = torch.randn(z0_f32.shape, generator=g, device=device, dtype=torch.float32)

    if args.noise_mode == "dino":
        print(f"Optimising DINO-preserving noise ({args.dino_opt_steps} steps)...")
        z1_star, _ = find_dino_preserving_noise(
            z0=z0_f32,
            t=1.0,
            vae_decoder=pipe.vae_decoder,
            dino=dino,
            n_steps=args.dino_opt_steps,
            loss_layers=args.dino_loss_layers,
        )
        noise = z1_star
    elif args.noise_mode == "ppd":
        from structured_noise import generate_structured_noise_batch_vectorized
        print(f"Building PPD structured noise (radius={args.ppd_radius})...")
        noise = generate_structured_noise_batch_vectorized(
            z0_f32, cutoff_radius=args.ppd_radius, input_noise=base_randn,
        )
    elif args.noise_mode == "wavelet":
        from wavelet_noise import generate_wavelet_structured_noise_batch_vectorized
        print(f"Building WPD wavelet noise (radius={args.wpd_radius})...")
        noise = generate_wavelet_structured_noise_batch_vectorized(
            z0_f32, radius_map=args.wpd_radius, input_noise=base_randn,
        )
    elif args.noise_mode == "gaussian":
        print("Using plain Gaussian noise baseline.")
        noise = base_randn
    else:
        raise ValueError(f"Unknown noise_mode: {args.noise_mode}")

    noise = noise.to(dtype=pipe.torch_dtype).contiguous()
    n_f = noise.float()
    print(
        f"[noise stats] mode={args.noise_mode}  mean={n_f.mean().item():+.4f}  "
        f"std={n_f.std().item():.4f}  min={n_f.min().item():+.3f}  max={n_f.max().item():+.3f}  "
        f"per-channel std={n_f.std(dim=(0,2,3)).mean().item():.4f}"
    )

    # ── 5. Hook scheduler.step to capture latents at every step ──────────────
    step_indices, dino_distances, sigma_vals = [], [], []
    save_steps = set(args.save_frames_at)
    original_step = pipe.scheduler.step

    # Make vae_decoder/dino accessible inside the hook
    _vae_decoder = pipe.vae_decoder
    _dino = dino

    step_counter = [0]  # mutable via closure

    def step_hook(model_output, timestep, sample, **kwargs):
        # Run the real scheduler step
        new_latents = original_step(model_output, timestep, sample, **kwargs)

        idx = step_counter[0]
        step_counter[0] += 1

        sigma = (timestep.cpu().item() / pipe.scheduler.num_train_timesteps)

        with torch.no_grad():
            cur_feats = latent_to_dino(_vae_decoder, _dino, new_latents.float())
            patch_d = (1.0 - F.cosine_similarity(cur_feats["patches"], target_feats["patches"], dim=-1).mean()).item()
            cls_d   = (1.0 - F.cosine_similarity(cur_feats["cls"],     target_feats["cls"],     dim=-1).mean()).item()
            # VGGT consumes x_norm_patchtokens only — headline metric is patch distance.
            total_d = patch_d

        step_indices.append(idx)
        dino_distances.append(total_d)
        sigma_vals.append(sigma)

        print(f"  step {idx:>3d}  sigma={sigma:.4f}  dino={total_d:.6f}  cls={cls_d:.6f}  patch={patch_d:.6f}")

        if idx in save_steps:
            frame = decode_to_pil(pipe, new_latents)
            frame.save(os.path.join(args.output_dir, f"step_{idx:03d}_sigma{sigma:.3f}.png"))

        return new_latents

    pipe.scheduler.step = step_hook

    # ── 6. Run normal inference ───────────────────────────────────────────────
    print("\nRunning denoising loop...")
    print(f"{'step':>5}  {'sigma':>7}  {'dino dist':>10}  {'cls dist':>10}  {'patch dist':>10}")
    print("-" * 57)

    output_img = pipe(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        cfg_scale=args.cfg_scale,
        embedded_guidance=args.embedded_guidance,
        num_inference_steps=args.num_inference_steps,
        noise=noise,
    )
    pipe.scheduler.step = original_step  # restore

    output_img.save(os.path.join(args.output_dir, "output_final.png"))

    # ── 7. Print summary ──────────────────────────────────────────────────────
    print(f"\nSummary:")
    print(f"  Step 0  DINO dist: {dino_distances[0]:.6f}")
    print(f"  Final   DINO dist: {dino_distances[-1]:.6f}")
    min_d = min(dino_distances)
    max_d = max(dino_distances)
    print(f"  Min: {min_d:.6f} at step {step_indices[dino_distances.index(min_d)]}")
    print(f"  Max: {max_d:.6f} at step {step_indices[dino_distances.index(max_d)]}")

    # ── 8. Plot ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(step_indices, dino_distances, marker=".", linewidth=1.5)
    axes[0].set_xlabel("Denoising step")
    axes[0].set_ylabel("DINO distance to z0")
    axes[0].set_title("DINO distance along trajectory (step index)")
    axes[0].grid(True, alpha=0.3)

    # sigma goes 1→0: invert x so left=noisy, right=clean
    axes[1].plot(sigma_vals, dino_distances, marker=".", linewidth=1.5, color="tab:orange")
    axes[1].invert_xaxis()
    axes[1].set_xlabel("sigma  ←  (1=pure noise,  0=clean image)")
    axes[1].set_ylabel("DINO distance to z0")
    axes[1].set_title("DINO distance vs noise level")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, "dino_trajectory.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to {plot_path}")
    print(f"Decoded frames saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
