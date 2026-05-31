import argparse
import torch
import os
import glob
import gc
from PIL import Image

from diffsynth import download_models
from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig as FluxModelConfig
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
    parser = argparse.ArgumentParser(description="Batch image sim2real using Flux PPD (FFT structured noise)")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--flux_lora", type=str, default="models/ppd/flux1-dev_phipd_lora_302000.safetensors")
    parser.add_argument("--flux_cutoff_radius", type=int, default=20)
    parser.add_argument("--prompt", type=str, default=(
        "A photorealistic photograph taken from a forward-facing vehicle-mounted camera. "
        "Natural outdoor lighting, authentic surface textures, real-world colors."
    ))
    parser.add_argument("--negative_prompt", type=str, default=(
        "ugly, low quality, CG, render, unreal, game, cartoon, blur, low res, "
        "dashboard, steering wheel, windshield frame, car interior, lens artifacts"
    ))
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    is_dist, rank, world, local_rank = init_distributed()
    device = f"cuda:{local_rank}"

    os.makedirs(args.output_dir, exist_ok=True)

    existing = set(os.listdir(args.output_dir))
    print(f"[Rank {rank}/{world}] output_dir={args.output_dir}, existing={len(existing)} files")

    all_images = sorted(glob.glob(os.path.join(args.input_dir, "*.jpg")) +
                        glob.glob(os.path.join(args.input_dir, "*.png")))

    todo = [p for p in all_images if os.path.basename(p) not in existing]
    images = todo[rank::world]
    print(f"[Rank {rank}/{world}] {len(todo)} remaining, assigned {len(images)} images")

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
        print(f"Warning: Flux LoRA not found at {args.flux_lora}")

    torch.manual_seed(args.seed)

    for i, img_path in enumerate(images):
        fname = os.path.basename(img_path)
        out_path = os.path.join(args.output_dir, fname)

        print(f"[{i+1}/{len(images)}] {fname}")
        try:
            image = Image.open(img_path).convert("RGB").resize((args.width, args.height), Image.LANCZOS)

            with torch.no_grad():
                image_tensor = pipe.preprocess_image(image).to(device=device, dtype=pipe.torch_dtype)
                input_latents = pipe.vae_encoder(image_tensor, tiled=False)

                input_noise = torch.randn_like(input_latents)
                noise = generate_structured_noise_batch_vectorized(
                    input_latents,
                    cutoff_radius=args.flux_cutoff_radius,
                    input_noise=input_noise,
                ).contiguous().to(device)

                result = pipe(
                    prompt=args.prompt,
                    negative_prompt=args.negative_prompt,
                    height=args.height,
                    width=args.width,
                    cfg_scale=args.cfg_scale,
                    num_inference_steps=args.num_inference_steps,
                    noise=noise,
                )

            result.save(out_path)
            print(f"  Saved to {out_path}")

        except Exception as e:
            print(f"  ERROR: {e}")

        gc.collect()
        torch.cuda.empty_cache()

    print(f"[Rank {rank}] Done.")


if __name__ == "__main__":
    main()
