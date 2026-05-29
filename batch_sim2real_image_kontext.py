import argparse
import torch
import os
import glob
import gc
from PIL import Image

from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig


def init_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return rank, world, local_rank
    return 0, 1, 0


def parse_args():
    parser = argparse.ArgumentParser(description="Batch image sim2real using FLUX.1-Kontext-dev")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--prompt", type=str, default=(
        "Convert this virtual rendering to a real photograph. Replace synthetic CG textures "
        "with real asphalt, concrete, and vegetation. Add natural outdoor lighting, film grain, "
        "and atmospheric haze. The result should look like a frame from a real dashcam."
    ))
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--embedded_guidance", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clone_only", action="store_true", help="Only process clone condition frames.")
    return parser.parse_args()


def flush():
    gc.collect()
    torch.cuda.empty_cache()


def main():
    args = parse_args()
    rank, world, local_rank = init_distributed()
    device = f"cuda:{local_rank}"

    os.makedirs(args.output_dir, exist_ok=True)

    existing = set(os.listdir(args.output_dir))
    print(f"[Rank {rank}/{world}] output_dir={args.output_dir}, existing={len(existing)} files")

    all_images = sorted(glob.glob(os.path.join(args.input_dir, "*.jpg")) +
                        glob.glob(os.path.join(args.input_dir, "*.png")))

    if args.clone_only:
        all_images = [p for p in all_images if "_clone_" in os.path.basename(p)]

    todo = [p for p in all_images if os.path.basename(p) not in existing]
    images = todo[rank::world]
    print(f"[Rank {rank}/{world}] {len(todo)} remaining, assigned {len(images)} images")

    pipe = FluxImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(model_id="black-forest-labs/FLUX.1-Kontext-dev", origin_file_pattern="flux1-kontext-dev.safetensors"),
            ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors"),
            ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/"),
            ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors"),
        ],
    )

    torch.manual_seed(args.seed)

    for i, img_path in enumerate(images):
        fname = os.path.basename(img_path)
        out_path = os.path.join(args.output_dir, fname)

        print(f"[{i+1}/{len(images)}] {fname}")
        try:
            image = Image.open(img_path).convert("RGB").resize((args.width, args.height), Image.LANCZOS)

            result = pipe(
                prompt=args.prompt,
                kontext_images=image,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                embedded_guidance=args.embedded_guidance,
                seed=args.seed,
            )

            result.save(out_path)
            print(f"  Saved to {out_path}")

        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"[Rank {rank}] Done.")


if __name__ == "__main__":
    main()
