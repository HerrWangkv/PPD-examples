import argparse
import torch
import os
import glob
import sys
from PIL import Image
from diffusers import FluxPipeline
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "FlowEdit"))
from FlowEdit_utils import FlowEditFLUX


def init_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return rank, world, local_rank
    return 0, 1, 0


def parse_args():
    parser = argparse.ArgumentParser(description="Batch image sim2real using FlowEdit + FLUX")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--src_prompt", type=str, default=(
        "A synthetic rendered indoor scene, computer graphics, 3D rendering, "
        "artificial lighting, CG textures."
    ))
    parser.add_argument("--tar_prompt", type=str, default=(
        "A photorealistic indoor scene, natural and artificial lighting, "
        "real photograph, high resolution, realistic textures and materials."
    ))
    parser.add_argument("--T_steps", type=int, default=28)
    parser.add_argument("--n_min", type=int, default=0)
    parser.add_argument("--n_max", type=int, default=24)
    parser.add_argument("--n_avg", type=int, default=1)
    parser.add_argument("--src_guidance_scale", type=float, default=1.5)
    parser.add_argument("--tar_guidance_scale", type=float, default=5.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    rank, world, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}")

    os.makedirs(args.output_dir, exist_ok=True)

    images = sorted(glob.glob(os.path.join(args.input_dir, "*.jpg")) +
                    glob.glob(os.path.join(args.input_dir, "*.png")))
    images = images[rank::world]
    print(f"[Rank {rank}/{world}] Processing {len(images)} images")

    torch.manual_seed(args.seed)

    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16)
    pipe = pipe.to(device)
    scheduler = pipe.scheduler

    for i, img_path in enumerate(images):
        fname = os.path.basename(img_path)
        out_path = os.path.join(args.output_dir, fname)

        if os.path.exists(out_path):
            print(f"[{i+1}/{len(images)}] SKIP {fname}")
            continue

        print(f"[{i+1}/{len(images)}] {fname}", flush=True)
        try:
            image = Image.open(img_path).convert("RGB")
            # crop to dimensions divisible by 16
            image = image.crop((0, 0, image.width - image.width % 16, image.height - image.height % 16))

            image_src = pipe.image_processor.preprocess(image).to(device).half()
            with torch.autocast("cuda"), torch.inference_mode():
                x0_src_denorm = pipe.vae.encode(image_src).latent_dist.mode()
            x0_src = (x0_src_denorm - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor

            x0_tar = FlowEditFLUX(
                pipe, scheduler,
                x0_src,
                args.src_prompt, args.tar_prompt,
                "",  # negative prompt
                args.T_steps, args.n_avg,
                args.src_guidance_scale, args.tar_guidance_scale,
                args.n_min, args.n_max,
            )

            x0_tar_denorm = (x0_tar / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
            with torch.autocast("cuda"), torch.inference_mode():
                image_tar = pipe.vae.decode(x0_tar_denorm, return_dict=False)[0]
            result = pipe.image_processor.postprocess(image_tar)[0]
            result.save(out_path)
            print(f"  Saved {out_path}", flush=True)

        except Exception as e:
            print(f"  ERROR: {e}", flush=True)

    print(f"[Rank {rank}] Done.")


if __name__ == "__main__":
    main()
