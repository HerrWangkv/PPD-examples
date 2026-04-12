import argparse
import torch
import os
from PIL import Image
from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig
from diffsynth import download_models
from dino_noise import load_dino, find_dino_preserving_noise


def parse_args():
    parser = argparse.ArgumentParser(description="DINO-PD inference with PPD checkpoint")
    parser.add_argument(
        "--lora_checkpoint_path",
        type=str,
        required=False,
        default="models/ppd/flux1-dev_lora_color_step=266000_biased.safetensors",
        help="Path to LoRA checkpoint (PPD or DINO-PD)"
    )
    parser.add_argument(
        "--input_image",
        type=str,
        default="data/synthia/RGB/0000820.png",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="output.png",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A photorealistic driving scene in a European city. Natural lighting, detailed asphalt road, urban buildings, trees, cars on the street. High resolution, cinematic, realistic textures, automotive photography."
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="ugly, low quality, CG, Render, unreal, game, cartoon, blur, low res",
    )
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width",  type=int, default=1280)
    parser.add_argument(
        "--dino_opt_steps",
        type=int,
        default=300,
        help="Adam steps to find the DINO-preserving noise endpoint"
    )
    parser.add_argument(
        "--dino_model_name",
        type=str,
        default="dinov2_vitl14_reg",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=2.0,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = "cuda"

    # 1. Load pipeline
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
    pipe.load_lora(pipe.dit, args.lora_checkpoint_path, alpha=1)

    # 2. Load DINOv2
    dino = load_dino(model_name=args.dino_model_name, device=device)

    # 3. Process input image
    image_in_pil = Image.open(args.input_image).convert("RGB")
    w, h = image_in_pil.size
    if args.height is not None and args.width is not None:
        new_w, new_h = args.width, args.height
        use_original_size = False
    else:
        new_w, new_h = w // 16 * 16, h // 16 * 16
        use_original_size = True
    image_in_pil = image_in_pil.resize((new_w, new_h), resample=Image.LANCZOS)

    # 4. Encode to latent
    with torch.no_grad():
        image_tensor = pipe.preprocess_image(image_in_pil).to(device=device, dtype=pipe.torch_dtype)
        z0 = pipe.vae_encoder(image_tensor, tiled=False)

    # 5. Find DINO-preserving noise endpoint (t=1 → z_t* = z1*)
    print(f"Finding DINO-preserving noise ({args.dino_opt_steps} steps)...")
    noise = find_dino_preserving_noise(
        z0=z0.float(),
        t=1.0,
        vae_decoder=pipe.vae_decoder,
        dino=dino,
        n_steps=args.dino_opt_steps,
    )
    noise = noise.to(dtype=pipe.torch_dtype, device=device).contiguous()

    # 6. Denoise
    image = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=new_h,
        width=new_w,
        cfg_scale=args.cfg_scale,
        num_inference_steps=args.num_inference_steps,
        noise=noise,
    )

    if use_original_size:
        image = image.resize((w, h))

    os.makedirs(os.path.dirname(args.output_name) or ".", exist_ok=True)
    image.save(args.output_name)
    print(f"Output saved to {args.output_name}")
