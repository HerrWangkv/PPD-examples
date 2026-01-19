# from structured_noise import generate_structured_noise_batch_vectorized
import argparse
import torch
from PIL import Image
from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig
from diffsynth import download_models
from pytorch_wavelets import DTCWTForward, DTCWTInverse
import torch.fft as fft

import torch
import torch.nn.functional as F
import torch.fft as fft
from pytorch_wavelets import DTCWTForward, DTCWTInverse

def dtcwt_structured_noise(source_img, noise_std=1.0, levels=4, mag_threshold=0.6):
    device = source_img.device
    dtype = source_img.dtype
    
    # 1. 初始化算子
    xfm = DTCWTForward(J=levels, biort='near_sym_b', qshift='qshift_b').to(device=device, dtype=dtype)
    ifm = DTCWTInverse(biort='near_sym_b', qshift='qshift_b').to(device=device, dtype=dtype)
    
    # 1. DTCWT Decomposition
    yl_src, yh_src = xfm(source_img)
    # Generate Gaussian noise (N(0,1))
    yl_nz, yh_nz = xfm(torch.randn_like(source_img))

    # --- 2. LL Layer (yl) Logic (NeuralRemaster FFT Style) ---
    h, w = yl_src.shape[-2:]
    pad_h, pad_w = h // 2, w // 2
    
    # Padding and Float32 for FFT
    yl_src_pad = F.pad(yl_src, (pad_w//2, pad_w//2, pad_h//2, pad_h//2), mode='reflect').float()
    yl_nz_pad = F.pad(yl_nz, (pad_w//2, pad_w//2, pad_h//2, pad_h//2), mode='reflect').float()
    
    # FFT and Shift
    f_src = fft.fftshift(fft.fft2(yl_src_pad, dim=(-2, -1)), dim=(-2, -1))
    f_nz = fft.fftshift(fft.fft2(yl_nz_pad, dim=(-2, -1)), dim=(-2, -1))
    
    # NeuralRemaster Core: Mag_nz * noise_std + Phase_src
    phase_src_yl = torch.angle(f_src)
    mag_nz_yl = torch.abs(f_nz) * noise_std # Scale by target std
    
    f_combined = torch.polar(mag_nz_yl, phase_src_yl)
    f_unshifted = fft.ifftshift(f_combined, dim=(-2, -1))
    yl_final_pad = torch.real(fft.ifft2(f_unshifted, dim=(-2, -1)))
    
    # Crop to original size
    yl_final = yl_final_pad[:, :, pad_h//2:pad_h//2 + h, pad_w//2:pad_w//2 + w].to(dtype)

    # --- 4. 高频层 (yh) 处理 (BFloat16 正常运行) ---
    yh_final = []
    for i in range(levels):
        real_src, imag_src = yh_src[i][..., 0], yh_src[i][..., 1]
        real_nz, imag_nz = yh_nz[i][..., 0], yh_nz[i][..., 1]

        mag_src = torch.sqrt(real_src**2 + imag_src**2)
        phase_src = torch.atan2(imag_src, real_src)
        
        mag_nz = torch.sqrt(real_nz**2 + imag_nz**2)
        phase_nz = torch.atan2(imag_nz, real_nz)

        mask = (mag_src > mag_threshold).to(dtype)
        mixed_phase = phase_src * mask + phase_nz * (1 - mask)

        new_real = mag_nz * torch.cos(mixed_phase)
        new_imag = mag_nz * torch.sin(mixed_phase)
        yh_final.append(torch.stack([new_real, new_imag], dim=-1))
    
    # 5. 重构
    return ifm((yl_final, yh_final))

def parse_args():
    parser = argparse.ArgumentParser(description="Generate videos with trained model")
    parser.add_argument(
        "--lora_checkpoint_path",
        type=str,
        required=False,
        default="models/ppd/flux1-dev_lora_color_step=266000_biased.safetensors",
        help="Path to lora checkpoint file"
    )
    parser.add_argument(
        "--input_image",
        type=str,
        default="ppd/test1.jpg",
        help="Input image filename"
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="output.png",
        help="Output image filename"
    )
    parser.add_argument(
        "--J",
        type=int,
        default=1,
        help="Number of wavelet decomposition levels"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="magnitude threshold for high-frequency phase mixing"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A high quality scene from a movie captured by a professional camera. A woman stands in a rugged cave-like environment with stone walls and patches of snow. She has an adventurous appearance, wearing a sleeveless top, shorts, gloves, and a utility belt, exuding a determined and alert expression as if ready to explore or face a challenge"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="ugly, low quality, CG, Render, unreal, game, cartoon, blur, low res",
        help="Negative prompt"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=704,
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_models(["FLUX.1-dev"])
    pipe = FluxImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="flux1-dev.safetensors"),
            ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors"),
            ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/"),
            ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors"),
        ],
    )

    embed_layers = None

    pipe.load_lora(pipe.dit, args.lora_checkpoint_path, alpha=1)

    image_in_pil = Image.open(args.input_image).convert("RGB")
    w,h = image_in_pil.size
    if args.height is not None and args.width is not None:
        use_original_size = False
        new_w, new_h = args.width, args.height
    else:
        use_original_size = True
        new_w, new_h = w//16*16, h//16*16
    image_in_pil = image_in_pil.resize((new_w, new_h), resample=Image.LANCZOS)
    prompt = args.prompt
    with torch.no_grad():
        image = pipe.preprocess_image(image_in_pil).to(device=pipe.device, dtype=pipe.torch_dtype)
        input_latents = pipe.vae_encoder(image, tiled=False)

        input_noise = torch.randn_like(input_latents)
        noise = dtcwt_structured_noise(input_latents, mag_threshold=args.threshold)
        noise = noise.contiguous()

        negative_prompt = args.negative_prompt

        image = pipe(
            prompt=prompt, negative_prompt=negative_prompt,
            height=new_h, width=new_w,
            cfg_scale=2, num_inference_steps=50, noise=noise
        )

        if use_original_size:
            image = image.resize((w,h))
        image.save(args.output_name)

