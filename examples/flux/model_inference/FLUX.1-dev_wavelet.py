# from structured_noise import generate_structured_noise_batch_vectorized
import argparse
import torch
import os
import cv2
import numpy as np
import torch.nn.functional as F
from PIL import Image
from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig
from diffsynth import download_models
from wavelet_noise import generate_wavelet_structured_noise_batch_vectorized

# SYNTHIA Dataset Target Classes (Traffic Participants)
# See trainIds in https://github.molgen.mpg.de/mohomran/cityscapes/blob/master/scripts/helpers/labels.py#L55
# 4: Fence, 5: Pole, 6: Traffic Light, 7: Traffic Sign, 11: Person, 12: Rider, 13: Car, 14: Truck, 15: Bus, 16: Train, 17: Motorcycle, 18: Bicycle
TARGET_CLASSES = [4, 5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18]

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
        default="data/synthia/RGB/test1.png", # Changed default to hint at dataset structure
        help="Input image filename. Script assumes standard SYNTHIA structure to find GT."
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
        default=4,
        help="Number of wavelet decomposition levels"
    )
    # Removed max_threshold and decay arguments
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

def get_gt_mask_path(image_path):
    """
    Infers the Ground Truth mask path from the input image path 
    based on SYNTHIA dataset structure.
    Replaces 'RGB' folder with 'GT/LABELS'.
    """
    # Check common dataset structure patterns
    if "RGB" in image_path:
        # Standard SYNTHIA structure: root/RGB/img.png -> root/GT/LABELS/img.png
        mask_path = image_path.replace("RGB", "GT/LABELS").replace(".png", "_labelTrainIds.png")
    else:
        # Fallback: Try to find a 'GT/LABELS' folder in the parent directory
        # This handles cases where user might point to a flat folder structure
        dir_name = os.path.dirname(image_path)
        base_name = os.path.basename(image_path)
        # Try moving up one level and looking for GT/LABELS
        parent_dir = os.path.dirname(dir_name)
        mask_path = os.path.join(parent_dir, "GT", "LABELS", base_name.replace(".png", "_labelTrainIds.png"))
    
    return mask_path

if __name__ == "__main__":
    args = parse_args()
    
    # 1. Infer and Check GT Mask Path
    mask_path = get_gt_mask_path(args.input_image)
    if not os.path.exists(mask_path):
        raise FileNotFoundError(
            f"\n[Error] Ground Truth Label not found!\n"
            f"Input Image: {args.input_image}\n"
            f"Expected Mask: {mask_path}\n"
            f"Please ensure your dataset follows the SYNTHIA structure (RGB/ vs GT/LABELS/) or adjust path logic."
        )
    print(f"Loading GT Mask from: {mask_path}")

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

    # 2. Load and Resize Image
    image_in_pil = Image.open(args.input_image).convert("RGB")
    w, h = image_in_pil.size
    if args.height is not None and args.width is not None:
        use_original_size = False
        new_w, new_h = args.width, args.height
    else:
        use_original_size = True
        new_w, new_h = w // 16 * 16, h // 16 * 16
    
    image_in_pil = image_in_pil.resize((new_w, new_h), resample=Image.LANCZOS)

    # 3. Load and Resize Mask (Crucial: Use NEAREST to preserve IDs)
    # Using cv2 to read unchanged (uint8/uint16) data
    mask_cv2 = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask_cv2 is None:
         raise ValueError(f"Failed to load mask image from {mask_path}. Is the file corrupted?")
         
    mask_pil = Image.fromarray(mask_cv2)
    mask_pil = mask_pil.resize((new_w, new_h), resample=Image.NEAREST)
    
    # Convert Mask to Tensor and Binary Mask
    mask_tensor = torch.from_numpy(np.array(mask_pil)).long().to(pipe.device) # (H, W)
    
    # Create Binary Mask: 1 for Target Classes, 0 for Background
    # isin requires a 1D tensor for test_elements
    target_classes_tensor = torch.tensor(TARGET_CLASSES, device=pipe.device)
    binary_mask = torch.isin(mask_tensor, target_classes_tensor).float() # (H, W)
    
    # Ensure shape is (B, 1, H, W) for the noise generator if needed, 
    # though the generator handles (H, W) or (1, H, W) usually.
    # The VAE output will be (B, C, H_lat, W_lat). The mask is (H_img, W_img).
    # The noise generator handles the downsampling of the mask internally.
    binary_mask = binary_mask.unsqueeze(0).unsqueeze(0) # (1, 1, H, W)

    prompt = args.prompt
    with torch.no_grad():
        image = pipe.preprocess_image(image_in_pil).to(device=pipe.device, dtype=pipe.torch_dtype)
        input_latents = pipe.vae_encoder(image, tiled=False)

        input_noise = torch.randn_like(input_latents)
        
        # 4. Generate Semantic Structured Noise
        # Passing binary_mask explicitly. No thresholds needed.
        noise = generate_wavelet_structured_noise_batch_vectorized(
            image_batch=input_latents, 
            binary_mask=binary_mask, 
            J=args.J
        )
        noise = noise.contiguous()

        negative_prompt = args.negative_prompt

        image = pipe(
            prompt=prompt, negative_prompt=negative_prompt,
            height=new_h, width=new_w,
            cfg_scale=2, num_inference_steps=50, noise=noise
        )

        if use_original_size:
            image = image.resize((w, h))
        os.makedirs(os.path.dirname(args.output_name), exist_ok=True)
        image.save(args.output_name)
        print(f"Output saved to {args.output_name}")