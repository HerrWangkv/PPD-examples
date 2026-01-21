import torch
import torch.nn.functional as F
from pytorch_wavelets import DTCWTForward, DTCWTInverse
from typing import Union, List, Optional
from einops import rearrange
import os

def generate_wavelet_structured_noise_batch_vectorized(
        image_batch: torch.Tensor, 
        binary_mask: Optional[torch.Tensor] = None,    # NEW: Binary Mask (1=Keep Sim, 0=Noise)
        thresholds: Union[torch.Tensor, List, float] = None, 
        J: int = 4,
        noise_std: float = 1.0,
        pad_factor: float = 1.5,
        input_noise: torch.Tensor = None,
        biort: str = 'near_sym_b',
        qshift: str = 'qshift_b',
    ):
    """
    Generate structured noise using DTCWT. 
    Can use a Binary Mask to explicitly define where to keep Sim structure.
    
    Args:
        image_batch (torch.Tensor): Input image batch (B, C, h, w).
        binary_mask (torch.Tensor): Optional mask (B, 1, H, W) or (B, H, W). 
                                    1.0 = Keep Sim Phase (Geometry).
                                    0.0 = Use Noise Phase (Random Texture).
        thresholds: Quantile thresholds (used only if binary_mask is None).
        J (int): Number of DTCWT levels.
        noise_std (float): Noise scale.
        pad_factor (float): Padding for LL FFT.
    """
    device = image_batch.device
    dtype = image_batch.dtype
    B, C, H, W = image_batch.shape
    image_batch = image_batch.float()

    # Pre-process Mask
    if binary_mask is not None:
        if binary_mask.ndim == 3:
            binary_mask = binary_mask.unsqueeze(1) # Ensure (B, 1, H, W)
        binary_mask = binary_mask.float().to(device)

    # Handle threshold inputs (Only needed if binary_mask is NOT provided)
    if binary_mask is None and thresholds is not None:
        if isinstance(thresholds, float):
            thresholds = torch.full((J, 6), thresholds, device=device)
        else:
            if isinstance(thresholds, list):
                thresholds = torch.tensor(thresholds, device=device)
            if thresholds.ndim == 1:
                thresholds = thresholds.view(J, 1).repeat(1, 6)
            thresholds = thresholds.to(device=device)

    # Initialize DTCWT operators
    xfm = DTCWTForward(J=J, biort=biort, qshift=qshift).to(device=device)
    ifm = DTCWTInverse(biort=biort, qshift=qshift).to(device=device, dtype=dtype)

    # 1. Decomposition
    yl_src, yh_src = xfm(image_batch)
    yl_nz, yh_nz = xfm(torch.randn_like(image_batch) if input_noise is None else input_noise)

    # 2. LL Layer Processing (Low Frequency Preservation)
    # ... (Standard FFT mixing logic for Low Frequency) ...
    height, width = yl_src.shape[-2:]
    pad_h = int(height * (pad_factor - 1)) // 2 * 2
    pad_w = int(width * (pad_factor - 1)) // 2 * 2

    yl_src_pad = F.pad(yl_src, (pad_w//2, pad_w//2, pad_h//2, pad_h//2), mode='reflect')
    yl_nz_pad = F.pad(yl_nz, (pad_w//2, pad_w//2, pad_h//2, pad_h//2), mode='reflect')

    fft_src = torch.fft.fft2(yl_src_pad, dim=(-2, -1))
    fft_shifted_src = torch.fft.fftshift(fft_src, dim=(-2, -1))
    fft_nz = torch.fft.fft2(yl_nz_pad, dim=(-2, -1))
    fft_shifted_nz = torch.fft.fftshift(fft_nz, dim=(-2, -1))

    # LL Mixing: Keep Sim Phase, Use Noise Magnitude
    phase_yl_src = torch.angle(fft_shifted_src)
    mag_yl_nz = torch.abs(fft_shifted_nz) * noise_std

    fft_combined = mag_yl_nz * torch.exp(1j * phase_yl_src)
    fft_unshifted = torch.fft.ifftshift(fft_combined, dim=(-2, -1))
    yl_final_pad = torch.real(torch.fft.ifft2(fft_unshifted, dim=(-2, -1)))
    yl_final = yl_final_pad[:, :, pad_h//2:pad_h//2 + height, pad_w//2:pad_w//2 + width].to(dtype)

    # 3. High-Frequency Layers Processing (J Levels)
    yh_final = []
    for i in range(J):
        src_c = torch.view_as_complex(yh_src[i]) # (B, C, 6, H_i, W_i)
        nz_c = torch.view_as_complex(yh_nz[i])
        
        mag_nz, phase_nz = nz_c.abs(), nz_c.angle()
        mag_src, phase_src = src_c.abs(), src_c.angle()

        # --- Mask Logic ---
        if binary_mask is not None:
            # STRATEGY A: Explicit Binary Masking
            if i == 0:
                mask = torch.zeros((B, C, 6, 1, 1), device=device, dtype=dtype)
            else:
                curr_h, curr_w = mag_src.shape[-2:]
                
                # Downsample mask to current level resolution using interpolation
                # This keeps the binary nature (hard edges) of the mask
                mask_resized = F.max_pool2d(
                    binary_mask, 
                    kernel_size=(binary_mask.shape[-2] // curr_h, binary_mask.shape[-1] // curr_w)
                ) # (B, 1, H_i, W_i)
                # import torchvision.utils as vutils

                # # Normalize mask_resized to [0, 1] for saving
                # mask_to_save = mask_resized.clone()
                # mask_to_save = (mask_to_save - mask_to_save.min()) / (mask_to_save.max() - mask_to_save.min() + 1e-8)

                # # Save the first mask in the batch as mask.png
                # save_dir = "./"
                # os.makedirs(save_dir, exist_ok=True)
                # vutils.save_image(mask_to_save[0], os.path.join(save_dir, f"mask_{i}.png"))

                # Expand for broadcasting: (B, 1, H_i, W_i) -> (B, 1, 1, H_i, W_i)
                # It will broadcast to C channels and 6 orientations automatically
                mask = mask_resized.unsqueeze(2).to(dtype)
                
        else:
            # STRATEGY B: Implicit Magnitude Thresholding (Fallback)
            if thresholds is None: 
                # Random threshold during training
                subband_max = mag_src.amax(dim=(-1, -2), keepdim=True) 
                r = torch.rand((B, C, 6, 1, 1), device=device) * 0.95
                current_threshold = subband_max * r
            else: 
                # Fixed threshold during inference
                mag_flat = rearrange(mag_src, 'b c d h w -> b c d (h w)')
                thresh_d = [torch.quantile(mag_flat[:, :, d, :], thresholds[i, d], dim=-1, keepdim=True) for d in range(6)]
                current_threshold = torch.stack(thresh_d, dim=-1).view(B, C, 6, 1, 1)

            mask = (mag_src >= current_threshold).to(dtype)

        # --- Phase Mixing ---
        # 1.0 (Mask) -> Keep Sim Phase
        # 0.0 (Mask) -> Use Noise Phase
        mixed_phase = phase_src * mask + phase_nz * (1 - mask)

        # Reconstruct: Use Random Magnitude (Realism) + Mixed Phase (Geometry)
        final_c = torch.polar(mag_nz, mixed_phase)
        yh_final.append(torch.view_as_real(final_c).to(dtype))

    # 4. Reconstruction
    structed_noise = ifm((yl_final, yh_final))
    return structed_noise