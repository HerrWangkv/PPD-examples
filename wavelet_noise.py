import torch
import torch.nn.functional as F
from pytorch_wavelets import DTCWTForward, DTCWTInverse
from typing import Union, List
from einops import rearrange

def generate_wavelet_structured_noise_batch_vectorized(
        image_batch: torch.Tensor, 
        thresholds: Union[torch.Tensor, List, float] = None, 
        J: int = 4,
        noise_std: float = 1.0,
        pad_factor: float = 1.5,
        input_noise: torch.Tensor = None,
        biort: str = 'near_sym_b',
        qshift: str = 'qshift_b',
        soft_margin: float = 0.05,
    ):
    """Generate structured noise using DTCWT with vectorized batch processing.
    Args:
        image_batch (torch.Tensor): Input image batch of shape (B, C, H, W).
        thresholds (Union[torch.Tensor, List, float]): Quantile thresholds for high-frequency layers of shape (J, 6), (J,) or a single float value.
        J (int): Number of DTCWT levels.
        noise_std (float): Standard deviation for noise scaling.
        pad_factor (float): Padding factor for LL layer processing.
        input_noise (torch.Tensor): Optional input noise tensor of same shape as image_batch.
        biort (str): Biorthogonal wavelet type.
        qshift (str): Q-shift wavelet type.
        soft_margin (float): Soft margin for masking.
    """
    device = image_batch.device
    dtype = image_batch.dtype
    B, C, H, W = image_batch.shape
    image_batch = image_batch.float()
    if thresholds is not None:
        if isinstance(thresholds, float):
            assert 0.0 <= thresholds <= 1.0, "Threshold float value must be in (0, 1)"
            thresholds = torch.full((J, 6), thresholds, device=device)
        else:
            if isinstance(thresholds, list):
                thresholds = torch.tensor(thresholds, device=device)
            if thresholds.ndim == 1:
                assert thresholds.shape[0] == J, "Threshold list must have length J"
                thresholds = thresholds.view(J, 1).repeat(1, 6)
            assert thresholds.shape == (J, 6), "Thresholds must have shape (J, 6)"
            assert (thresholds >= 0).all() and (thresholds <= 1).all(), "Threshold tensor values must be in (0, 1)"     
            thresholds = thresholds.to(device=device)

    # Initialize DTCWT operators
    xfm = DTCWTForward(J=J, biort=biort, qshift=qshift).to(device=device)
    ifm = DTCWTInverse(biort=biort, qshift=qshift).to(device=device, dtype=dtype)

    # DTCWT Decomposition
    yl_src, yh_src = xfm(image_batch)
    yl_nz, yh_nz = xfm(torch.randn_like(image_batch) if input_noise is None else input_noise)

    # LL Layer Processing
    height, width = yl_src.shape[-2:]
    pad_h = int(height * (pad_factor - 1))
    pad_h = pad_h // 2 * 2 # make it even
    pad_w = int(width * (pad_factor - 1))
    pad_w = pad_w // 2 * 2 # make it even

    yl_src_pad = F.pad(yl_src, (pad_w//2, pad_w//2, pad_h//2, pad_h//2), mode='reflect')
    yl_nz_pad = F.pad(yl_nz, (pad_w//2, pad_w//2, pad_h//2, pad_h//2), mode='reflect')

    # Apply 2D FFT to padded LL layers
    fft_src = torch.fft.fft2(yl_src_pad, dim=(-2, -1))
    fft_shifted_src = torch.fft.fftshift(fft_src, dim=(-2, -1))

    fft_nz = torch.fft.fft2(yl_nz_pad, dim=(-2, -1))
    fft_shifted_nz = torch.fft.fftshift(fft_nz, dim=(-2, -1))

    phase_yl_src = torch.angle(fft_shifted_src)
    mag_yl_nz = torch.abs(fft_shifted_nz) * noise_std

    fft_combined = mag_yl_nz * torch.exp(1j * phase_yl_src)
    fft_unshifted = torch.fft.ifftshift(fft_combined, dim=(-2, -1))
    yl_final_pad = torch.real(torch.fft.ifft2(fft_unshifted, dim=(-2, -1)))
    yl_final = yl_final_pad[:, :, pad_h//2:pad_h//2 + height, pad_w//2:pad_w//2 + width].to(dtype)

    # High-Frequency Layers Processing
    yh_final = []
    for i in range(J):
        # Convert to complex for easier math
        src_c = torch.view_as_complex(yh_src[i])
        nz_c = torch.view_as_complex(yh_nz[i])

        mag_src, phase_src = src_c.abs(), src_c.angle()
        mag_nz, phase_nz = nz_c.abs(), nz_c.angle()

        if thresholds is None: # training
            subband_max = mag_src.amax(dim=(-1, -2), keepdim=True) 
            r = torch.rand((B, C, 6, 1, 1), device=device) * 0.95
            current_threshold = subband_max * r
        else: # inference
            mag_flat = rearrange(mag_src, 'b c d h w -> b c d (h w)')
            thresh_d = [torch.quantile(mag_flat[:, :, d, :], thresholds[i, d], dim=-1, keepdim=True) for d in range(6)]
            current_threshold = torch.stack(thresh_d, dim=-1).view(B, C, 6, 1, 1)
            
        if soft_margin > 0:
            # Soft mask
            eps = 1e-6
            lower = current_threshold * (1.0 - soft_margin)
            upper = current_threshold * (1.0 + soft_margin)
            mask = torch.clamp((mag_src - lower) / (upper - lower + eps), 0, 1)
        else:
            mask = (mag_src >= current_threshold).to(dtype)

        # Mix phases based on mask
        mixed_phase = phase_src * mask + phase_nz * (1 - mask)

        # Reconstruct complex subband
        final_c = torch.polar(mag_nz, mixed_phase)
        yh_final.append(torch.view_as_real(final_c).to(dtype))

    # Reconstruct the final image
    structed_noise = ifm((yl_final, yh_final))
    return structed_noise

