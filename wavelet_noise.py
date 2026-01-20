import torch
import torch.nn.functional as F
from pytorch_wavelets import DTCWTForward, DTCWTInverse
from typing import Union

def generate_wavelet_structured_noise_batch_vectorized(
        image_batch: torch.Tensor, 
        thresholds: Union[torch.Tensor, float] = None, 
        J: int = 4,
        noise_std: float = 1.0,
        pad_factor: float = 1.5,
        input_noise: torch.Tensor = None,
        biort: str = 'near_sym_b',
        qshift: str = 'qshift_b'
    ):
    """Generate structured noise using DTCWT with vectorized batch processing.
    Args:
        image_batch (torch.Tensor): Input image batch of shape (B, C, H, W).
        thresholds (Union[torch.Tensor, float]): Thresholds for high-frequency layers of shape (J, 6) or a single float value.
        J (int): Number of DTCWT levels.
        noise_std (float): Standard deviation for noise scaling.
        pad_factor (float): Padding factor for LL layer processing.
        input_noise (torch.Tensor): Optional input noise tensor of same shape as image_batch.
        biort (str): Biorthogonal wavelet type.
        qshift (str): Q-shift wavelet type.
    """
    device = image_batch.device
    dtype = image_batch.dtype
    B, C, H, W = image_batch.shape
    image_batch = image_batch.float()
    if thresholds is not None:
        if isinstance(thresholds, float):
            thresholds = torch.full((J, 6), thresholds, device=device)
        else:
            assert thresholds.shape == (J, 6), "Thresholds must have shape (J, 6)"
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

        real_src, imag_src = yh_src[i][..., 0], yh_src[i][..., 1]
        real_nz, imag_nz = yh_nz[i][..., 0], yh_nz[i][..., 1]

        mag_src = torch.sqrt(real_src**2 + imag_src**2)
        phase_src = torch.atan2(imag_src, real_src)

        mag_nz = torch.sqrt(real_nz**2 + imag_nz**2)
        phase_nz = torch.atan2(imag_nz, real_nz)

        if thresholds is None:
            subband_max = mag_src.amax(dim=(-1, -2, -4), keepdim=True) 
            r = torch.rand((B, 1, 6, 1, 1), device=device) * 0.95
            current_threshold = subband_max * r
        else:
            current_threshold = thresholds[i].view(1, 1, 6, 1, 1)
            
        # Create mask based on thresholds
        mask = (mag_src >= current_threshold).to(dtype)

        # Mix phases based on mask
        mixed_phase = phase_src * mask + phase_nz * (1 - mask)

        real_final = mag_nz * torch.cos(mixed_phase)
        imag_final = mag_nz * torch.sin(mixed_phase)

        yh_final.append(torch.stack((real_final, imag_final), dim=-1).to(dtype))

    # Reconstruct the final image
    structed_noise = ifm((yl_final, yh_final))
    return structed_noise

