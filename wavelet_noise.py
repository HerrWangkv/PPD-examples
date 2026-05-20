from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
from typing import Optional, Tuple, Union, List

import math
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pytorch_wavelets import DTCWTForward, DTCWTInverse
from structured_noise import generate_structured_noise_batch_vectorized
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def split_yh_to_C(yh_list, o_dim=2):
    """
    yh_list: list length J, each ~ (N,C,6,H,W,2)
    returns:
      C: list length J, each is list length 6, each (N,C,H,W,2)
    """
    C = []
    for yh_l in yh_list:
        # find orientation dim (usually 2 if shape is N,C,6,H,W,2)
        # but we keep o_dim as param (consistent with your DTCWTForward)
        C_l = [yh_l.select(o_dim, k) for k in range(6)]
        C.append(C_l)
    return C


class DTCWTDecomposer(nn.Module):
    def __init__(self, J=3, biort="antonini", qshift="qshift_d", mode="symmetric", o_dim=2, ri_dim=-1):
        super().__init__()
        self.J = J
        self.o_dim = o_dim
        self.xfm = DTCWTForward(J=J, biort=biort, qshift=qshift, mode=mode, o_dim=o_dim, ri_dim=ri_dim)

    def forward(self, x_nchw):
        """
        x_nchw: (N,C,H,W) real
        returns:
          LL: (N,C,H/2^J,W/2^J)  (approx)
          C:  list[J][6] of (N,C,H_l,W_l,2)
        """
        LL, yh = self.xfm(x_nchw)   # yh: list len J
        C = split_yh_to_C(yh, o_dim=self.o_dim)
        return LL, C

def resize_tensor(tensor: torch.Tensor, h: int, w: int, mode: str = "nearest", use_maxpool: bool = False) -> torch.Tensor:
    """
    Helper to resize masks or leak maps to match wavelet coefficient dimensions.
    
    Args:
        tensor: Input tensor (N, C, H, W)
        h: Target height
        w: Target width
        mode: Interpolation mode ('nearest', 'bilinear') used for upsampling or if use_maxpool is False.
        use_maxpool: If True, uses adaptive max pooling for downsampling. 
                     This allows small high-frequency features (like poles) to survive 
                     downsampling instead of being averaged out by the background.
    """
    # 1. If dimensions match, return immediately
    if tensor.shape[-2:] == (h, w):
        return tensor

    # 2. Check if this is a downsampling operation (Target size <= Original size)
    is_downsample = (h <= tensor.shape[-2]) and (w <= tensor.shape[-1])

    # 3. If MaxPool is enabled and we are downsampling -> Use Adaptive Max Pool
    # This ensures that if a pixel block contains a "structure" value (1.0), 
    # the downsampled block retains 1.0 instead of averaging with the "blur" value (0.0).
    if use_maxpool and is_downsample:
        return F.adaptive_max_pool2d(tensor.float(), output_size=(h, w))

    # 4. Otherwise (Upsampling OR MaxPool disabled) -> Use standard interpolation
    # Note: align_corners is typically None for 'nearest', but False for 'bilinear'
    align = False if mode != 'nearest' else None
    return F.interpolate(tensor.float(), size=(h, w), mode=mode, align_corners=align)

def complex_phase(re: torch.Tensor, im: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # atan2 is stable; eps not strictly needed, kept for symmetry with mag
    return torch.atan2(im, re)

def complex_mag(re: torch.Tensor, im: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.sqrt(re * re + im * im + eps)

def fuse_subband_generic(
    C_signal: torch.Tensor,
    C_noise: torch.Tensor,
    mask: Union[float, torch.Tensor],
    leak: Union[float, torch.Tensor] = 0.0,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Unified fusion function for DTCWPT coefficients.
    
    Controls Phase and Magnitude separately:
    - Phase Source: Controlled by 'mask' (1.0 = Signal Phase, 0.0 = Noise Phase).
    - Magnitude Source: Controlled by 'leak' (1.0 = Signal Mag, 0.0 = Noise Mag).
    
    Args:
        C_signal: Signal coefficients (N, C, H, W, 2).
        C_noise: Noise coefficients (N, C, H, W, 2).
        mask: Phase mixing ratio. Can be a float or Tensor broadcastable to coefficients.
        leak: Magnitude mixing ratio. Can be a float or Tensor broadcastable to coefficients.
        eps: Small value for numerical stability.
    """
    # 1. Decompose inputs
    re_s, im_s = C_signal[..., 0], C_signal[..., 1]
    re_n, im_n = C_noise[..., 0], C_noise[..., 1]

    # 2. Phase Mixing
    # Calculate phases
    phi_s = complex_phase(re_s, im_s)
    phi_n = complex_phase(re_n, im_n)
    
    m = mask

    # Mix Phase: Mask=1 keeps Signal Phase, Mask=0 takes Noise Phase
    # Note: Simple linear interpolation of phase is standard in PPD for binary masks.
    phi = m * phi_s + (1.0 - m) * phi_n

    # 3. Magnitude Mixing
    # Calculate magnitudes
    mag_s = complex_mag(re_s, im_s, eps=eps)
    mag_n = complex_mag(re_n, im_n, eps=eps)
    
    l = leak

    # Mix Magnitude: Leak=1 keeps Signal Mag, Leak=0 takes Noise Mag
    mag = l * mag_s + (1.0 - l) * mag_n

    # 4. Reconstruct Complex Coefficients
    re_out = mag * torch.cos(phi)
    im_out = mag * torch.sin(phi)
    
    return torch.stack([re_out, im_out], dim=-1)

def pack_C_to_yh_list(C, o_dim: int = 2):
    """
    C: list[J][6] each (N,C,H,W,2)
    return yh: list[J] each (N,C,6,H,W,2)
    """
    yh = []
    for C_l in C:
        # stack orientations
        yh_l = torch.stack(C_l, dim=o_dim)
        yh.append(yh_l)
    return yh

def ll_fusion_fftshift_global_phase(
    yl_src: torch.Tensor,     # (N,C,H,W) real, global LL phase source (mask-independent)
    yl_nz: torch.Tensor,      # (N,C,H,W) real, noise LL magnitude source
    pad_factor: float = 1.5,
    dc_suppress_radius: int = 0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    LL_mix = IFFT( |FFT(yl_nz)| * exp(j * angle(FFT(yl_src))) )
    with reflect padding + fftshift.

    dc_suppress_radius: if > 0, phase within this radius (in LL-pixel units) of DC is taken
    from noise instead of src. This removes the global illumination bias (mean brightness,
    large-scale colour cast) while preserving all other coarse structural phase.
    """
    if yl_src.shape != yl_nz.shape:
        raise ValueError(f"LL shapes must match. got src={yl_src.shape}, nz={yl_nz.shape}")

    N, C, H, W = yl_src.shape

    if pad_factor < 1.0:
        raise ValueError("pad_factor must be >= 1.0")
    if pad_factor == 1.0:
        pad_h = pad_w = 0
    else:
        pad_h = int(H * (pad_factor - 1)) // 2 * 2
        pad_w = int(W * (pad_factor - 1)) // 2 * 2

    def _pad(x):
        if pad_h == 0 and pad_w == 0:
            return x
        return F.pad(x, (pad_w//2, pad_w//2, pad_h//2, pad_h//2), mode="reflect")

    def _unpad(x):
        if pad_h == 0 and pad_w == 0:
            return x
        return x[..., pad_h//2:pad_h//2 + H, pad_w//2:pad_w//2 + W]

    yl_src_pad = _pad(yl_src)
    yl_nz_pad  = _pad(yl_nz)

    fft_src = torch.fft.fftshift(torch.fft.fft2(yl_src_pad, dim=(-2, -1)), dim=(-2, -1))
    fft_nz  = torch.fft.fftshift(torch.fft.fft2(yl_nz_pad,  dim=(-2, -1)), dim=(-2, -1))

    phi_src = torch.angle(fft_src)
    phi_nz  = torch.angle(fft_nz)
    mag = torch.abs(fft_nz)

    if dc_suppress_radius > 0:
        # Build a mask: 1 = keep src phase, 0 = use noise phase (DC region)
        pH, pW = yl_src_pad.shape[-2], yl_src_pad.shape[-1]
        cy, cx = pH // 2, pW // 2
        yy = torch.arange(pH, device=yl_src.device).float() - cy
        xx = torch.arange(pW, device=yl_src.device).float() - cx
        rr = torch.sqrt(yy[:, None] ** 2 + xx[None, :] ** 2)   # (pH, pW)
        dc_mask = (rr > dc_suppress_radius).float()             # 0 inside DC radius, 1 outside
        phi = dc_mask * phi_src + (1.0 - dc_mask) * phi_nz
    else:
        phi = phi_src

    fft_mix = mag * torch.exp(1j * phi)
    fft_mix = torch.fft.ifftshift(fft_mix, dim=(-2, -1))
    yl_mix_pad = torch.fft.ifft2(fft_mix, dim=(-2, -1)).real

    return _unpad(yl_mix_pad)

# --- helper: (N,C,H,W,2) <-> (N,2C,H,W) ---
def _cplx_to_chan(x):
    return torch.cat([x[...,0], x[...,1]], dim=1)

def _chan_to_cplx(x):
    n, c2, h, w = x.shape
    c = c2 // 2
    re, im = x[:, :c], x[:, c:]
    return torch.stack([re, im], dim=-1)

class ComplexBandPacketSplitter(nn.Module):
    """Split a *complex* subband into low/high halves using one-level DTCWPT on stacked (re,im) channels."""
    def __init__(self, biort="antonini", qshift="qshift_d", mode="symmetric", o_dim=2, ri_dim=-1):
        super().__init__()
        self.fwd = DTCWTForward(J=1, biort=biort, qshift=qshift, mode=mode, o_dim=o_dim, ri_dim=ri_dim)
        self.inv = DTCWTInverse(biort=biort, qshift=qshift, mode=mode, o_dim=o_dim, ri_dim=ri_dim)

    def split_once(self, band_nc_hw_2):
        # 1. Store original size
        orig_h, orig_w = band_nc_hw_2.shape[-3], band_nc_hw_2.shape[-2]
        x = _cplx_to_chan(band_nc_hw_2)
        yl, yh = self.fwd(x)
        yh0 = yh[0]
        low  = self.inv((yl, [torch.zeros_like(yh0)]))
        high = self.inv((torch.zeros_like(yl), [yh0]))
        
        # 2. Convert back to complex
        low_cplx = _chan_to_cplx(low)
        high_cplx = _chan_to_cplx(high)
        
        # 3. Crop if necessary (DTCWPT padding fix)
        if low_cplx.shape[-3] != orig_h or low_cplx.shape[-2] != orig_w:
            low_cplx = low_cplx[..., :orig_h, :orig_w, :]
            high_cplx = high_cplx[..., :orig_h, :orig_w, :]
            
        return low_cplx, high_cplx


class DTCWTFusePhaseMag_Recursive(nn.Module):
    def __init__(self, biort="antonini", qshift="qshift_d", mode="symmetric", o_dim=2, ri_dim=-1):
        super().__init__()
        self.o_dim = o_dim
        self.ifm = DTCWTInverse(biort=biort, qshift=qshift, mode=mode, o_dim=o_dim, ri_dim=ri_dim)
        self.splitter = ComplexBandPacketSplitter(biort=biort, qshift=qshift, mode=mode, o_dim=o_dim, ri_dim=ri_dim)
        self.early_exit_p = 0.05  # If mask is mostly one side, do direct fusion

    def forward(
        self,
        LL_img, C_img,
        LL_z, C_z,
        freq_map: torch.Tensor,
        pad_factor: float = 1.5,
        max_packet_level: int = 4,
        eps: float = 1e-8,
        drop_ll: bool = False,
    ):
        # freq_map: (N,1,H,W) upper cutoff (Nyquist-norm). Higher = more structure preserved.
        # drop_ll:  If True, replace LL entirely with noise.

        if drop_ll:
            LL_mix = LL_z
        else:
            LL_mix = ll_fusion_fftshift_global_phase(LL_img, LL_z, pad_factor=pad_factor)

        J = len(C_img)
        C_mix = []

        for l in range(J):
            level_idx = l + 1
            freq_high = 1.0 / (2.0 ** (level_idx - 1))
            freq_low  = 1.0 / (2.0 ** level_idx)

            ref = C_z[l][0]
            H_l, W_l = ref.shape[-3], ref.shape[-2]

            freq_l = resize_tensor(freq_map, H_l, W_l, mode='bilinear')

            # C_img[l] and C_z[l] are lists of 6 tensors, each (N, C, H_l, W_l, 2)
            Cimg6 = torch.stack(C_img[l], dim=2)   # (N, C, 6, H, W, 2)
            Cz6   = torch.stack(C_z[l],   dim=2)   # (N, C, 6, H, W, 2)

            N, C, O, H, W, two = Cimg6.shape
            assert O == 6 and two == 2

            # reshape to (N, C*6, H, W, 2)
            Cimg_merged = Cimg6.contiguous().view(N, C * 6, H, W, 2)
            Cz_merged   = Cz6.contiguous().view(N, C * 6, H, W, 2)

            # One recursive call instead of 6
            Cfused_merged = self._process_band_recursive(
                Cimg_merged, Cz_merged,
                freq_low, freq_high,
                freq_l,
                0, max_packet_level, eps
            )  # (N, C*6, H, W, 2)

            # reshape back to list-of-6 for pack_C_to_yh_list
            Cfused6 = Cfused_merged.view(N, C, 6, H, W, 2)
            C_mix_l = list(Cfused6.unbind(dim=2))  # 6 tensors of shape (N,C,H,W,2)

            C_mix.append(C_mix_l)

        yh_mix = pack_C_to_yh_list(C_mix, o_dim=self.o_dim)
        x_hat = self.ifm((LL_mix, yh_mix))
        return x_hat, LL_mix, C_mix

    def _process_band_recursive(self, c_img, c_nz, f_start, f_end, freq_map, level, max_level, eps):
        # freq_map:     upper cutoff — bands below this are preserved from image.
        # min_freq_map: lower cutoff — bands below this are replaced with noise (not preserved).
        print(f"Processing level {level} band [{f_start:.4f}, {f_end:.4f}]", end=".")

        # All pixels preserve structure in this band (below upper cutoff)
        if f_end <= freq_map.min().item() + 1e-9:
            print(" Preserving entire band from image.")
            return fuse_subband_generic(c_img, c_nz, mask=1.0, eps=eps)

        # All pixels use noise in this band (above upper cutoff)
        if f_start >= freq_map.max().item() - 1e-9:
            print(" Replacing entire band with noise.")
            return fuse_subband_generic(c_img, c_nz, mask=0.0, eps=eps)

        # Mixed band: per-pixel decision
        mid = (f_start + f_end) / 2
        decision_map = (mid <= freq_map).float()
        if level >= max_level:
            print(" Reaching maximum level. Using mixed decision.")
            return fuse_subband_generic(c_img, c_nz, mask=decision_map, eps=eps)
        print(" Splitting band and processing recursively.")
        lo_img, hi_img = self.splitter.split_once(c_img)
        lo_nz, hi_nz   = self.splitter.split_once(c_nz)

        H_sub, W_sub = lo_img.shape[-3], lo_img.shape[-2]
        sub_freq = resize_tensor(freq_map, H_sub, W_sub, mode='bilinear')
        mid_freq = (f_start + f_end) / 2.0
        out_lo = self._process_band_recursive(lo_img, lo_nz, f_start, mid_freq, sub_freq, level + 1, max_level, eps)
        out_hi = self._process_band_recursive(hi_img, hi_nz, mid_freq, f_end, sub_freq, level + 1, max_level, eps)
        return out_lo + out_hi

def _generate_wavelet_noise_impl(
    image_batch: torch.Tensor,
    freq_map: torch.Tensor,
    noise_std: float = 1.0,
    pad_factor: float = 1.5,
    input_noise: Optional[torch.Tensor] = None,
    biort: str = 'near_sym_b',
    qshift: str = 'qshift_b',
    drop_ll: bool = False,
    j_override: Optional[int] = None,
) -> torch.Tensor:
    """
    Args:
        freq_map:   (N,1,H,W) upper cutoff frequency, Nyquist-normalized. Bands below this
                    are preserved from image_batch (structure).
        drop_ll:    Drop the entire LL subband (replace with noise). Use with j_override to
                    control how small LL becomes before dropping (higher J = smaller LL = safer).
        j_override: Force a specific J decomposition depth (1–6). Overrides the auto J from freq_map.
    """
    if image_batch.ndim != 4:
        raise ValueError("Expected image_batch in NCHW format")

    device = image_batch.device
    dtype = image_batch.dtype
    image_batch = image_batch.float()
    freq_map = freq_map.to(device).float().clamp(0.0, 1.0)

    # Prepare Noise
    if input_noise is None:
        z = torch.randn_like(image_batch) * float(noise_std)
    else:
        z = input_noise.to(device)

    # Determine Decomposition Depth J
    if j_override is not None:
        J = max(1, min(j_override, 6))
    else:
        f_min = max(freq_map.min().item(), 1e-2)
        J = math.ceil(-math.log2(f_min))
        J = max(1, min(J, 6))

    H, W = image_batch.shape[-2:]
    ll_h, ll_w = H // (2 ** max(J - 1, 0)), W // (2 ** max(J - 1, 0))
    ll_f_end = 1.0 / (2 ** J)
    ll_mode = "dropped" if drop_ll else "preserved"
    print(f"LL band [0.0000, {ll_f_end:.4f}] (J={J}, {ll_h}x{ll_w}). {ll_mode}.")

    # Execute
    decomp = DTCWTDecomposer(J=J, biort=biort, qshift=qshift).to(device)
    fuser = DTCWTFusePhaseMag_Recursive(biort=biort, qshift=qshift).to(device)

    with torch.no_grad():
        LL_img, C_img = decomp(image_batch)
        LL_z, C_z = decomp(z)

        x_hat, _, _ = fuser(
            LL_img, C_img,
            LL_z, C_z,
            freq_map=freq_map,
            pad_factor=pad_factor,
            drop_ll=drop_ll,
        )
    clamp_mask = x_hat.abs() > 5
    structured_noise = torch.where(clamp_mask, z, x_hat)
    return structured_noise.to(dtype=dtype)

def generate_wavelet_structured_noise_batch_vectorized(
    image_batch: torch.Tensor,
    radius_map: Union[torch.Tensor, int],
    input_noise: Optional[torch.Tensor] = None,
    noise_std: float = 1.0,
    pad_factor: float = 1.5,
    biort: str = 'near_sym_b',
    qshift: str = 'qshift_b',
    drop_ll: bool = False,
    J: Optional[int] = None,
) -> torch.Tensor:
    """
    Generates DTCWT structured noise with a per-pixel frequency cutoff map.

    Args:
        image_batch: (N, C, H, W) source latents/images.
        radius_map:  (N,1,H,W) or int — upper cutoff radius in pixel-space.
                     Bands below this frequency are preserved from image_batch (structure).
        drop_ll:     Drop the entire LL subband (replace with noise).
                     Use with J to control LL size: higher J → smaller LL → less structure lost.
        J:           Override decomposition depth (1–6). If None, auto-determined from radius_map.
                     When using drop_ll, set J=3+ so LL contains mostly global illumination.
        input_noise: Optional pre-generated noise tensor (same shape as image_batch).
        noise_std:   Std of generated noise when input_noise is None.
    """
    N, _, H, W = image_batch.shape
    nyquist_radius = min(H, W) / 2.0

    if not isinstance(radius_map, torch.Tensor):
        freq_map = torch.full((N, 1, H, W), max(float(radius_map), 1.0) / nyquist_radius, device=image_batch.device)
    else:
        freq_map = radius_map.to(image_batch.device).float() / nyquist_radius

    return _generate_wavelet_noise_impl(
        image_batch=image_batch,
        freq_map=freq_map,
        noise_std=noise_std,
        pad_factor=pad_factor,
        input_noise=input_noise,
        biort=biort,
        qshift=qshift,
        drop_ll=drop_ll,
        j_override=J,
    )
