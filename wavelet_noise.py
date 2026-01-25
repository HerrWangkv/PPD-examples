from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
from operator import inv
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


@dataclass(frozen=True)
class DyadicCutoff:
    k: int  # numerator
    n: int  # denominator power: t = k / 2**n

    @property
    def denom(self) -> int:
        return 1 << self.n

    @property
    def t(self) -> float:
        return self.k / float(self.denom)


def _is_power_of_two(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


def parse_dyadic_cutoff(
    t: Union[float, Tuple[int, int], DyadicCutoff],
    *,
    max_n: int = 6,
    tol: float = 0.0,
) -> DyadicCutoff:
    """
    Accept:
      - DyadicCutoff(k,n)
      - (k,n) tuple
      - float t, but must be exactly representable as k/2^n up to max_n (unless tol>0)
    """
    if isinstance(t, DyadicCutoff):
        k, n = t.k, t.n
    elif isinstance(t, tuple):
        k, n = int(t[0]), int(t[1])
    else:
        frac = Fraction(str(t)).limit_denominator(1 << max_n)
        k, d = frac.numerator, frac.denominator
        if tol > 0.0:
            best = None
            for n_try in range(max_n + 1):
                denom = 1 << n_try
                k_try = int(round(float(t) * denom))
                val = k_try / denom
                err = abs(val - float(t))
                if best is None or err < best[0]:
                    best = (err, k_try, n_try)
            assert best is not None
            err, k, n = best
            if err > tol:
                raise ValueError(f"t={t} not within tol={tol} of a dyadic fraction (best err={err}).")
            return DyadicCutoff(k=k, n=n)

        if not _is_power_of_two(d):
            raise ValueError(f"t={t} -> {k}/{d} not dyadic (denominator is not power of two).")
        n = int(d).bit_length() - 1  # since d is power of 2

    if n < 0:
        raise ValueError("n must be >= 0")
    denom = 1 << n
    if not (0 <= k <= denom):
        raise ValueError(f"Need 0 <= k <= 2^n, got k={k}, n={n}.")
    return DyadicCutoff(k=k, n=n)

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

def resize_tensor(tensor: torch.Tensor, h: int, w: int, mode: str = "nearest") -> torch.Tensor:
    """Helper to resize masks or leak maps to match wavelet coefficient dimensions."""
    if tensor.shape[-2:] == (h, w):
        return tensor
    return F.interpolate(tensor.float(), size=(h, w), mode=mode, align_corners=False)

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
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    LL_mix = IFFT( |FFT(yl_nz)| * exp(j * angle(FFT(yl_src))) )
    with reflect padding + fftshift (same style as your snippet).
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

    phi = torch.angle(fft_src)
    mag = torch.abs(fft_nz)

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
        depth_map: torch.Tensor,
        cutoff_norm: float,
        maximal_norm: float,
        gamma: float = 0.5,
        pad_factor: float = 1.5,
        max_packet_level: int = 4,
        eps: float = 1e-8,
    ):  
        disparity_map = 1 / (depth_map + eps)  # convert depth (m) to disparity (1/m)

        # Assuming depth_map is (N, 1, H, W)
        flattened = disparity_map.view(disparity_map.size(0), -1)
        d_min = flattened.min(dim=1, keepdim=True)[0].view(disparity_map.size(0), 1, 1, 1)
        d_max = flattened.max(dim=1, keepdim=True)[0].view(disparity_map.size(0), 1, 1, 1)
        
        div = d_max - d_min + eps
        
        # Calculate Normalized Map
        # 1.0 = Near (High Disparity), 0.0 = Far (Low Disparity)
        norm_disp = (disparity_map - d_min) / div

        # Invert to create Control Map: 
        # 0.0 = Near (Use Cutoff/Degrade), 1.0 = Far (Use Max/Preserve)
        control_map = 1.0 - norm_disp

        # Handle "Flat Depth" Case per image
        # If dynamic range is tiny, force control map to 0.0 (Cutoff/Degrade)
        is_flat = (d_max - d_min) < 1e-5
        # Broadcast mask to (N, 1, H, W) and fill zeros where flat
        control_map = torch.where(is_flat, torch.zeros_like(control_map), control_map)

        LL_mix = ll_fusion_fftshift_global_phase(LL_img, LL_z, pad_factor=pad_factor)
        J = len(C_img)
        C_mix = []

        for l in range(J):
            level_idx = l + 1
            freq_high = 1.0 / (2.0 ** (level_idx - 1))
            freq_low  = 1.0 / (2.0 ** level_idx)
            
            ref = C_z[l][0]
            H_l, W_l = ref.shape[-3], ref.shape[-2]
            
            d_l = resize_tensor(control_map, H_l, W_l, mode='bilinear')

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
                d_l,
                cutoff_norm, maximal_norm, gamma,
                0, max_packet_level, eps
            )  # (N, C*6, H, W, 2)

            # reshape back to list-of-6 for pack_C_to_yh_list
            Cfused6 = Cfused_merged.view(N, C, 6, H, W, 2)
            C_mix_l = list(Cfused6.unbind(dim=2))  # 6 tensors of shape (N,C,H,W,2)

            C_mix.append(C_mix_l)

        yh_mix = pack_C_to_yh_list(C_mix, o_dim=self.o_dim)
        x_hat = self.ifm((LL_mix, yh_mix))
        return x_hat, LL_mix, C_mix

    def _process_band_recursive(self, c_img, c_nz, f_start, f_end, depth_map, r_min, r_max, gamma, level, max_level, eps):
        # 1. Global Low Freq: Structure kept, Mag replaced
        if f_end <= r_min + 1e-9:
             return fuse_subband_generic(c_img, c_nz, mask=1.0, eps=eps)
        
        # 2. Pure Noise Band:
        if f_start >= r_max - 1e-9:
            return fuse_subband_generic(c_img, c_nz, mask=0.0, eps=eps)
            
        # 3. Transition Band (The Masked Region): Use PPD mixing
        mid = (f_start + f_end) / 2
        pixel_thresholds = r_min + (depth_map ** gamma) * (r_max - r_min)
        decision_map = (mid <= pixel_thresholds).float()
        p_per = decision_map.mean(dim=(-2,-1), keepdim=True)  # (N,1,1,1)
        uniform = (p_per < self.early_exit_p) | (p_per > 1.0 - self.early_exit_p)
        if uniform.all().item():
            mask = (p_per > 0.5).float()
            return fuse_subband_generic(c_img, c_nz, mask=mask, eps=eps)
        if level >= max_level:
            return fuse_subband_generic(c_img, c_nz, mask=decision_map, eps=eps)

        
        lo_img, hi_img = self.splitter.split_once(c_img)
        lo_nz, hi_nz   = self.splitter.split_once(c_nz)
        
        H_sub, W_sub = lo_img.shape[-3], lo_img.shape[-2]
        sub_depth = resize_tensor(depth_map, H_sub, W_sub, mode='bilinear')
        mid_freq = (f_start + f_end) / 2.0
        out_lo = self._process_band_recursive(lo_img, lo_nz, f_start, mid_freq, sub_depth, r_min, r_max, gamma, level + 1, max_level, eps)
        out_hi = self._process_band_recursive(hi_img, hi_nz, mid_freq, f_end, sub_depth, r_min, r_max, gamma, level + 1, max_level, eps)
        return out_lo + out_hi


def _sample_dyadic(min_val, max_val, max_n=5):
    """
    Returns a random float k / 2^n in [min_val, max_val].
    Using fixed denominator 2^max_n ensures all values are exactly representable
    in the wavelet packet tree up to depth max_n.
    """
    denom = 1 << max_n
    k_min = math.ceil(min_val * denom)
    k_max = math.floor(max_val * denom)
    
    # Ensure range is valid
    if k_min > k_max: 
        k_min = k_max
    k = random.randint(k_min, k_max)
    return k / float(denom)

def generate_wavelet_structured_noise_batch_vectorized(
    image_batch: torch.Tensor,
    cutoff_radius: int, 
    maximal_radius: Optional[int] = None,  
    depth_map: Optional[torch.Tensor] = None,
    noise_std: float = 1.0,
    pad_factor: float = 1.5,
    input_noise: torch.Tensor = None,
    biort: str = 'near_sym_b',
    qshift: str = 'qshift_b',
    gamma: float = 1.0
):
    """
    Generates Depth-Guided DTCWPT Structured Noise using pixel-defined radii.
    
    Args:
        image_batch: (N, C, H, W) source images.
        depth_map: (N, 1, H, W) depth map (meters).
        cutoff_radius: Int. Pixel radius for the 'Near' degradation cutoff. 
                       (e.g., 1 = heavy blur/noise, 30 = moderate).
        maximal_radius: Int. Pixel radius for 'Far' structure preservation.
                        If None, defaults to Nyquist (min_dim/2).
        gamma: Float. Controls the curvature of the frequency transition.
               0.5 is recommended for preserving structure (concave curve).
    """
    if image_batch.ndim != 4:
        raise ValueError(f"Expected image_batch in NCHW")
    
    device = image_batch.device
    dtype = image_batch.dtype
    N, C, H, W = image_batch.shape
    image_batch = image_batch.float()
    
    # 1. Parameter Normalization (Pixel -> Normalized Freq)
    # Nyquist frequency corresponds to radius = min_dim / 2
    min_dim = min(H, W)
    nyquist_radius = min_dim / 2.0
    
    # Normalize Cutoff (Near limit)
    # Clamp to ensure we don't divide by zero or go out of bounds
    r_pix = max(float(cutoff_radius), 1.0)
    r_min_norm = r_pix / nyquist_radius
    # Normalize Maximal (Far limit)
    if maximal_radius is None:
        # Default to Nyquist (keep everything for infinite distance)
        r_max_norm = 1.0
    else:
        r_max_norm = float(maximal_radius) / nyquist_radius
    if depth_map is None:
        depth_map = torch.ones((N, 1, H, W), device=device) # Will just use r_min_norm everywhere

    # Safety clamping
    r_min_norm = min(max(r_min_norm, 0.0), 1.0)
    r_max_norm = min(max(r_max_norm, r_min_norm), 1.0)

    # 2. Prepare Noise
    if input_noise is None:
        z = torch.randn_like(image_batch) * float(noise_std)
    else:
        z = input_noise.to(device)

    # 3. Determine Decomposition Depth J
    # We need J deep enough so that the lowest band is below r_min_norm.
    # Level J lowest freq is roughly 1 / 2^J.
    # We want 1 / 2^J <= r_min_norm  =>  2^J >= 1/r_min  =>  J >= log2(1/r_min)
    if r_min_norm < 1e-2:
        J = 6 # Arbitrary max depth for safety
    else:
        J = math.ceil(-math.log2(r_min_norm))
        J = max(1, min(J, 6)) # Clamp J between 1 and 6 (practical limits)
    
    # 4. Execution
    decomp = DTCWTDecomposer(J=J, biort=biort, qshift=qshift).to(device)
    fuser = DTCWTFusePhaseMag_Recursive(biort=biort, qshift=qshift).to(device)
    
    with torch.no_grad():
        LL_img, C_img = decomp(image_batch)
        LL_z, C_z = decomp(z)
        
        x_hat, _, _ = fuser(
            LL_img, C_img,
            LL_z, C_z,
            depth_map=depth_map,
            cutoff_norm=r_min_norm,
            maximal_norm=r_max_norm,
            gamma=gamma,
            pad_factor=pad_factor
        )
    clamp_mask = x_hat.abs() > 5
    structured_noise = torch.where(clamp_mask, z, x_hat)
    return structured_noise.to(dtype=dtype)

def make_zone_plate(size):
    """Generates a pattern where frequency increases with distance from center."""
    x = torch.linspace(-1, 1, size)
    y = torch.linspace(-1, 1, size)
    Y, X = torch.meshgrid(y, x, indexing='ij')
    R2 = X**2 + Y**2
    # Factor 140 covers the spectrum nicely for 512x512
    return torch.cos(140 * R2).unsqueeze(0).unsqueeze(0)

def make_brushed_metal(size):
    """Generates diagonal lines."""
    x = torch.linspace(-1, 1, size)
    y = torch.linspace(-1, 1, size)
    Y, X = torch.meshgrid(y, x, indexing='ij')
    base = torch.sin(100 * (X + Y)) 
    variation = 0.5 * torch.sin(20 * X) 
    return (base + variation).unsqueeze(0).unsqueeze(0)

def make_depth_gradient(size, near=1.0, far=100.0):
    """
    Creates a horizontal depth gradient from Near (Left) to Far (Right).
    """
    # Linear interpolation in depth space
    d = torch.linspace(near, far, size)
    # Expand to (1, 1, H, W)
    depth_map = d.view(1, 1, 1, size).expand(1, 1, size, size)
    return depth_map

def make_step_depth(size, split_idx, near=1.0, far=100.0):
    """
    Left side = Near (Degrade), Right side = Far (Keep).
    """
    depth_map = torch.ones(1, 1, size, size) * near
    depth_map[..., split_idx:] = far
    return depth_map

def generate_fft_depth_blend(image, depth_map, cutoff_radius, maximal_radius, gamma=1.0):
    """
    FFT baseline
    """
    near_img = generate_structured_noise_batch_vectorized(
        image_batch=image, 
        cutoff_radius=float(cutoff_radius),
        noise_std=1.0,
        pad_factor=1.5,
        sampling_method='fft'
    )
    
    far_img = generate_structured_noise_batch_vectorized(
        image_batch=image,
        cutoff_radius=float(maximal_radius),
        noise_std=1.0,
        pad_factor=1.5,
        sampling_method='fft'
    )
    
    eps = 1e-8
    disparity = 1.0 / (depth_map + eps)
    d_min = disparity.min()
    d_max = disparity.max()
    
    # Disparity: 1.0 = Near, 0.0 = Far
    norm_disp = (disparity - d_min) / (d_max - d_min + eps)
    
    # Control: 0.0 = Near, 1.0 = Far
    control_map = 1.0 - norm_disp
    control_map = control_map ** gamma
    
    return control_map * far_img + (1.0 - control_map) * near_img

def run_comparison():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    size = 512

    # --- Setup Data ---
    x_zone = make_zone_plate(size).to(device)
    x_metal = make_brushed_metal(size).to(device)

    # Keep depth generation internally (used for blending), but DO NOT present it as "depth" in the plot.
    near_m, far_m = 1.0, 100.0
    depth_grad = make_depth_gradient(size, near=near_m, far=far_m).to(device)

    r_near, r_far, gamma = 5, 256, 100

    # --- Execute ---
    print("Running Zone Plate Tests...")
    fft_zone = generate_fft_depth_blend(x_zone, depth_grad, r_near, r_far, gamma)
    wav_zone = generate_wavelet_structured_noise_batch_vectorized(
        x_zone, r_near, r_far, depth_grad, noise_std=1.0, gamma=gamma
    )

    print("Running Brushed Metal Tests...")
    fft_metal = generate_fft_depth_blend(x_metal, depth_grad, r_near, r_far, gamma)
    wav_metal = generate_wavelet_structured_noise_batch_vectorized(
        x_metal, r_near, r_far, depth_grad, noise_std=1.0, gamma=gamma
    )

    # --- To numpy (avoid repeated cpu/numpy calls) ---
    zone_in   = x_zone[0, 0].detach().cpu().numpy()
    zone_fft  = fft_zone[0, 0].detach().cpu().numpy()
    zone_wav  = wav_zone[0, 0].detach().cpu().numpy()
    metal_in  = x_metal[0, 0].detach().cpu().numpy()
    metal_fft = fft_metal[0, 0].detach().cpu().numpy()
    metal_wav = wav_metal[0, 0].detach().cpu().numpy()

    # --- Visualization (2x3 grid) ---
    fig, axs = plt.subplots(2, 3, figsize=(20, 14))
    plt.subplots_adjust(bottom=0.15, hspace=0.3)

    zx, zy = size // 2, size // 2
    z_size = 60

    def _add_threshold_frequency_bar(fig_, parent_ax):
        """Add a horizontal legend bar under `parent_ax` explaining threshold frequency."""
        pos = parent_ax.get_position()
        # [left, bottom, width, height] in figure coordinates
        cax = fig_.add_axes([pos.x0, pos.y0 - 0.05, pos.width, 0.012])

        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        cax.imshow(gradient, aspect="auto", cmap="gray")
        cax.set_yticks([])
        cax.set_xticks([0, 255])
        cax.set_xticklabels(
            ["Lower threshold frequency\n(more noise)",
             "Higher threshold frequency\n(more structure)"],
            fontsize=8
        )
        cax.set_xlabel("Threshold frequency", fontsize=9, labelpad=2)
        for spine in cax.spines.values():
            spine.set_visible(False)

    def format_plot(ax, data, title, *, show_bar=False, roi_color=None):
        ax.imshow(data, cmap="gray")
        ax.set_title(title, fontsize=15, pad=15, fontweight="bold")
        ax.axis("off")

        # Draw ROI on the parent axes if requested
        if roi_color is not None:
            rect = patches.Rectangle(
                (zx - z_size, zy - z_size),
                2 * z_size, 2 * z_size,
                linewidth=2, edgecolor=roi_color, facecolor="none", alpha=0.9
            )
            ax.add_patch(rect)

        if show_bar:
            _add_threshold_frequency_bar(fig, ax)

    def add_zoom_and_connect(ax, data, color):
        """Add zoom inset + connect it to ROI on the parent axis."""
        # --- ROI rectangle on the main axis (in data coords) ---
        x0, x1 = zx - z_size, zx + z_size
        y0, y1 = zy - z_size, zy + z_size

        roi = patches.Rectangle(
            (x0, y0), 2 * z_size, 2 * z_size,
            linewidth=5, edgecolor=color, facecolor="none", alpha=0.9
        )
        ax.add_patch(roi)

        # --- Inset axis with controlled size ---
        axins = inset_axes(ax, width="33%", height="33%", loc="lower right", borderpad=1.0)
        axins.imshow(data, cmap="gray")
        axins.set_xlim(x0, x1)
        axins.set_ylim(y1, y0)  # invert y to match image display
        axins.set_xticks([]); axins.set_yticks([])

        for spine in axins.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(5)

        # --- Connect BL->BL and TR->TR (use axis-fraction coordinates to hit inset corners) ---
        # Main axis corners (data coords)
        bl_data = (x0, y1)  # bottom-left in image display sense
        tr_data = (x1, y0)  # top-right in image display sense

        # Inset corners (axes-fraction coords): BL=(0,0), TR=(1,1)
        con_bl = patches.ConnectionPatch(
            xyA=(0, 0), coordsA=axins.transAxes,
            xyB=bl_data, coordsB=ax.transData,
            color=color, linewidth=1.5
        )
        con_tr = patches.ConnectionPatch(
            xyA=(1, 1), coordsA=axins.transAxes,
            xyB=tr_data, coordsB=ax.transData,
            color=color, linewidth=1.5
        )

        ax.add_artist(con_bl)
        ax.add_artist(con_tr)

        return axins

    # Row 1: Zone Plate
    format_plot(axs[0, 0], zone_in,  "Input: Zone Plate", show_bar=False)
    format_plot(axs[0, 1], zone_fft, "FFT Blending", show_bar=False, roi_color="red")
    format_plot(axs[0, 2], zone_wav, "DTCWPT",       show_bar=False, roi_color="green")
    add_zoom_and_connect(axs[0, 1], zone_fft, "red")
    add_zoom_and_connect(axs[0, 2], zone_wav, "green")

    # Row 2: Brushed Metal
    format_plot(axs[1, 0], metal_in,  "Input: Brushed Metal", show_bar=False)
    format_plot(axs[1, 1], metal_fft, "FFT Blending",         show_bar=False)
    format_plot(axs[1, 2], metal_wav, "DTCWPT",               show_bar=False)

    # Save the full grid (existing behavior)
    plt.savefig("fft_vs_wavelet.png", dpi=300, bbox_inches="tight")

    # --- Save 6 standalone panels (for LaTeX subfigures) ---
    def _save_panel(fname_stem, data, title, *, add_bar, add_zoom, color):
        f, ax = plt.subplots(1, 1, figsize=(6.5, 5.2))
        ax.imshow(data, cmap="gray")
        ax.set_title(title, fontsize=14, pad=10, fontweight="bold")
        ax.axis("off")

        if add_zoom:
            # --- ROI rectangle on the main axis (in data coords) ---
            x0, x1 = zx - z_size, zx + z_size
            y0, y1 = zy - z_size, zy + z_size

            roi = patches.Rectangle(
                (x0, y0), 2 * z_size, 2 * z_size,
                linewidth=5, edgecolor=color, facecolor="none", alpha=0.9
            )
            ax.add_patch(roi)

            # --- Inset axis with controlled size ---
            axins = inset_axes(ax, width="33%", height="33%", loc="lower right", borderpad=1.0)
            axins.imshow(data, cmap="gray")
            axins.set_xlim(x0, x1)
            axins.set_ylim(y1, y0)  # invert y to match image display
            axins.set_xticks([]); axins.set_yticks([])

            for spine in axins.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(5)

            # --- Connect BL->BL and TR->TR (use axis-fraction coordinates to hit inset corners) ---
            # Main axis corners (data coords)
            bl_data = (x0, y1)  # bottom-left in image display sense
            tr_data = (x1, y0)  # top-right in image display sense

            # Inset corners (axes-fraction coords): BL=(0,0), TR=(1,1)
            con_bl = patches.ConnectionPatch(
                xyA=(0, 0), coordsA=axins.transAxes,
                xyB=bl_data, coordsB=ax.transData,
                color=color, linewidth=1.5
            )
            con_tr = patches.ConnectionPatch(
                xyA=(1, 1), coordsA=axins.transAxes,
                xyB=tr_data, coordsB=ax.transData,
                color=color, linewidth=1.5
            )

            ax.add_artist(con_bl)
            ax.add_artist(con_tr)

        if add_bar:
            pos = ax.get_position()
            cax = f.add_axes([pos.x0, pos.y0 - 0.05, pos.width, 0.018])
            gradient = np.linspace(0, 1, 256).reshape(1, -1)
            cax.imshow(gradient, aspect="auto", cmap="gray")
            cax.set_yticks([])
            cax.set_xticks([0, 255])
            cax.set_xticklabels(
                ["Lower threshold frequency\n(more noise)",
                 "Higher threshold frequency\n(more structure)"],
                fontsize=8
            )
            cax.set_xlabel("Threshold frequency", fontsize=9, labelpad=2)
            for spine in cax.spines.values():
                spine.set_visible(False)

        for ext in ("pdf",):
            f.savefig(f"{fname_stem}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(f)

    _save_panel("panel_a_zone_input",  zone_in,   "",             add_bar=False, add_zoom=False, color="red")
    _save_panel("panel_b_zone_fft",    zone_fft,  "",                  add_bar=False,  add_zoom=True,  color="red")
    _save_panel("panel_c_zone_dtcwpt", zone_wav,  "",                        add_bar=False,  add_zoom=True,  color="green")
    _save_panel("panel_d_metal_input", metal_in,  "",          add_bar=False, add_zoom=False, color="red")
    _save_panel("panel_e_metal_fft",   metal_fft, "",                  add_bar=False,  add_zoom=False, color="red")
    _save_panel("panel_f_metal_dtcwpt",metal_wav, "",                        add_bar=False,  add_zoom=False, color="green")

    plt.close(fig)



if __name__ == "__main__":
    run_comparison()