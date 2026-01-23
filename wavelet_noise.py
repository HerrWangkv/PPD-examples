from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
from operator import inv
from typing import Optional, Tuple, Union, List

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pytorch_wavelets import DTCWTForward, DTCWTInverse
from structured_noise import generate_structured_noise_batch_vectorized
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes


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
    max_n: int = 12,
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
        # Convert float -> rational, then verify denominator is power of two.
        # Using Fraction(str(t)) avoids some binary-float artifacts if you pass a decimal literal.
        frac = Fraction(str(t)).limit_denominator(1 << max_n)
        k, d = frac.numerator, frac.denominator
        if tol > 0.0:
            # allow small mismatch: find best dyadic approx
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
    
def resize_mask(mask_n1hw: torch.Tensor, h: int, w: int, mode: str = "nearest") -> torch.Tensor:
    return F.interpolate(mask_n1hw.float(), size=(h, w), mode=mode)


def complex_phase(re: torch.Tensor, im: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # atan2 is stable; eps not strictly needed, kept for symmetry with mag
    return torch.atan2(im, re)


def complex_mag(re: torch.Tensor, im: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.sqrt(re * re + im * im + eps)

def fuse_noise_only(C_noise: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # magnitude + phase from noise
    re_n, im_n = C_noise[..., 0], C_noise[..., 1]
    phi_n = complex_phase(re_n, im_n, eps=eps)
    mag_n = complex_mag(re_n, im_n, eps=eps)
    re = mag_n * torch.cos(phi_n)
    im = mag_n * torch.sin(phi_n)
    return torch.stack([re, im], dim=-1)

def fuse_subband_phase_from_signal_mag_from_noise(
    C_in: torch.Tensor,     # (N,C,H,W,2)
    C_noise: torch.Tensor,  # (N,C,H,W,2)
    mask_n1hw: torch.Tensor,  # (N,1,H,W) in this subband resolution
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Build C_mix = |C_noise| * exp(j * (mask*phase(C_in) + (1-mask)*phase(C_out))).
    """
    re_in,  im_in  = C_in[..., 0], C_in[..., 1]
    re_n,   im_n   = C_noise[..., 0], C_noise[..., 1]

    phi_in  = complex_phase(re_in,  im_in,  eps=eps)
    phi_n = complex_phase(re_n,   im_n,   eps=eps)

    # broadcast mask to (N,C,H,W)
    m = mask_n1hw
    if m.ndim != 4:
        raise ValueError(f"mask must be (N,1,H,W), got {tuple(m.shape)}")
    m = m.expand(phi_in.shape[0], phi_in.shape[1], phi_in.shape[2], phi_in.shape[3])

    phi = m * phi_in + (1.0 - m) * phi_n#phi_out

    mag = complex_mag(re_n, im_n, eps=eps)

    re = mag * torch.cos(phi)
    im = mag * torch.sin(phi)
    return torch.stack([re, im], dim=-1)

def fuse_subband_phase_from_signal_mag_from_noise_global(C_phase_src, C_noise, eps=1e-8):
    re_p, im_p = C_phase_src[...,0], C_phase_src[...,1]
    re_n, im_n = C_noise[...,0], C_noise[...,1]
    phi = torch.atan2(im_p, re_p)
    mag = torch.sqrt(re_n*re_n + im_n*im_n + eps)
    re = mag * torch.cos(phi)
    im = mag * torch.sin(phi)
    return torch.stack([re, im], dim=-1)

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
    """Split a *complex* subband into low/high halves using one-level DTCWT on stacked (re,im) channels."""
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
        
        # 3. Crop if necessary (DTCWT padding fix)
        if low_cplx.shape[-3] != orig_h or low_cplx.shape[-2] != orig_w:
            low_cplx = low_cplx[..., :orig_h, :orig_w, :]
            high_cplx = high_cplx[..., :orig_h, :orig_w, :]
            
        return low_cplx, high_cplx

def dyadic_to_grid(dc, n_target: int):
    """Return (k_grid, n_target, t_float) where k_grid is numerator on 2^n_target grid."""
    if n_target < dc.n:
        raise ValueError("n_target must be >= dc.n")
    k_grid = dc.k << (n_target - dc.n)
    # t is exact dyadic; float is just for comparisons/debug
    t_float = k_grid / float(1 << n_target)
    return k_grid, n_target, t_float

class DTCWTFusePhaseMag_Recursive(nn.Module):
    def __init__(self, biort="antonini", qshift="qshift_d", mode="symmetric", o_dim=2, ri_dim=-1):
        super().__init__()
        self.o_dim = o_dim
        # Only need inverse for final reconstruction, and splitter for the recursion
        self.ifm = DTCWTInverse(biort=biort, qshift=qshift, mode=mode, o_dim=o_dim, ri_dim=ri_dim)
        self.splitter = ComplexBandPacketSplitter(biort=biort, qshift=qshift, mode=mode, o_dim=o_dim, ri_dim=ri_dim)

    def forward(
        self,
        LL_img, C_img,      # Source Image Coefficients
        LL_z, C_z,          # Noise Coefficients
        mask_n1hw,          # Binary Mask
        t_all: float,       # Global keep threshold (e.g. 0.375)
        t_mask: float,      # Mask keep threshold (e.g. 0.625)
        pad_factor: float = 1.5,
        max_packet_depth: int = 12,
        mask_mode: str = "nearest",
        eps: float = 1e-8
    ):
        # 1. Fuse LL Band (Global Phase)
        # We assume LL is always low-freq enough to be <= t_all
        LL_mix = ll_fusion_fftshift_global_phase(LL_img, LL_z, pad_factor=pad_factor)

        # 2. Recursive Fusion for High Bands
        J = len(C_img)
        C_mix = []

        for l in range(J):
            # Level l covers freq range [2^{-(l+1)}, 2^{-l}]
            level_idx = l + 1
            freq_high = 1.0 / (2.0 ** (level_idx - 1))
            freq_low  = 1.0 / (2.0 ** level_idx)
            
            C_mix_l = []
            
            # Prepare mask for this level
            ref = C_z[l][0]
            H_l, W_l = ref.shape[-3], ref.shape[-2]
            m_l = resize_mask(mask_n1hw, H_l, W_l, mode=mask_mode)

            for o in range(6):
                # Process each orientation
                c_fused = self._process_band_recursive(
                    C_img[l][o], C_z[l][o],
                    freq_low, freq_high,
                    m_l, 
                    t_all, t_mask,
                    0, max_packet_depth, eps
                )
                C_mix_l.append(c_fused)
            
            C_mix.append(C_mix_l)

        # 3. Inverse Transform
        yh_mix = pack_C_to_yh_list(C_mix, o_dim=self.o_dim)
        x_hat = self.ifm((LL_mix, yh_mix))
        
        return x_hat, LL_mix, C_mix

    def _process_band_recursive(self, c_img, c_nz, f_start, f_end, mask, t_all, t_mask, depth, max_depth, eps):
        # Base Case: Global (Frequency <= t_all)
        if f_end <= t_all + 1e-9:
             return fuse_subband_phase_from_signal_mag_from_noise_global(c_img, c_nz, eps=eps)

        # Base Case: Noise Only (Frequency > t_mask)
        if f_start >= t_mask - 1e-9:
            return fuse_noise_only(c_nz, eps=eps)

        # Base Case: Masked (t_all < Frequency <= t_mask)
        # If the entire packet fits inside the "Mask" zone
        if (f_start >= t_all - 1e-9) and (f_end <= t_mask + 1e-9):
            return fuse_subband_phase_from_signal_mag_from_noise(c_img, c_nz, mask, eps=eps)

        # Base Case: Max Depth Limit (Avoid infinite recursion)
        if depth >= max_depth:
            mid = (f_start + f_end) / 2
            if mid <= t_all:
                return fuse_subband_phase_from_signal_mag_from_noise_global(c_img, c_nz, eps=eps)
            elif mid <= t_mask:
                return fuse_subband_phase_from_signal_mag_from_noise(c_img, c_nz, mask, eps=eps)
            else:
                return fuse_noise_only(c_nz, eps=eps)

        # Recursive Split
        lo_img, hi_img = self.splitter.split_once(c_img)
        lo_nz, hi_nz   = self.splitter.split_once(c_nz)
        
        H_sub, W_sub = lo_img.shape[-3], lo_img.shape[-2]
        sub_mask = F.interpolate(mask.float(), size=(H_sub, W_sub), mode='nearest')
        
        mid_freq = (f_start + f_end) / 2.0
        
        # Recurse
        out_lo = self._process_band_recursive(lo_img, lo_nz, f_start, mid_freq, sub_mask, t_all, t_mask, depth + 1, max_depth, eps)
        out_hi = self._process_band_recursive(hi_img, hi_nz, mid_freq, f_end, sub_mask, t_all, t_mask, depth + 1, max_depth, eps)
        return out_lo + out_hi

def _generate_random_ellipse_masks(N, H, W, device):
    """Generates random rotated elliptical masks for training robustness."""
    masks = torch.zeros((N, 1, H, W), device=device)
    
    # Create coordinate grid once
    # Shape: (H, W)
    y_grid = torch.arange(H, device=device).float().view(H, 1).expand(H, W)
    x_grid = torch.arange(W, device=device).float().view(1, W).expand(H, W)
    
    for i in range(N):
        num_dots = random.randint(5, 15)
        
        for _ in range(num_dots):
            # Center
            cx = random.uniform(0, W)
            cy = random.uniform(0, H)
            
            # Radii (Major/Minor axes)
            min_dim = min(H, W)
            rx = random.uniform(min_dim * 0.01, min_dim * 0.15)
            ry = random.uniform(min_dim * 0.01, min_dim * 0.15)
            
            # Rotation angle (radians)
            theta = random.uniform(0, 2 * math.pi)
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            
            # Translate to (0,0) -> Rotate -> Scale -> Check <= 1
            dx = x_grid - cx
            dy = y_grid - cy
            
            # Rotated coordinates
            # x' = x*cos + y*sin
            # y' = -x*sin + y*cos
            x_rot = dx * cos_t + dy * sin_t
            y_rot = -dx * sin_t + dy * cos_t
            
            # Ellipse equation: (x'/rx)^2 + (y'/ry)^2 <= 1
            ellipse_mask = ((x_rot / rx)**2 + (y_rot / ry)**2 <= 1.0).float()
            
            # Combine
            masks[i, 0] = torch.max(masks[i, 0], ellipse_mask)
            
    return masks

def _sample_dyadic(min_val, max_val, max_n=5):
    """
    Returns a random float k / 2^n in [min_val, max_val].
    Using fixed denominator 2^max_n ensures all values are exactly representable
    in the wavelet packet tree up to depth max_n.
    """
    denom = 1 << max_n # e.g. 32
    k_min = math.ceil(min_val * denom)
    k_max = math.floor(max_val * denom)
    
    # Ensure range is valid
    if k_min > k_max: 
        k_min = k_max
        
    k = random.randint(k_min, k_max)
    return k / float(denom)

def generate_wavelet_structured_noise_batch_vectorized(
    image_batch: torch.Tensor,
    mask_keep: Optional[float] = None, # If None, sample automatically (training mode)
    all_keep: Optional[float] = None,  # If None, sample automatically (training mode)
    binary_mask: Optional[torch.Tensor] = None,
    noise_std: float = 1.0,
    pad_factor: float = 1.5,
    input_noise: torch.Tensor = None,
    biort: str = 'near_sym_b',
    qshift: str = 'qshift_b',
    random_mask_prob: float = 0.5, # Probability to use random box masks if in training mode
):
    """
    Generates CDTWPT Structured Noise.
    
    Updates:
    1. Sampling: Uses strictly Dyadic Rationals (k/32) for thresholds.
    2. Masks: Uses Random Rotated Ellipses instead of Dots.
    """
    if image_batch.ndim != 4:
        raise ValueError(f"Expected image_batch in NCHW")
    
    device = image_batch.device
    dtype = image_batch.dtype
    N, C, H, W = image_batch.shape
    image_batch = image_batch.float()
    
    # 1. Dyadic Parameter Sampling (Training Mode)
    training_mode = (mask_keep is None) or (all_keep is None)
    if training_mode:
        # Sample base quality: e.g. 0.125 (4/32) to 0.5 (16/32)
        # We use strict dyadic sampling to ensure recursion termination
        all_keep = _sample_dyadic(0.125, 0.5)
        
        # Constraint: Ensure distinct separation (at least 0.2 gap)
        # but strictly aligned to the grid
        mask_keep = _sample_dyadic(all_keep, 1.0)
    # 2. Mask Handling (Ellipses)
    if binary_mask is None:
        if training_mode and random.random() < random_mask_prob:
            # Synthetic Random Rotated Ellipse Mask
            binary_mask = _generate_random_ellipse_masks(N, 8*H, 8*W, device)
        else:
            # Global Degradation
            binary_mask = torch.ones((N, 1, H, W), device=device)
    else:
        binary_mask = binary_mask.to(device)

    if input_noise is None:
        z = torch.randn_like(image_batch) * float(noise_std)
    else:
        z = input_noise.to(device)

    # 3. Decomposition
    J = max(1, math.ceil(-math.log2(all_keep + 1e-9)))
    decomp = DTCWTDecomposer(J=J, biort=biort, qshift=qshift).to(device)
    
    with torch.no_grad():
        LL_img, C_img = decomp(image_batch)
        LL_z, C_z = decomp(z)

        fuser = DTCWTFusePhaseMag_Recursive(biort=biort, qshift=qshift).to(device)
        
        x_hat, _, _ = fuser(
            LL_img, C_img,
            LL_z, C_z,
            binary_mask,
            t_all=all_keep,
            t_mask=mask_keep,
            pad_factor=pad_factor
        )

    return x_hat.to(dtype=dtype)

def create_frequency_probe(shape, target_band_idx, biort='near_sym_b', qshift='qshift_b', device='cuda'):
    """
    Creates an image that has energy ONLY at a specific wavelet decomposition level (frequency band).
    target_band_idx: 0 = Highest Freq (Level 1), 1 = Level 2, etc.
    """
    N, C, H, W = shape
    J = 6  # Deep enough decomposition
    
    # We construct the probe in the WAVELET domain directly
    # Start with zeros
    inv = DTCWTInverse(biort=biort, qshift=qshift).to(device)
    
    # Create empty coefficients
    # We need to know the shapes, so let's do a dummy forward pass
    dummy = torch.zeros(N, C, H, W, device=device)
    decomp = DTCWTDecomposer(J=J, biort=biort, qshift=qshift).to(device)
    LL, Cs = decomp(dummy)
    
    # Construct our probe coefficients
    C_probe = []
    for l, C_level in enumerate(Cs):
        C_l_new = []
        for o in range(6): # For all orientations
            # If this is our target level, fill it with 1.0 (signal)
            # Otherwise keep it 0.0
            if l == target_band_idx:
                # specific pattern to be visible
                t = torch.ones_like(C_level[o]) 
            else:
                t = torch.zeros_like(C_level[o])
            C_l_new.append(t)
        C_probe.append(C_l_new)
        
    # If target is deeper than J, put it in LL (lowest freq)
    if target_band_idx >= J:
        LL_probe = torch.ones_like(LL)
    else:
        LL_probe = torch.zeros_like(LL)
        
    # Inverse to get the spatial image
    yh_probe = pack_C_to_yh_list(C_probe)
    x_probe = inv((LL_probe, yh_probe))
    
    return x_probe

def generate_fft_blending_wrapper(image, mask, noise, all_keep, mask_keep):
    """
    Calls the FFT Blending function twice and blends them.
    Converts normalized cutoffs (0.0-1.0) to pixel radii.
    """
    N, C, H, W = image.shape
    # Approximate radius conversion: 1.0 = Nyquist (H/2)
    # Note: Pad factor 1.5 increases the FFT grid size. 
    # The function expects cutoff relative to the PADDED size in pixels.
    pad_factor = 1.5
    H_pad, W_pad = int(H * pad_factor), int(W * pad_factor)
    
    # Calculate radius in pixels for the padded grid
    # all_keep is roughly "fraction of max radius"
    r_global = all_keep * (min(H_pad, W_pad) / 2)
    r_mask   = mask_keep * (min(H_pad, W_pad) / 2)
    
    # 1. Generate Global Stream
    out_global = generate_structured_noise_batch_vectorized(
        image, noise_std=1.0, pad_factor=pad_factor,
        cutoff_radius=r_global,
        input_noise=noise, sampling_method='fft'
    )
    
    # 2. Generate Mask Stream
    out_mask = generate_structured_noise_batch_vectorized(
        image, noise_std=1.0, pad_factor=pad_factor,
        cutoff_radius=r_mask,
        input_noise=noise, sampling_method='fft'
    )
    
    # 3. Blend
    return mask * out_mask + (1 - mask) * out_global

def make_zone_plate(size):
    """Generates a pattern where frequency increases with distance from center."""
    x = torch.linspace(-1, 1, size)
    y = torch.linspace(-1, 1, size)
    Y, X = torch.meshgrid(y, x, indexing='ij')
    R2 = X**2 + Y**2
    
    # Calculate factor to hit Nyquist at R=1
    # Freq = d/dr(k*r^2) = 2kr. Nyquist is pi*(size/2). Normalized... 
    # Empirical factor ~ 400 fills the 512 spectrum nicely without excessive aliasing.
    return torch.cos(400 * R2).unsqueeze(0).unsqueeze(0)

def make_split_mask(size):
    """
    Left Half = 0 (Outside Mask)
    Right Half = 1 (Inside Mask)
    """
    mask = torch.zeros(1, 1, size, size)
    mask[..., size//2:] = 1.0 # Right side is Keep
    return mask

def make_brushed_metal(size):
    """Generates diagonal lines to simulate brushed metal or wood grain."""
    x = torch.linspace(-1, 1, size)
    y = torch.linspace(-1, 1, size)
    Y, X = torch.meshgrid(y, x, indexing='ij')
    
    # Create diagonal sine waves (45 degrees)
    # sin(k * (x + y)) -> Diagonal wavefronts
    # Add some low freq variation to make it look organic
    base = torch.sin(100 * (X + Y)) 
    variation = 0.5 * torch.sin(20 * X) 
    return (base + variation).unsqueeze(0).unsqueeze(0)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    size = 512 
    
    # Data
    x_img = make_zone_plate(size).to(device)
    z_noise = torch.randn_like(x_img)
    mask = make_split_mask(size).to(device)
    
    all_keep = 0.15 
    mask_keep = 0.65
    
    print(f"Generating Comparison (Size {size})...")

    out_wavelet = generate_wavelet_structured_noise_batch_vectorized(
        image_batch=x_img, mask_keep=mask_keep, all_keep=all_keep,
        binary_mask=mask, input_noise=z_noise
    )

    # 2. FFT (Hard Cutoff)
    out_fft = generate_fft_blending_wrapper(
        x_img, mask, z_noise, all_keep=all_keep, mask_keep=mask_keep
    )

    # ==========================================
    # 3. Visualization
    # ==========================================
    
    # Convert to numpy
    img_np  = x_img[0,0].cpu().numpy()
    mask_np = mask[0,0].cpu().numpy()
    wave_np = out_wavelet[0,0].cpu().numpy()
    fft_np  = out_fft[0,0].cpu().numpy()

    # Create 1x4 Grid
    fig, axs = plt.subplots(1, 4, figsize=(24, 6))
    
    zoom_half_size = 40
    # X: Centered on vertical seam (256)
    x1, x2 = size//2 - zoom_half_size, size//2 + zoom_half_size # 216 - 296
    
    # Y: Centered on the Green Circle top edge (~218)
    # Let's pick center y=210. 
    y_center = size//2 - 46 # approx 210
    y1, y2 = y_center - zoom_half_size, y_center + zoom_half_size # 170 - 250
    
    # Rectangle for main image (Top-Left corner, Width, Height)
    # In imshow, y1 (170) is higher/top, y2 (250) is lower/bottom.
    rect_x = x1
    rect_y = y1 
    rect_w = x2 - x1
    rect_h = y2 - y1

    def plot_clean(ax, data, title, is_fft=False, show_circles=True):
        ax.imshow(data, cmap='gray')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Circles
        if show_circles:
            center = size // 2
            r1 = int(all_keep * center)
            r2 = int(mask_keep * center)
            ax.add_patch(plt.Circle((center, center), r1, color='lime', fill=False, lw=1, ls='--', alpha=0.7))
            ax.add_patch(plt.Circle((center, center), r2, color='red', fill=False, lw=1, ls='--', alpha=0.7))

        # 1. Draw Rectangle (Corrected logic)
        rect = patches.Rectangle((rect_x, rect_y), rect_w, rect_h, linewidth=2, edgecolor='yellow', facecolor='none')
        ax.add_patch(rect)
        
        # 2. Draw Zoom Inset
        axins = zoomed_inset_axes(ax, zoom=2.5, loc='lower right')
        axins.imshow(data, cmap='gray')
        axins.set_xlim(x1, x2)
        axins.set_ylim(y2, y1) # Flip Y: Bottom(250) -> Top(170)
        axins.set_xticks([])
        axins.set_yticks([])
        
        # Border
        for spine in axins.spines.values():
            spine.set_edgecolor('yellow')
            spine.set_linewidth(2)

        # Annotations
        if is_fft:
            ax.text(0.5, -0.1, "Artifact: Ghosting at Seam", transform=ax.transAxes, 
                    ha='center', color='red', fontweight='bold')
        elif show_circles:
            ax.text(0.5, -0.1, "Benefit: Natural Transition", transform=ax.transAxes, 
                    ha='center', color='green', fontweight='bold')

    # Plot
    plot_clean(axs[0], img_np, "Input Zone Plate", show_circles=False)
    plot_clean(axs[1], mask_np, "Mask (Right=Keep)", show_circles=False)
    plot_clean(axs[2], fft_np, "FFT Blend (Soft)", is_fft=True)
    plot_clean(axs[3], wave_np, "Wavelet Output", show_circles=True)

    plt.savefig("fft_vs_wavelet1.png", dpi=150)
    plt.close()

    # 1. Data: Brushed Metal (Diagonal)
    x_img = make_brushed_metal(size).to(device)
    z_noise = torch.randn_like(x_img)
    
    # Simple Split Mask (Left=Noise, Right=Signal)
    mask = torch.zeros(1, 1, size, size).to(device)
    mask[..., size//2:] = 1.0 
    
    # Cutoffs: We want to preserve the MAIN diagonal lines (Low Freq)
    # but replace the fine detail with noise.
    all_keep = 0.10  # Keep base structure
    mask_keep = 0.90 # Keep almost everything in mask
    
    print(f"Generating Brushed Metal Comparison...")

    # 2. Generate
    out_wavelet = generate_wavelet_structured_noise_batch_vectorized(
        image_batch=x_img, mask_keep=mask_keep, all_keep=all_keep,
        binary_mask=mask, input_noise=z_noise
    )

    out_fft = generate_fft_blending_wrapper(
        x_img, mask, z_noise, all_keep=all_keep, mask_keep=mask_keep
    )

    # 3. Visualization
    img_np  = x_img[0,0].cpu().numpy()
    wave_np = out_wavelet[0,0].cpu().numpy()
    fft_np  = out_fft[0,0].cpu().numpy()

    fig, axs = plt.subplots(1, 4, figsize=(24, 6))
    
    # Zoom Region: Focus on the LEFT side (Mask=0 / Noise Zone)
    # We want to see how the texture degrades.
    # Center y, Left x.
    zoom_size = 80
    x_center = 256 # Middle of the left half (0-256)
    y_center = 256
    
    x1, x2 = x_center - zoom_size//2, x_center + zoom_size//2
    y1, y2 = y_center - zoom_size//2, y_center + zoom_size//2
    
    # Rect coords (Top-Left for patches)
    # y1 is top (smaller index), y2 is bottom (larger index)
    rect_x = x1
    rect_y = y1
    rect_w = x2 - x1
    rect_h = y2 - y1

    def plot_clean_texture(ax, data, title, is_fft=False):
        ax.imshow(data, cmap='gray')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')

        # 1. Draw Indicator Box on Main Image
        rect = patches.Rectangle((rect_x, rect_y), rect_w, rect_h, 
                               linewidth=2, edgecolor='yellow', facecolor='none')
        ax.add_patch(rect)
        
        # 2. Floating Zoom Inset
        axins = zoomed_inset_axes(ax, zoom=2.0, loc='lower right')
        axins.imshow(data, cmap='gray')
        axins.set_xlim(x1, x2)
        axins.set_ylim(y2, y1) # Flip Y
        axins.set_xticks([])
        axins.set_yticks([])
        
        # Yellow border for inset
        for spine in axins.spines.values():
            spine.set_edgecolor('yellow')
            spine.set_linewidth(2)

        # 3. Text Annotations
        if is_fft:
            ax.text(0.5, -0.1, "Result: Speckle / 'Sand' Noise", transform=ax.transAxes, 
                    ha='center', color='red', fontweight='bold')
        elif "Wavelet" in title:
            ax.text(0.5, -0.1, "Result: Directional 'Scratches'", transform=ax.transAxes, 
                    ha='center', color='green', fontweight='bold')
        elif "Mask" in title:
             ax.text(0.5, -0.1, "(Black = Noise Zone)", transform=ax.transAxes, 
                    ha='center', color='black', fontweight='bold')

    plot_clean_texture(axs[0], img_np, "Input: Brushed Metal")
    plot_clean_texture(axs[1], mask_np, "Mask")
    plot_clean_texture(axs[2], fft_np, "FFT Blending Output", is_fft=True)
    plot_clean_texture(axs[3], wave_np, "Wavelet Output")

    plt.savefig("fft_vs_wavelet2.png", bbox_inches='tight')