"""
DINO-preserving noise generation for DINO-PD training.

For each input latent z0 and timestep t, optimizes a noisy point z_t* on the
flow path that preserves DINOv2 semantic features of z0.

The implied DINO-preserving flow endpoint is recovered as:
    z1* = (z_t* - (1-t)*z0) / t

Training signal (flow-matching velocity target):
    v = z1* - z0 = (z_t* - z0) / t

Usage:
    dino = load_dino(device)
    z_t_star = find_dino_preserving_noise(z0, t, pipe.vae_decoder, dino)
    z1_star  = (z_t_star - (1 - t) * z0) / t
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn.functional as F


def load_dino(model_name: str = "dinov2_vitl14_reg", device: str = "cuda") -> torch.nn.Module:
    """Load DINOv2 ViT-L/14 with registers in float32, all params frozen."""
    dino = torch.hub.load("facebookresearch/dinov2", model_name)
    dino = dino.to(device).to(torch.float32).eval()
    for p in dino.parameters():
        p.requires_grad_(False)
    return dino


def latent_to_dino(
    vae_decoder: torch.nn.Module,   # DiffSynth FluxVAEDecoder
    dino: torch.nn.Module,          # DINOv2
    z: torch.Tensor,                # (N, 16, H, W) DiffSynth-scaled latent, float32
) -> dict[str, torch.Tensor]:
    """
    Differentiable path: DiffSynth FLUX latent → DINOv2 features.

    Uses the DiffSynth VAE decoder directly (which handles its own
    scaling_factor / shift_factor unscaling), then passes the decoded
    [0, 1] image to DINOv2.

    Returns:
        {"cls": (N, 1024), "patches": (N, num_patches, 1024)}
    """
    vae_dtype = next(vae_decoder.parameters()).dtype

    # Decode via DiffSynth decoder — handles unscaling internally
    decoded = vae_decoder(z.to(vae_dtype), tiled=False).float()  # (N, 3, H, W) in ~[-1, 1]

    # [-1, 1] → [0, 1], resize to DINOv2 patch-aligned size
    img = (decoded + 1.0) / 2.0
    img = img.clamp(0.0, 1.0)
    dino_size = dino.patch_size * (img.shape[-1] // dino.patch_size)  # keep aspect-ratio agnostic
    img = F.interpolate(img, size=(dino_size, dino_size), mode="bilinear", align_corners=False)

    out = dino.forward_features(img)
    return {
        "cls":     out["x_norm_clstoken"],     # (N, 1024)
        "patches": out["x_norm_patchtokens"],  # (N, num_patches, 1024)
    }


def dino_distance(feat_a: dict, feat_b: dict) -> torch.Tensor:
    """
    Symmetric cosine distance: 0.5 * (1 - cos_cls) + 0.5 * (1 - mean_patch_cos).
    Range [0, 1]; 0 = identical, 1 = orthogonal.
    """
    cls_dist   = 1.0 - F.cosine_similarity(feat_a["cls"],     feat_b["cls"],     dim=-1).mean()
    patch_dist = 1.0 - F.cosine_similarity(feat_a["patches"], feat_b["patches"], dim=-1).mean()
    return 0.5 * cls_dist + 0.5 * patch_dist


def find_dino_preserving_noise(
    z0: torch.Tensor,               # (N, 16, H, W) float32 DiffSynth latent
    t: float,                       # flow timestep in (0, 1]
    vae_decoder: torch.nn.Module,   # DiffSynth FluxVAEDecoder on the correct device
    dino: torch.nn.Module,          # DINOv2 on the correct device
    n_steps: int = 300,
    z1: Optional[torch.Tensor] = None,  # noise endpoint initialization; sampled N(0,1) if None
) -> torch.Tensor:
    """
    Find z_t* = argmin_{z_t} dino_distance(z_t, z0) s.t. z_t initialized on the flow path.

    Concretely:
      1. If z1 is None, sample z1 ~ N(0, std(z0)).
      2. Initialize z_t = (1-t)*z0 + t*z1.
      3. Optimize z_t with Adam to minimise dino_distance(F(dec(z_t)), F(dec(z0))).

    The optimised z_t* can be converted back to a flow endpoint via:
        z1* = (z_t* - (1-t)*z0) / t

    Args:
        z0:       Clean image latent (DiffSynth-scaled, float32).
        t:        Flow timestep in (0, 1].
        vae_decoder: DiffSynth FluxVAEDecoder, same device as z0.
        dino:     DINOv2 model (float32), same device as z0.
        n_steps:  Adam optimisation steps.
        z1:       Optional initialisation noise (same shape as z0).

    Returns:
        z_t*: (N, 16, H, W) float32, detached.
    """
    z0 = z0.float()

    # Compute DINOv2 target features once (no grad needed)
    with torch.no_grad():
        target = latent_to_dino(vae_decoder, dino, z0)
        target = {k: v.detach() for k, v in target.items()}

    # Initialise noise endpoint
    if z1 is None:
        z1 = torch.randn_like(z0) * z0.std()

    # Start from the flow-interpolated point
    z_t = ((1.0 - t) * z0 + t * z1.float()).detach().requires_grad_(True)

    optimizer = torch.optim.Adam([z_t], lr=1e-2)

    for _ in range(n_steps):
        optimizer.zero_grad()
        feats = latent_to_dino(vae_decoder, dino, z_t)
        loss  = dino_distance(feats, target)
        loss.backward()
        optimizer.step()

    return z_t.detach()
