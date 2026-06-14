"""
E3: Phase trajectory experiment.
For a single vKITTI image, run 50-step ψ-PD (dropll J=4 r=12) inference and
track per-subband DTCWT phase across all denoising steps.

Preserved (mid-freq, below r=12): HF subbands whose center frequency < cutoff
Non-preserved: LL (dropped), HF subbands above cutoff (high-freq)

Output: figures/phase_trajectory.png
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pytorch_wavelets import DTCWTForward

import glob, random
from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig
from wavelet_noise import generate_wavelet_structured_noise_batch_vectorized, DTCWTDecomposer

# ── Config ──────────────────────────────────────────────────────────────────
N_SAMPLES    = 10
SEED         = 42
LORA_PATH    = "models/train/FLUX.1-dev_lora_wpd_dropll/step-6000.safetensors"
RADIUS       = 12
J            = 4
HEIGHT, WIDTH = 384, 1280
NUM_STEPS    = 50
PROMPT       = ("A photorealistic photograph taken from a forward-facing "
                "vehicle-mounted camera. Natural outdoor lighting, authentic "
                "surface textures, real-world colors.")
NEG_PROMPT   = ("blurry, low quality, cartoon, cg render, unrealistic, "
                "dashboard, steering wheel, windshield frame, car interior")
OUT_FIG      = "figures/phase_trajectory.png"

# ── Load pipeline ────────────────────────────────────────────────────────────
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
pipe.load_lora(pipe.dit, LORA_PATH, alpha=1)

# ── Sample images ────────────────────────────────────────────────────────────
all_images = sorted(glob.glob(
    "/mrtstorage/datasets_tmp/vkitti/vkitti_1.3.1_rgb/*/clone/*.png"
))
random.seed(SEED)
sampled_images = random.sample(all_images, N_SAMPLES)
print(f"Sampled {N_SAMPLES} images from {len(all_images)} total")

# ── DTCWT decomposer on GPU ──────────────────────────────────────────────────
decomp = DTCWTDecomposer(J=J).cuda()

def get_bands(z: torch.Tensor):
    """
    z: (1,C,H,W) on GPU
    Returns:
      LL:      (C, H_ll, W_ll)
      hf:      list[J] of list[6] of (re, im) tuples, each (C,H,W)
    """
    z_f = z.float().cuda()
    LL, C_list = decomp(z_f)
    hf = []
    for level_bands in C_list:
        level = []
        for band in level_bands:   # (1,C,H,W,2)
            level.append((band[0, :, :, :, 0], band[0, :, :, :, 1]))
        hf.append(level)
    return LL[0], hf

# ── Determine preserved vs non-preserved HF subbands ────────────────────────
# From wavelet_noise.py:
#   level l (0-indexed): freq_high = 1/2^l,  freq_low = 1/2^(l+1)
#   preserved  if freq_high <= freq_cutoff  (entire band below cutoff → image phase)
#   NOT preserved if freq_low >= freq_cutoff (entire band above cutoff → noise phase)
# freq_cutoff = RADIUS / nyquist_radius,  nyquist = min(H_lat, W_lat) / 2
H_lat = HEIGHT // 8
W_lat = WIDTH  // 8
nyquist = min(H_lat, W_lat) / 2.0
freq_cutoff = RADIUS / nyquist

preserved_levels = []
nonpreserved_levels = []
for l in range(J):
    freq_high = 1.0 / (2.0 ** l)
    freq_low  = 1.0 / (2.0 ** (l + 1))
    if freq_high <= freq_cutoff + 1e-9:
        preserved_levels.append(l)
    elif freq_low >= freq_cutoff - 1e-9:
        nonpreserved_levels.append(l)
    else:
        # mixed band — count as preserved (majority phase from image)
        preserved_levels.append(l)

print(f"Latent size: {H_lat} x {W_lat},  nyquist={nyquist},  freq_cutoff={freq_cutoff:.3f}")
print(f"Preserved HF levels  (freq below cutoff): {preserved_levels}")
print(f"Non-preserved HF levels (freq above cutoff): {nonpreserved_levels}")

# ── Phase / magnitude metrics ─────────────────────────────────────────────────
def phase_collinearity_per_level(hf_bands_a, hf_bands_b, level):
    """Magnitude-weighted |cos(Δφ)| = |Σ Re(a·conj(b))| / (Σ |a|·|b|),
    averaged over 6 orientations.  Near-zero coefficients get negligible weight."""
    vals = []
    for o in range(6):
        re_a, im_a = hf_bands_a[level][o]
        re_b, im_b = hf_bands_b[level][o]
        dot   = (re_a * re_b + im_a * im_b)   # Re(a · conj(b))
        mag_a = torch.sqrt(re_a**2 + im_a**2)
        mag_b = torch.sqrt(re_b**2 + im_b**2)
        w     = mag_a * mag_b
        vals.append((dot.abs().sum() / w.sum().clamp(min=1e-8)).item())
    return float(np.mean(vals))


# ── Hook: capture (z_t, v_θ) pairs ───────────────────────────────────────────
step_pairs = []   # list of (z_t, v_theta) CPU tensors per step
_orig_step = pipe.scheduler.step
def _hooked_step(model_output, timestep, sample, **kw):
    result = _orig_step(model_output, timestep, sample, **kw)
    step_pairs.append((sample.detach().cpu().clone(),
                       model_output.detach().cpu().clone()))
    return result

# ── Per-image loop ────────────────────────────────────────────────────────────
# all_cos_eps[l]  → (N_SAMPLES, NUM_STEPS): cos(∠z_t,  ∠ε_struct)
# all_cos_vel[l]  → (N_SAMPLES, NUM_STEPS): cos(∠v_θ,  ∠z_t)
all_cos_eps = {l: [] for l in range(J)}
all_cos_vel = {l: [] for l in range(J)}

for img_idx, img_path in enumerate(sampled_images):
    print(f"\n[{img_idx+1}/{N_SAMPLES}] {img_path}")
    step_pairs.clear()

    with torch.no_grad():
        img_pil = Image.open(img_path).convert("RGB").resize((WIDTH, HEIGHT), Image.LANCZOS)
        img_t   = pipe.preprocess_image(img_pil).to(device=pipe.device, dtype=pipe.torch_dtype)
        z_sim   = pipe.vae_encoder(img_t, tiled=False)

        input_noise = torch.randn_like(z_sim).float()
        eps_struct  = generate_wavelet_structured_noise_batch_vectorized(
            z_sim.float(), radius_map=RADIUS, input_noise=input_noise,
            drop_ll=True, J=J
        ).to(dtype=pipe.torch_dtype).contiguous()

        _, ref_hf = get_bands(eps_struct)

        pipe.scheduler.step = _hooked_step
        _ = pipe(
            prompt=PROMPT, negative_prompt=NEG_PROMPT,
            height=HEIGHT, width=WIDTH,
            cfg_scale=2, num_inference_steps=NUM_STEPS,
            noise=eps_struct,
        )
        pipe.scheduler.step = _orig_step

    img_cos_eps = {l: [] for l in range(J)}
    img_cos_vel = {l: [] for l in range(J)}
    for z_t_cpu, v_cpu in step_pairs:
        z_t_gpu = z_t_cpu.to(pipe.device, dtype=pipe.torch_dtype)
        v_gpu   = v_cpu.to(pipe.device, dtype=pipe.torch_dtype)
        _, z_hf = get_bands(z_t_gpu)
        _, v_hf = get_bands(v_gpu)
        for l in range(J):
            img_cos_eps[l].append(phase_collinearity_per_level(z_hf, ref_hf, l))
            img_cos_vel[l].append(phase_collinearity_per_level(v_hf, z_hf, l))

    for l in range(J):
        all_cos_eps[l].append(img_cos_eps[l])
        all_cos_vel[l].append(img_cos_vel[l])
    print(f"  [eps] L0={img_cos_eps[0][-1]:.3f} L1={img_cos_eps[1][-1]:.3f} "
          f"L2={img_cos_eps[2][-1]:.3f} L3={img_cos_eps[3][-1]:.3f}")
    print(f"  [vel] L0={img_cos_vel[0][-1]:.3f} L1={img_cos_vel[1][-1]:.3f} "
          f"L2={img_cos_vel[2][-1]:.3f} L3={img_cos_vel[3][-1]:.3f}")

steps = list(range(NUM_STEPS))
eps_mean = {l: np.mean(all_cos_eps[l], axis=0) for l in range(J)}
eps_std  = {l: np.std(all_cos_eps[l],  axis=0) for l in range(J)}
vel_mean = {l: np.mean(all_cos_vel[l], axis=0) for l in range(J)}
vel_std  = {l: np.std(all_cos_vel[l],  axis=0) for l in range(J)}

# ── Plot: two panels ──────────────────────────────────────────────────────────
import os; os.makedirs("figures", exist_ok=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

colors = {0: 'tomato', 1: '#2196F3', 2: '#4CAF50', 3: '#9C27B0'}
labels = {0: 'L0 (non-preserved)', 1: 'L1 (preserved)',
          2: 'L2 (preserved)', 3: 'L3 (preserved)'}

for l in range(J):
    ls = '--' if l in nonpreserved_levels else '-'
    for ax, mean, std in [(ax1, eps_mean, eps_std), (ax2, vel_mean, vel_std)]:
        ax.plot(steps, mean[l], color=colors[l], lw=2, ls=ls, label=labels[l])
        ax.fill_between(steps, mean[l] - std[l], mean[l] + std[l],
                        color=colors[l], alpha=0.15)

for ax in (ax1, ax2):
    ax.axhline(0, color='gray', ls=':', lw=1)
    ax.set_xlabel('Denoising step  (0 = noise  →  50 = clean)')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)

ax1.set_ylabel('Phase collinearity  |cos Δφ|')
ax1.set_title(f'|cos(∠z_t − ∠ε_struct)|\n(does z_t stay collinear with structured noise phase?)')
ax2.set_title(f'|cos(∠v_θ − ∠z_t)|\n(does velocity stay collinear with input phase?)')

fig.suptitle(f'ψ-PD Phase Self-Consistency  (dropll J=4 r=12, N={N_SAMPLES} vKITTI)',
             fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig(OUT_FIG, dpi=150, bbox_inches='tight')
print(f"\nSaved: {OUT_FIG}")

# ── Save raw data ────────────────────────────────────────────────────────────
np.savez("figures/phase_trajectory_data.npz",
         steps=np.array(steps),
         **{f"eps_mean_L{l}": eps_mean[l] for l in range(J)},
         **{f"eps_std_L{l}":  eps_std[l]  for l in range(J)},
         **{f"vel_mean_L{l}": vel_mean[l] for l in range(J)},
         **{f"vel_std_L{l}":  vel_std[l]  for l in range(J)},
         **{f"eps_all_L{l}":  np.array(all_cos_eps[l]) for l in range(J)},
         **{f"vel_all_L{l}":  np.array(all_cos_vel[l]) for l in range(J)})
print("Saved: figures/phase_trajectory_data.npz")
