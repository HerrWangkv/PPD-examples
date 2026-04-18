# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Phase-Preserving Diffusion (PPD)** — research implementation for structure-aligned image/video generation. Two complementary approaches:
- **PPD**: Learned phase-preserving via LoRA + `structured-noise` library (`generate_structured_noise_batch_vectorized`)
- **WPD (Wavelet Phase-Preserving Diffusion)**: Wavelet-based structured noise (`wavelet_noise.py` → `generate_wavelet_structured_noise_batch_vectorized`)

Paper: "NeuralRemaster: Phase-Preserving Diffusion for Structure-Aligned Generation" (Zeng et al., 2025)

## Setup

```bash
pip install -r requirements.txt
pip install git+https://github.com/zengxianyu/structured-noise
```

Download PPD model weights from HuggingFace (`zengxianyu/ppd`) into `models/ppd/`.

All scripts must be run with `PYTHONPATH=.` from repo root.

## Common Commands

### Inference

```bash
# SD1.5 image
PYTHONPATH=. python examples/image_synthesis/sd_text_to_image_ppd.py --input_image dog.jpg --radius 15 --prompt "..." --output output.png

# FLUX image
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python examples/flux/model_inference/FLUX.1-dev_ppd.py --input_image test.jpg --prompt "..." --output output.png --radius 30

# Wan2.2 video
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python examples/wanvideo/model_inference/Wan2.2-I2V-A14B_ppd.py --input_image output.png --input_video test.mp4 --prompt "..." --radius 30 --output output.mp4
```

### Sim-to-real video pipelines

```bash
# PPD pipeline (Flux re-render → Wan video, sliding window)
PYTHONPATH=. python sim2real_video_ppd.py --rgb_video input.mp4 --output_video output_ppd.mp4

# WPD wavelet pipeline
PYTHONPATH=. python sim2real_video_wavelet.py --rgb_video input.mp4 --output_video output_wpd.mp4

# Batch processing (distributed)
bash run_batch_sim2real_video_ppd.sh
bash run_batch_sim2real_video_wavelet.sh
```

### Training

```bash
# FLUX LoRA with WPD (8 GPUs via accelerate)
PYTHONPATH=. bash examples/flux/model_training/lora/WPD-FLUX.1-dev.sh

# Wan video training
bash train_wan_low.sh   # low-noise Wan model
bash train_wan_high.sh  # high-noise Wan model

# Docker-based FLUX training
bash train.sh
```

### Evaluation (Synthia dataset)

```bash
bash eval_synthia.sh            # mIoU, AS, depth metrics
bash eval_synthia_all.sh        # batch evaluation

# Individual metrics
PYTHONPATH=. python calc_miou_synthia.py --pred_dir data/synthia_wavelet_20_40_100
PYTHONPATH=. python calc_fid_synthia.py
PYTHONPATH=. python calc_as_synthia.py
PYTHONPATH=. python calc_depth_metrics_synthia.py
```

### Generation baselines

```bash
bash generate_synthia_wavelet.sh      # WPD (8-GPU distributed torchrun)
bash generate_synthia_baseline.sh     # baseline (4 GPUs)
bash generate_synthia_controlnet.sh
bash generate_synthia_sdedit.sh
```

## Architecture

### Core libraries

- **`diffsynth/`** — local fork of [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio). Contains all pipeline/model/trainer code. Key submodules: `pipelines/`, `models/`, `trainers/`, `lora/`, `schedulers/`.
- **`structured_noise`** (external pip package) — FFT-based phase-preserving noise for PPD
- **`wavelet_noise.py`** — local DTCWT-based wavelet noise for WPD (uses `pytorch_wavelets`)

### Pipelines

- `diffsynth/pipelines/flux_image_new.py` — FLUX inference with PPD/WPD noise injection
- `diffsynth/pipelines/wan_video_new.py` — Wan2.2 I2V with structured noise
- `sim2real_video_ppd.py` — two-stage sim-to-real pipeline: (1) FLUX re-renders first frame, (2) Wan generates video with sliding window (stride = window_size - 1)
- `sim2real_video_wavelet.py` — same pipeline but WPD wavelet noise instead of PPD LoRA

### Training

- `examples/flux/model_training/train.py` — unified training script using `accelerate` + `diffsynth` trainers
- LoRA targets: DiT attention/MLP layers (`a_to_qkv`, `b_to_qkv`, `ff_a/b`, `a/b_to_out`, etc.), rank 32
- Trained models saved to `models/train/`

### Key parameters

| Parameter | Typical values | Effect |
|-----------|---------------|--------|
| `--radius` / `--cutoff_radius` | 15–40 | Low-frequency cutoff; higher = more structure from input |
| `--maximal_radius` (WPD) | 20–100 | Upper frequency band |
| `--gamma` (WPD) | varies | Wavelet noise blending weight |

### Data layout

- `data/synthia/` — SYNTHIA dataset (RGB, GT segmentation, Depth)
- `data/synthia_wavelet_*/` — WPD generated results (named by radius parameters)
- `data/ImageNetR/` — ImageNetR evaluation set
- `models/ppd/` — PPD pretrained weights
- `models/train/` — fine-tuned LoRA checkpoints
- `outputs/` — inference outputs (ppd/, wavelet/, etc.)

## Docker

```bash
# Build and run (train.sh does this automatically)
docker build -t ppd-examples .
docker run --gpus all -v $(pwd):/workspace ppd-examples bash train.sh
```

The Dockerfile installs `pytorch_wavelets`, `structured-noise`, and all deps on top of `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime`.

## DinoPD (DINO-Preserving Diffusion) — Research Notes

Third experimental track (branch `dino`) replacing hand-designed noise structure (PPD/WPD) with learned DINOv2-feature preservation.

### Core idea

For each training sample (z0, t):
1. **Find a DINO-preserving noisy latent** `z_t*` by running Adam on `z_t` to minimize `dino_distance(decode(z_t), decode(z0))`, initialized on the flow path `(1-t)*z0 + t*z1`. Recover flow endpoint as `z1* = (z_t* - (1-t)*z0) / t`.
2. Inject `z1*` as the noise endpoint so structure is embedded in the training trajectory.
3. **Flow-matching MSE loss** at that timestep — standard velocity prediction.
4. **Auxiliary trajectory DINO loss**: compute `x0_hat = z_t - t * v_pred`, decode, and penalize DINO distance to z0. Gated at `sigma < dino_loss_t_threshold` (default 0.8) because x0_hat is meaningless near pure noise.

### Key files

- `dino_noise.py` — `load_dino()` (file-locked multi-GPU safe), `latent_to_dino()` (differentiable VAE→DINO), `dino_distance()` (cos on CLS+patches), `find_dino_preserving_noise()` returns `(z_t_star, final_dist)`
- `examples/flux/model_training/train_dino_pd.py` — training module + loop
- `examples/flux/model_training/lora/DinoPD-FLUX.1-dev.sh` — launcher (8-GPU accelerate)
- `validate_dino_trajectory.py` — monkeypatches scheduler.step to log DINO distance at every denoising step; produces distance-vs-step/sigma plot

### Training runs

**v1** (`models/train/FLUX.1-dev_lora_dino_pd/`)
- **Objective**: flow-matching MSE only. `L = training_weight(t) * ||v_pred - (z1* - z0)||²`
- **Idea**: by injecting the DINO-preserving `z1*` as the flow endpoint, the model should implicitly learn to decode structured noise back toward z0's DINO features.
- **Config**: LR=1e-4, 5 epochs planned, 300 dino_opt_steps, no trajectory supervision.
- **Observed**: loss plateaued at ~0.42 after 500 steps, flat for 3000+ steps — model converged but to the wrong objective.
- **Inference result**: poor structural preservation. Validation script revealed DINO distance to input explodes from 0.04 (step 0) → 0.69 (step 45). Small per-step velocity errors compound over 50 denoising steps; CFG (cfg_scale=2.0) amplifies drift. MSE alone is blind to multi-step trajectory divergence.

**v2** (`models/train/FLUX.1-dev_lora_dino_pd_v2/`)
- **Objective**: flow MSE + **auxiliary trajectory DINO loss** on the model's predicted clean image.
  ```
  x0_hat   = z_t* - sigma * v_pred                        # predicted clean latent
  L_dino   = dino_distance(decode(x0_hat), decode(z0))    if sigma < 0.8 else 0
  L_total  = L_flow + λ_dino * L_dino
  ```
  VAE decoder params are frozen during L_dino, but gradients flow through decode→DINO→x0_hat→v_pred→LoRA, supervising the *trajectory endpoint* rather than just velocity direction.
- **Config**: LR=5e-5, 1 epoch, 300 dino_opt_steps, λ_dino=1.0, dino_loss_t_threshold=0.8.
- **Training metrics (3614 steps)**: L_flow 0.448→0.436, L_dino 0.080→0.076. Both flat. L_dino is only ~15% of total loss.
- **Validation on step-3000** (`outputs/dino_validate_v2/`): step-0 DINO dist = 0.051 → final = 0.667 (max 0.716 @ step 44). **Identical drift pattern to v1** — flat until sigma ≈ 0.8, explodes below it. The aux loss did not transfer to inference behavior.

### Why v2 failed: the training signal was trivial

At training time we set `inputs["latents"] = z_t*` which is *already DINO-preserving* by construction. Then:
```
x0_hat = z_t* - sigma * v_pred
```
Once the flow MSE fits (`v_pred ≈ z1* - z0`), this gives `x0_hat ≈ z0` **automatically**, so `L_dino(x0_hat, z0)` is near-zero at the training point regardless of what the LoRA does. The gradient teaches the model nothing about drift correction.

At inference, `z_t` comes from multi-step rollout and drifts *off* the z_t* manifold. The model never saw off-manifold inputs during training and has no learned response to pull them back. This explains the validation plot: DINO distance stays flat where the loss was gated off (sigma > 0.8) **and** explodes where it was supposedly active (sigma < 0.8), because the loss was vacuously satisfied either way.

**Root cause**: train-inference distribution shift in the DiT input `z_t`. Training saw only perfectly-aligned `z_t*`.

### Why z1\* must stay in training (both v1 and v2)

The DiT never sees a DINO target as input at inference. Structure enters only through `z_t`. Training with plain Gaussian noise would create a train/inference mismatch — the model would never learn what structured noise "looks like" or how to decode it. `z1*` is the carrier that embeds input structure into the training distribution.

**v3** (`models/train/FLUX.1-dev_lora_dino_pd_v3/`) — *rollout-aware rectified flow matching, no L_dino*. Inspired by Self Forcing / Diffusion Forcing. Trains on a 1-step Euler-rolled-out latent instead of the clean z_t*, using a rectified velocity target that self-corrects drift.

```
scheduler.set_timesteps(num_train_timesteps=1000, training=True)
i = randint(0, N-1)                                  # timestep_id_start  (more noisy)
j = randint(i+1, N)                                  # timestep_id_target (less noisy)
sigma_i, sigma_j = scheduler.sigmas[i], scheduler.sigmas[j]

z_t_star, _ = find_dino_preserving_noise(z0, t=sigma_i, ...)   # grad enabled, then detached

with no_grad:                                        # 1-step Euler jump over (sigma_i → sigma_j)
    v_start  = model_fn(latents=z_t_star, timestep=timestep_i, ...)
    z_target = z_t_star + (sigma_j - sigma_i) * v_start

# Rectified target via the noise reparameterisation
# (algebraically identical to (z_target - z0) / sigma_j):
z1_target            = (z_target - (1 - sigma_j) * z0) / sigma_j
inputs["noise"]      = z1_target
inputs["latents"]    = z_target
training_target      = scheduler.training_target(z0, z1_target, timestep_j)
v_pred               = model_fn(latents=z_target, timestep=timestep_j, ...)
loss                 = MSE(v_pred, training_target) * scheduler.training_weight(timestep_j)
```

- No L_dino, no VAE/DINO forward in the hot loop — rectified target implies `x0_hat → z0`, which implies DINO match.
- Detached rollout → compute ≈ 2× per sample, memory ≈ baseline.
- `dino_opt_steps=300` retained from v2.

**Training metrics (717 steps, run 20260417_142627)**: loss 6.9 → 3.5 over first ~30%, then plateau ~3.5–3.8 (median 2.8, p90 ~7–9, max 127). Loss scale dominated by extreme-drift samples where the random sigma gap is large. `dino/distance_start` ≈ 0.047 flat (just confirms noise opt converges). Unlike v2's flat loss, training is moving.

**Validation trajectory (final DINO distance to z0 on `models/ppd/test1.jpg`):**

| Run | step 1000 | step 2000 | step 3000 |
|---|---|---|---|
| v2 | 0.752 | 0.684 | 0.667 |
| **v3** | 0.778 | **0.554** | — |

- Both runs show the same qualitative shape: DINO distance flat until sigma ≈ 0.85, explodes through mid-trajectory (peak ≈ step 43–45), small tail recovery below sigma 0.2.
- v2 *does* improve with training — previous "checkpoints are all equally bad" read was wrong; step1k→3k drops 0.085.
- **v3 improves ~3× faster per step**: 0.22 drop in 1k steps (step1k→2k), vs v2's 0.07 in the same interval. v3 step-2000 already beats v2 step-3000 by 0.11.
- v3 step-1000 is *worse* than v2 step-1000 — the extreme-drift outliers (loss max 127) dominate the gradient early and slow initial convergence. Once the model stabilizes past ~1.5k steps, v3's signal-per-step advantage takes over.
- v3 also shows a small tail recovery (0.645 → 0.554) in the final few steps that v2 doesn't; the rectified target is doing late-stage correction but can't prevent the mid-sigma explosion.

**Read**: v3 is the better direction, just slower to take off. Worth training further (3–5k steps) before judging. If the mid-sigma explosion persists at 5k steps, add L_dino back on top of v3's drifted input (v4a).
