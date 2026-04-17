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
  x0_hat   = z_t - t * v_pred                # model's estimate of z0 given current z_t
  L_dino   = dino_distance(decode(x0_hat), decode(z0))  if sigma < 0.8 else 0
  L_total  = L_flow + λ_dino * L_dino
  ```
  VAE decoder params are frozen during L_dino computation, but gradients flow through decode→DINO→x0_hat→v_pred→LoRA, supervising the *trajectory endpoint* rather than just velocity direction.
- **Config**: LR=5e-5 (halved), 1 epoch, 300 dino_opt_steps, λ_dino=1.0, dino_loss_t_threshold=0.8.
- **Observed (3614 steps)**:
  - flow loss 0.448 → 0.436 (Q1→Q4), dino_x0 loss 0.080 → 0.076 — both barely moving
  - DINO term is only ~15% of total loss (0.076 vs 0.44 flow) — likely under-weighted
  - z_t* optimization stable at ~0.048 distance (sanity check on find_dino_preserving_noise)
- **Validation on step-3000 checkpoint** (`outputs/dino_validate_v2/`):
  - Step 0 DINO dist = 0.051 → Final = 0.667 (max 0.716 at step 44)
  - **Identical drift pattern to v1** — flat until sigma ≈ 0.8, explodes after that
  - The aux DINO loss did not transfer to inference behavior

### Why v2 failed: training signal was trivial

During training we set `inputs["latents"] = z_t*` (already DINO-preserving by construction) and compute:
```
x0_hat = z_t* - sigma * v_pred
```
With flow MSE well-fit (`v_pred ≈ z1* - z0`), x0_hat ≈ z0 automatically — so `L_dino(x0_hat, z0)` is near zero at the training point regardless of what the LoRA does. The gradient teaches the model nothing about drift correction.

At inference, `z_t` comes from multi-step rollout and drifts off the z_t* manifold. The model never saw off-manifold inputs during training and has no learned response to pull them back. That's why DINO distance stays flat where the training signal was inactive (sigma > 0.8) *and* explodes where it was active (sigma < 0.8) — the loss was vacuously satisfied either way.

**Root cause**: train-inference distribution shift in the DiT input `z_t`. Training sees perfectly-aligned `z_t*`, inference sees drifted z_t.

### Proposed v3 fix

Train with **deliberately-drifted inputs** so the model learns to correct rather than preserve:
```
z_t_perturbed = z_t* + ε      # ε ~ small Gaussian, or short rollout with current LoRA
x0_hat        = z_t_perturbed - sigma * v_pred(z_t_perturbed, t)
L_dino        = dino_distance(decode(x0_hat), decode(z0))
```
This gives the DINO loss real gradient: "given an off-manifold z_t, predict a velocity that brings x0_hat back to DINO-match z0."

Design knobs:
- Perturbation source: random Gaussian (cheap) vs. short rollout with current LoRA (more realistic but 2-3× slower).
- Perturbation magnitude: fixed small σ, or scheduled to match observed drift at each sigma bucket.
- Keep flow MSE on the clean z_t* path (so the "if z_t is on-manifold, stay on it" behavior is also learned).

### Why z1\* must stay in training (both v1 and v2)

The DiT never sees a DINO target as input at inference. Structure enters only through `z_t`. Training with plain Gaussian noise would create a train/inference mismatch — the model would never learn what structured noise "looks like" or how to decode it. `z1*` is the carrier that embeds input structure into the training distribution.

### Next steps to try

- **Bump `--lambda_dino`** to 3–5 so DINO contribution matches or exceeds flow MSE.
- **Run validation script** on v2 checkpoint to see if trajectory drift actually improved at inference time — training metric is not the real test.
- `--dino_opt_steps 300` is the current default (100 was tested but loss curves suggest converging well before 100 is possible; verify from optimizer printouts).
