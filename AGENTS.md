## Project Overview

This is the **Phase-Preserving Diffusion (PPD) / Wavelet Phase Diffusion (WPD)** research codebase. It adapts SD1.5, FLUX.1-dev, and Wan2.2-14b with structured wavelet noise for sim-to-real image/video translation. The main research contribution is DTCWT-based structured noise injection (`wavelet_noise.py`) that preserves phase structure during diffusion.

The `diffsynth/` package is a fork of [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio). PPD-modified pipelines live in `diffsynth/pipelines/flux_image_new.py` and `diffsynth/pipelines/wan_video_new.py`.

## Environment

All training and inference runs must go through Docker — GPU commands run as root inside the container:

```bash
# Build image and open interactive shell with datasets mounted
bash debug.sh

# Inside Docker, always set PYTHONPATH=. before running scripts
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python examples/flux/model_inference/FLUX.1-dev_ppd.py ...
```

For host-side Python work (data scripts, eval, etc.), activate the venv first:
```bash
source .venv/bin/activate
# install with: uv pip install <pkg>
```

## Key Scripts

**Inference (single image/video):**
```bash
# FLUX PPD image
PYTHONPATH=. python examples/flux/model_inference/FLUX.1-dev_ppd.py \
  --input_image <img> --radius 30 --prompt "..." --output output.png

# WPD two-stage (Flux image → Wan video)
PYTHONPATH=. python sim2real_video_wavelet.py \
  --rgb_video <video> --flux_lora <lora> --wan_low_lora <lora> --wan_high_lora <lora>
```

**Batch inference (multi-GPU via torchrun):**
```bash
# Hypersim batch (see run_hypersim.sh for full Docker invocation)
torchrun --nproc_per_node=8 batch_sim2real_image_wavelet.py \
  --input_dir data/hypersim --output_dir data/hypersim_wavelet --flux_lora <lora>
```

**Training (FLUX LoRA):**
```bash
PYTHONPATH=. bash examples/flux/model_training/lora/PPD-FLUX.1-dev.sh
```

**Evaluation:**
```bash
PYTHONPATH=. python calc_fid_hypersim.py   # FID
PYTHONPATH=. python calc_as_hypersim.py    # Aesthetic Score
PYTHONPATH=. python calc_depth_metrics_hypersim.py  # Depth SSIM
PYTHONPATH=. python calc_miou_synthia.py   # mIoU (semantic segmentation)
```

## Model Weights

Weights live in `models/ppd/`. Key files:
- `flux1-dev_phipd_lora_302000.safetensors` — FLUX PPD LoRA
- `wan2.2-14b-high-step-12400.safetensors` / `wan2.2-14b-low-step-12400.safetensors` — Wan WPD LoRAs (high/low frequency)
- Base FLUX.1-dev weights are downloaded via `diffsynth.download_models(["FLUX.1-dev"])`

## Architecture

```
wavelet_noise.py          # Core: DTCWT decomposer, structured noise generation
diffsynth/
  pipelines/
    flux_image_new.py     # PPD-modified FLUX pipeline (ModelConfig-based API)
    wan_video_new.py      # PPD-modified Wan video pipeline
  models/                 # Model definitions (flux_dit.py, wan_video_dit.py, ...)
  trainers/               # Training loops
examples/
  flux/                   # FLUX inference + training scripts
  wanvideo/               # Wan2.2 inference scripts
batch_sim2real_*.py       # Multi-GPU distributed inference over dataset dirs
sim2real_video_*.py       # Single-video two-stage pipelines
calc_*.py                 # Evaluation metrics (FID, AS, depth SSIM, mIoU)
Results.md                # Experiment results log (Synthia mIoU / AS / SSIM tables)
```

## Datasets

Data is mounted into Docker at runtime (see `debug.sh` / `run_hypersim.sh`):
- **Synthia**: `/workspace/data/synthia` — primary sim2real benchmark (mIoU via `eval_synthia.sh`)
- **Hypersim**: `/workspace/data/hypersim` — indoor scenes, depth metrics
- **SURREAL**: human body sim2real (scripts in `surreal/`)
- **nuCarla**: driving video dataset at `/mrtstorage/users/kwang/nucarla_videos/`
- **Cityscapes**: mounted via squashfuse from `/data/cityscapes.sqfs`

## WPD Naming Convention

Experiment names in `Results.md` follow `WPD<flux_radius>_<wan_radius>[_<scale>]`, e.g. `WPD20_40_0.5` means flux cutoff radius=20, wan cutoff radius=40, noise scale=0.5. `PPD<radius>` denotes FLUX-only (no Wan stage).
