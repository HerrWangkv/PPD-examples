#!/bin/bash
# B: rerun v4 step-1000 under patch-only metric (cfg=2.0, embed=3.5 matches current inference default).
# Run INSIDE docker container. Pinned to GPU 1 via CUDA_VISIBLE_DEVICES.
set -e
mkdir -p outputs/ablation_logs
PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 python validate_dino_trajectory.py \
  --lora_checkpoint models/train/FLUX.1-dev_lora_dino_pd_v4/step-1000.safetensors \
  --input_image models/ppd/test1.jpg \
  --prompt "$(cat models/ppd/test1.txt)" \
  --cfg_scale 2.0 \
  --embedded_guidance 3.5 \
  --output_dir outputs/ablation_B_v4_base \
  2>&1 | tee outputs/ablation_logs/B_v4_base.log
