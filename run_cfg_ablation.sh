#!/bin/bash
# CFG/embed_guidance 2x2 ablation + v4 baseline re-measurement.
# Run INSIDE docker container (bash debug.sh first).
# 5 validations, sequential on GPU 0. ~15-25 min total.
set -e
mkdir -p outputs/ablation_logs

V4_CKPT="models/train/FLUX.1-dev_lora_dino_pd_v4/step-1000.safetensors"
V6_CKPT="models/train/FLUX.1-dev_lora_dino_pd_v6/step-300.safetensors"

run_val() {
  local name=$1 ckpt=$2 cfg=$3 emb=$4
  echo "=== $name (cfg=$cfg emb=$emb) ==="
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python validate_dino_trajectory.py \
    --lora_checkpoint "$ckpt" \
    --input_image models/ppd/test1.jpg \
    --prompt "$(cat models/ppd/test1.txt)" \
    --cfg_scale "$cfg" \
    --embedded_guidance "$emb" \
    --output_dir "outputs/ablation_$name" \
    2>&1 | tee "outputs/ablation_logs/${name}.log"
}

run_val A1_v6_cfg1_e1    "$V6_CKPT" 1.0 1.0
run_val A2_v6_cfg1_e35   "$V6_CKPT" 1.0 3.5
run_val A3_v6_cfg2_e1    "$V6_CKPT" 2.0 1.0
run_val A4_v6_cfg2_e35   "$V6_CKPT" 2.0 3.5

echo "=== DONE ==="
grep -H "Final   DINO dist\|Max:" outputs/ablation_logs/*.log
