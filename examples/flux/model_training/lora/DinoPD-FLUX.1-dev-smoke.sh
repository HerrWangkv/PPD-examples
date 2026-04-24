#!/bin/bash
# Smoke test for inline validation hook. Warm-starts from v4 step-1000,
# fires validation every 20 train steps, caps dataset so the run finishes
# after ~3 validations (~30 min on 8 GPUs).
accelerate launch --multi_gpu --num_processes 8 examples/flux/model_training/train_dino_pd.py \
  --max_pixels 262144 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "black-forest-labs/FLUX.1-dev:flux1-dev.safetensors,black-forest-labs/FLUX.1-dev:text_encoder/model.safetensors,black-forest-labs/FLUX.1-dev:text_encoder_2/,black-forest-labs/FLUX.1-dev:ae.safetensors" \
  --learning_rate 5e-5 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/FLUX.1-dev_lora_dino_pd_v5_smoke" \
  --lora_base_model "dit" \
  --lora_target_modules "a_to_qkv,b_to_qkv,ff_a.0,ff_a.2,ff_b.0,ff_b.2,a_to_out,b_to_out,proj_out,norm.linear,norm1_a.linear,norm1_b.linear,to_qkv_mlp" \
  --lora_rank 32 \
  --align_to_opensource_format \
  --save_steps 20 \
  --max_samples 500 \
  --use_gradient_checkpointing \
  --dino_model_name "dinov2_vitl14_reg" \
  --dino_opt_steps 300 \
  --max_sigma_gap_min 0.05 \
  --max_sigma_gap_max 0.3 \
  --rollout_steps_min 1 \
  --rollout_steps_max 15 \
  --sigma_target_min 0.1 \
  --min_substep_sigma 0.02 \
  --val_enabled true \
  --val_dino_opt_steps 300 \
  --lora_checkpoint "./models/train/FLUX.1-dev_lora_dino_pd_v4/step-1000.safetensors"
