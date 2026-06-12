CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --num_processes 4 examples/wanvideo/model_training/train.py \
  --height 704 \
  --width 1280 \
  --num_frames 49 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-T2V-A14B:low_noise_model/diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-T2V-A14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-T2V-A14B:Wan2.1_VAE.pth" \
  --learning_rate 1e-5 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.2-T2V-A14B_low_lora_wpd_dropll" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 64 \
  --lora_checkpoint "models/train/Wan2.2-I2V-A14B_low_lora_wpd_dropll/step-400.safetensors" \
  --max_timestep_boundary 1 \
  --min_timestep_boundary 0.358 \
  --save_steps 100 \
  --use_gradient_checkpointing

# boundary corresponds to timesteps [0, 900)
# initialized from I2V step-12400 checkpoint (LoRA on transformer layers, fully compatible with T2V)
