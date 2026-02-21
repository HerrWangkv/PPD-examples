#!/bin/bash
#SBATCH --job-name=wan_high
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --partition=accelerated-h100
#SBATCH --output=logs/train_wan_high_%j.out
#SBATCH --error=logs/train_wan_high_%j.err

mkdir -p logs

srun --label --export=ALL --ntasks-per-node=1 --gres=gpu:4 \
    apptainer exec --nv --writable-tmpfs \
    --bind /home/hk-project-p0023969/xw2723/test/PPD-examples:/workspace \
    --env HF_TOKEN=$HUGGING_FACE_TOKEN \
    wpd.sif bash -c "
        cd /workspace && \
        PYTHONPATH=. \
        torchrun --nproc_per_node=4 --master_port=29500 examples/wanvideo/model_training/train.py \
        --height 704 \
        --width 1280 \
        --num_frames 49 \
        --dataset_repeat 1 \
        --model_id_with_origin_paths "Wan-AI/Wan2.2-I2V-A14B:high_noise_model/diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-I2V-A14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-I2V-A14B:Wan2.1_VAE.pth" \
        --learning_rate 1e-6 \
        --num_epochs 5 \
        --remove_prefix_in_ckpt "pipe.dit." \
        --output_path "./models/train/Wan2.2-I2V-A14B_high_noise_lora_wpd" \
        --lora_base_model "dit" \
        --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
        --lora_rank 64 \
        --lora_checkpoint "models/ppd/wan2.2-14b-high-step-12400.safetensors" \
        --extra_inputs "input_image" \
        --max_timestep_boundary 0.358 \
        --min_timestep_boundary 0 \
        --save_steps 100 \
        --use_gradient_checkpointing"