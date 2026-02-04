#!/bin/bash
#SBATCH --job-name=flux_openscene
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --partition=accelerated
#SBATCH --output=logs/train_flux_openscene_%j.out
#SBATCH --error=logs/train_flux_openscene_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

srun --label --export=ALL --ntasks-per-node=1 --gres=gpu:4 \
    apptainer exec --nv --writable-tmpfs \
    --bind /home/hk-project-p0023969/xw2723/test/PPD-examples:/workspace \
    --bind /hkfs/work/workspace/scratch/xw2723-generation/front_camera_10hz:/workspace/data/openscene \
    --env HF_TOKEN=$HUGGING_FACE_TOKEN \
    wpd.sif bash -c "
        cd /workspace && \
        PYTHONPATH=. \
        torchrun --nproc_per_node=4 --master_port=29500 examples/flux/model_training/train_flux_openscene.py \
        --height 704 \
        --width 1280 \
        --dataset_path "./data/openscene" \
        --dataset_repeat 1 \
        --model_id_with_origin_paths \"black-forest-labs/FLUX.1-dev:flux1-dev.safetensors,black-forest-labs/FLUX.1-dev:text_encoder/model.safetensors,black-forest-labs/FLUX.1-dev:text_encoder_2/,black-forest-labs/FLUX.1-dev:ae.safetensors\" \
        --learning_rate 1e-6 \
        --num_epochs 5 \
        --remove_prefix_in_ckpt \"pipe.dit.\" \
        --output_path \"./models/train/FLUX.1-dev_lora_wpd_openscene\" \
        --lora_base_model \"dit\" \
        --lora_target_modules \"a_to_qkv,b_to_qkv,ff_a.0,ff_a.2,ff_b.0,ff_b.2,a_to_out,b_to_out,proj_out,norm.linear,norm1_a.linear,norm1_b.linear,to_qkv_mlp\" \
        --lora_rank 32 \
        --lora_checkpoint \"./step-15000.safetensors\" \
        --align_to_opensource_format \
        --save_steps 1000 \
        --use_gradient_checkpointing"