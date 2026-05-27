#!/bin/bash
#SBATCH --job-name=inference_vkitti_wpd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00
#SBATCH --partition=dev_accelerated
#SBATCH --output=logs/inference_vkitti_wpd_%j.out
#SBATCH --error=logs/inference_vkitti_wpd_%j.err

# Runs 4 vKITTI WPD variants in parallel (one per GPU), with resume support.
# Variants:
#   GPU 0 — baseline         (flux.safetensors, r=16, no drop_ll)
#   GPU 1 — dropll_step6000_J3 (step-6000, r=16, drop_ll, J=3)
#   GPU 2 — dropll_step6000_J4 (step-6000, r=16, drop_ll, J=4)
#   GPU 3 — dropll_step6000_J5 (step-6000, r=16, drop_ll, J=5)
#
# Usage: sbatch sbatch_inference_vkitti_wavelet.sh
# NOTE:  mkdir -p logs  before submitting

WORKSPACE=/hkfs/home/project/hk-project-p0023969/xw2723/test/PPD-examples
SIF=${WORKSPACE}/wpd.sif
DATA=/hkfs/work/workspace/scratch/xw2723-generation

PROMPT="A photorealistic photograph taken from a forward-facing vehicle-mounted camera. Natural outdoor lighting, authentic surface textures, real-world colors."
NEG="ugly, low quality, CG, render, unreal, game, cartoon, blur, low res, dashboard, steering wheel, windshield frame, car interior, lens artifacts"

CKPT6000=models/train/FLUX.1-dev_lora_wpd_dropll/step-6000.safetensors

srun --label --export=ALL \
    apptainer exec --nv --writable-tmpfs \
    --bind ${WORKSPACE}:/workspace \
    --bind ${DATA}:/data \
    --env HF_TOKEN="${HUGGING_FACE_TOKEN}" \
    ${SIF} bash -c "
        cd /workspace

        # GPU 0: baseline (flux.safetensors, r=16, no drop_ll)
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python batch_sim2real_image_wavelet.py \
            --input_dir /data/vkitti/neutral_flat \
            --output_dir /data/vkitti_wpd/baseline_newprompt \
            --flux_lora flux.safetensors \
            --flux_cutoff_radius 16 \
            --height 384 --width 1280 \
            --prompt '${PROMPT}' --negative_prompt '${NEG}' \
            > /workspace/logs/vkitti_baseline_\${SLURM_JOB_ID}.out 2>&1 &

        # GPU 1: drop_ll J=3
        CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python batch_sim2real_image_wavelet.py \
            --input_dir /data/vkitti/neutral_flat \
            --output_dir /data/vkitti_wpd/dropll_step6000_J3 \
            --flux_lora ${CKPT6000} \
            --flux_cutoff_radius 16 \
            --flux_drop_ll --flux_J 3 \
            --height 384 --width 1280 \
            --prompt '${PROMPT}' --negative_prompt '${NEG}' \
            > /workspace/logs/vkitti_dropll_J3_\${SLURM_JOB_ID}.out 2>&1 &

        # GPU 2: drop_ll J=4
        CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python batch_sim2real_image_wavelet.py \
            --input_dir /data/vkitti/neutral_flat \
            --output_dir /data/vkitti_wpd/dropll_step6000_J4 \
            --flux_lora ${CKPT6000} \
            --flux_cutoff_radius 16 \
            --flux_drop_ll --flux_J 4 \
            --height 384 --width 1280 \
            --prompt '${PROMPT}' --negative_prompt '${NEG}' \
            > /workspace/logs/vkitti_dropll_J4_\${SLURM_JOB_ID}.out 2>&1 &

        # GPU 3: drop_ll J=5
        CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python batch_sim2real_image_wavelet.py \
            --input_dir /data/vkitti/neutral_flat \
            --output_dir /data/vkitti_wpd/dropll_step6000_J5 \
            --flux_lora ${CKPT6000} \
            --flux_cutoff_radius 16 \
            --flux_drop_ll --flux_J 5 \
            --height 384 --width 1280 \
            --prompt '${PROMPT}' --negative_prompt '${NEG}' \
            > /workspace/logs/vkitti_dropll_J5_\${SLURM_JOB_ID}.out 2>&1 &

        wait
        echo 'All 4 vKITTI variants done.'
    "
