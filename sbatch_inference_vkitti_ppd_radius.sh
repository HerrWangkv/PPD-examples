#!/bin/bash
#SBATCH --job-name=inference_vkitti_ppd_radius
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00
#SBATCH --partition=dev_accelerated
#SBATCH --output=logs/inference_vkitti_ppd_radius_%j.out
#SBATCH --error=logs/inference_vkitti_ppd_radius_%j.err

# PPD radius sweep on vKITTI clone frames (one variant per GPU), with resume support.
# Variants:
#   GPU 0 — ppd_r8  (flux1-dev_phipd_lora_302000, r=8,  no drop_ll)
#   GPU 1 — ppd_r12 (flux1-dev_phipd_lora_302000, r=12, no drop_ll)
#   GPU 2 — ppd_r20 (flux1-dev_phipd_lora_302000, r=20, no drop_ll)
#   GPU 3 — ppd_r24 (flux1-dev_phipd_lora_302000, r=24, no drop_ll)
#
# Usage: sbatch sbatch_inference_vkitti_ppd_radius.sh
# For accelerated (48h): sed 's/00:10:00/48:00:00/;s/dev_accelerated/accelerated/' sbatch_inference_vkitti_ppd_radius.sh | sbatch
# NOTE: mkdir -p logs  before submitting

WORKSPACE=/hkfs/home/project/hk-project-p0023969/xw2723/test/PPD-examples
SIF=${WORKSPACE}/wpd.sif
DATA=/hkfs/work/workspace/scratch/xw2723-generation

PROMPT="A photorealistic photograph taken from a forward-facing vehicle-mounted camera. Natural outdoor lighting, authentic surface textures, real-world colors."
NEG="ugly, low quality, CG, render, unreal, game, cartoon, blur, low res, dashboard, steering wheel, windshield frame, car interior, lens artifacts"

PPD_LORA=models/ppd/flux1-dev_phipd_lora_302000.safetensors

srun --label --export=ALL \
    apptainer exec --nv --writable-tmpfs \
    --bind ${WORKSPACE}:/workspace \
    --bind ${DATA}:/data \
    --env HF_TOKEN="${HUGGING_FACE_TOKEN}" \
    ${SIF} bash -c "
        cd /workspace

        # GPU 0: PPD r=8
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python batch_sim2real_image_wavelet.py \
            --input_dir /data/vkitti/clone_flat \
            --output_dir /data/vkitti_wpd/ppd_r8 \
            --flux_lora ${PPD_LORA} \
            --flux_cutoff_radius 8 \
            --height 384 --width 1280 \
            --prompt '${PROMPT}' --negative_prompt '${NEG}' \
            > /workspace/logs/vkitti_ppd_r8_\${SLURM_JOB_ID}.out 2>&1 &

        # GPU 1: PPD r=12
        CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python batch_sim2real_image_wavelet.py \
            --input_dir /data/vkitti/clone_flat \
            --output_dir /data/vkitti_wpd/ppd_r12 \
            --flux_lora ${PPD_LORA} \
            --flux_cutoff_radius 12 \
            --height 384 --width 1280 \
            --prompt '${PROMPT}' --negative_prompt '${NEG}' \
            > /workspace/logs/vkitti_ppd_r12_\${SLURM_JOB_ID}.out 2>&1 &

        # GPU 2: PPD r=20
        CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python batch_sim2real_image_wavelet.py \
            --input_dir /data/vkitti/clone_flat \
            --output_dir /data/vkitti_wpd/ppd_r20 \
            --flux_lora ${PPD_LORA} \
            --flux_cutoff_radius 20 \
            --height 384 --width 1280 \
            --prompt '${PROMPT}' --negative_prompt '${NEG}' \
            > /workspace/logs/vkitti_ppd_r20_\${SLURM_JOB_ID}.out 2>&1 &

        # GPU 3: PPD r=24
        CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python batch_sim2real_image_wavelet.py \
            --input_dir /data/vkitti/clone_flat \
            --output_dir /data/vkitti_wpd/ppd_r24 \
            --flux_lora ${PPD_LORA} \
            --flux_cutoff_radius 24 \
            --height 384 --width 1280 \
            --prompt '${PROMPT}' --negative_prompt '${NEG}' \
            > /workspace/logs/vkitti_ppd_r24_\${SLURM_JOB_ID}.out 2>&1 &

        wait
        echo 'All 4 PPD radius variants done.'
    "
