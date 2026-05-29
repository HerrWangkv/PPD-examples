#!/bin/bash
#SBATCH --job-name=inference_vkitti_radius
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00
#SBATCH --partition=dev_accelerated
#SBATCH --output=logs/inference_vkitti_radius_%j.out
#SBATCH --error=logs/inference_vkitti_radius_%j.err

# Radius sweep for drop_ll J=4 on vKITTI (one per GPU), with resume support.
# Variants:
#   GPU 0 — dropll_J4_r8  (step-6000, r=8,  drop_ll, J=4)
#   GPU 1 — dropll_J4_r12 (step-6000, r=12, drop_ll, J=4)
#   GPU 2 — dropll_J4_r20 (step-6000, r=20, drop_ll, J=4)
#   GPU 3 — dropll_J4_r24 (step-6000, r=24, drop_ll, J=4)
#
# Usage: sbatch sbatch_inference_vkitti_radius.sh
# For accelerated (48h): sed 's/00:10:00/48:00:00/;s/dev_accelerated/accelerated/' sbatch_inference_vkitti_radius.sh | sbatch
# NOTE: mkdir -p logs  before submitting

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

        # GPU 0: drop_ll J=4 r=8
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python batch_sim2real_image_wavelet.py \
            --input_dir /data/vkitti/clone_flat \
            --output_dir /data/vkitti_wpd/dropll_J4_r8 \
            --flux_lora ${CKPT6000} \
            --flux_cutoff_radius 8 \
            --flux_drop_ll --flux_J 4 \
            --height 384 --width 1280 \
            --prompt '${PROMPT}' --negative_prompt '${NEG}' \
            > /workspace/logs/vkitti_J4_r8_\${SLURM_JOB_ID}.out 2>&1 &

        # GPU 1: drop_ll J=4 r=12
        CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python batch_sim2real_image_wavelet.py \
            --input_dir /data/vkitti/clone_flat \
            --output_dir /data/vkitti_wpd/dropll_J4_r12 \
            --flux_lora ${CKPT6000} \
            --flux_cutoff_radius 12 \
            --flux_drop_ll --flux_J 4 \
            --height 384 --width 1280 \
            --prompt '${PROMPT}' --negative_prompt '${NEG}' \
            > /workspace/logs/vkitti_J4_r12_\${SLURM_JOB_ID}.out 2>&1 &

        # GPU 2: drop_ll J=4 r=20
        CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python batch_sim2real_image_wavelet.py \
            --input_dir /data/vkitti/clone_flat \
            --output_dir /data/vkitti_wpd/dropll_J4_r20 \
            --flux_lora ${CKPT6000} \
            --flux_cutoff_radius 20 \
            --flux_drop_ll --flux_J 4 \
            --height 384 --width 1280 \
            --prompt '${PROMPT}' --negative_prompt '${NEG}' \
            > /workspace/logs/vkitti_J4_r20_\${SLURM_JOB_ID}.out 2>&1 &

        # GPU 3: drop_ll J=4 r=24
        CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python batch_sim2real_image_wavelet.py \
            --input_dir /data/vkitti/clone_flat \
            --output_dir /data/vkitti_wpd/dropll_J4_r24 \
            --flux_lora ${CKPT6000} \
            --flux_cutoff_radius 24 \
            --flux_drop_ll --flux_J 4 \
            --height 384 --width 1280 \
            --prompt '${PROMPT}' --negative_prompt '${NEG}' \
            > /workspace/logs/vkitti_J4_r24_\${SLURM_JOB_ID}.out 2>&1 &

        wait
        echo 'All 4 radius variants done.'
    "
