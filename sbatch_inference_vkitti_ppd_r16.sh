#!/bin/bash
#SBATCH --job-name=inference_vkitti_ppd_r16
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00
#SBATCH --partition=dev_accelerated
#SBATCH --output=logs/inference_vkitti_ppd_r16_%j.out
#SBATCH --error=logs/inference_vkitti_ppd_r16_%j.err

# PPD r=16 on vKITTI clone frames (2126 frames) using 4-GPU torchrun, with resume support.
#
# Usage: sbatch sbatch_inference_vkitti_ppd_r16.sh
# For accelerated (48h): sed 's/00:10:00/48:00:00/;s/dev_accelerated/accelerated/' sbatch_inference_vkitti_ppd_r16.sh | sbatch
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
        torchrun --nproc_per_node=4 batch_sim2real_image_ppd.py \
            --input_dir /data/vkitti/clone_flat \
            --output_dir /data/vkitti_wpd/ppd_r16 \
            --flux_lora ${PPD_LORA} \
            --flux_cutoff_radius 16 \
            --height 384 --width 1280 \
            --prompt '${PROMPT}' --negative_prompt '${NEG}'
        echo 'PPD r=16 done.'
    "
