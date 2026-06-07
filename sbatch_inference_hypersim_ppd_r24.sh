#!/bin/bash
#SBATCH --job-name=hypersim_ppd_r24
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00
#SBATCH --partition=dev_accelerated
#SBATCH --output=logs/hypersim_ppd_r24_%j.out
#SBATCH --error=logs/hypersim_ppd_r24_%j.err
#SBATCH --signal=B:USR1@300

# Hypersim sim-to-real translation: PPD r=24, 4-GPU torchrun, with resume support.
# Input:  /hkfs/work/workspace/scratch/xw2723-generation/hypersim   (7402 frames)
# Output: /hkfs/work/workspace/scratch/xw2723-generation/hypersim_wpd/ppd_r24
#
# Usage: sbatch sbatch_inference_hypersim_ppd_r24.sh
# For accelerated (48h): sed 's/00:10:00/48:00:00/;s/dev_accelerated/accelerated/' sbatch_inference_hypersim_ppd_r24.sh | sbatch
# NOTE: mkdir -p logs  before submitting

# If ACCEL_JOB is set and already running, yield to it and exit cleanly.
if [ -n "${ACCEL_JOB}" ] && squeue -j "${ACCEL_JOB}" --noheader -o "%T" 2>/dev/null | grep -q "RUNNING"; then
    echo "Accelerated job ${ACCEL_JOB} is RUNNING — skipping dev chain slot."
    exit 0
fi

WORKSPACE=/hkfs/home/project/hk-project-p0023969/xw2723/test/PPD-examples
SIF=${WORKSPACE}/wpd.sif
DATA=/hkfs/work/workspace/scratch/xw2723-generation

PROMPT="A photorealistic indoor scene, natural and artificial lighting, real photograph, high resolution, realistic textures and materials."
NEG="ugly, low quality, CG, render, unreal, game, cartoon, blur, low res, lens artifacts"

PPD_LORA=models/ppd/flux1-dev_phipd_lora_302000.safetensors

chain_resubmit() {
    echo "USR1 received — approaching time limit. Resubmitting next chain slot."
    if [ -n "${ACCEL_JOB}" ]; then
        if squeue -j "${ACCEL_JOB}" --noheader -o "%T" 2>/dev/null | grep -q "RUNNING"; then
            echo "Accelerated job ${ACCEL_JOB} is now RUNNING — chain complete, not resubmitting."
            return
        fi
    fi
    NEXT=$(sbatch --parsable \
           --time=01:00:00 \
           --export=ALL,ACCEL_JOB=${ACCEL_JOB},HUGGING_FACE_TOKEN=${HUGGING_FACE_TOKEN} \
           "${WORKSPACE}/sbatch_inference_hypersim_ppd_r24.sh")
    echo "Resubmitted next chain slot: job ${NEXT}"
}
trap 'chain_resubmit' USR1

srun --label --export=ALL \
    apptainer exec --nv --writable-tmpfs \
    --bind ${WORKSPACE}:/workspace \
    --bind ${DATA}:/data \
    --env HF_TOKEN="${HUGGING_FACE_TOKEN}" \
    ${SIF} bash -c "
        cd /workspace
        CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 batch_sim2real_image_ppd.py \
            --input_dir /data/hypersim \
            --output_dir /data/hypersim_wpd/ppd_r24 \
            --flux_lora ${PPD_LORA} \
            --flux_cutoff_radius 24 \
            --prompt '${PROMPT}' --negative_prompt '${NEG}'
        echo 'Hypersim PPD r=24 done.'
    " &

SRUN_PID=$!
while kill -0 $SRUN_PID 2>/dev/null; do
    wait $SRUN_PID 2>/dev/null || true
done
