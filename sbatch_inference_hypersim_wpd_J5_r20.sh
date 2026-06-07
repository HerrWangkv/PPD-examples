#!/bin/bash
#SBATCH --job-name=hypersim_wpd_J5_r20
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00
#SBATCH --partition=dev_accelerated
#SBATCH --output=logs/hypersim_wpd_J5_r20_%j.out
#SBATCH --error=logs/hypersim_wpd_J5_r20_%j.err
#SBATCH --signal=B:USR1@300

# Hypersim sim-to-real: WPD J=5 r=20 (step-6000, drop_ll), 4-GPU torchrun, resume support.
# Input:  /hkfs/work/workspace/scratch/xw2723-generation/hypersim   (7402 frames)
# Output: /hkfs/work/workspace/scratch/xw2723-generation/hypersim_wpd/dropll_J5_r20

if [ -n "${ACCEL_JOB}" ] && squeue -j "${ACCEL_JOB}" --noheader -o "%T" 2>/dev/null | grep -q "RUNNING"; then
    echo "Accelerated job ${ACCEL_JOB} is RUNNING — skipping dev chain slot."
    exit 0
fi

WORKSPACE=/hkfs/home/project/hk-project-p0023969/xw2723/test/PPD-examples
SIF=${WORKSPACE}/wpd.sif
DATA=/hkfs/work/workspace/scratch/xw2723-generation

PROMPT="A photorealistic indoor scene, natural and artificial lighting, real photograph, high resolution, realistic textures and materials."
NEG="ugly, low quality, CG, render, unreal, game, cartoon, blur, low res, lens artifacts"

CKPT6000=models/train/FLUX.1-dev_lora_wpd_dropll/step-6000.safetensors

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
           "${WORKSPACE}/sbatch_inference_hypersim_wpd_J5_r20.sh")
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
        CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 \
            batch_sim2real_image_wavelet.py \
            --input_dir /data/hypersim \
            --output_dir /data/hypersim_wpd/dropll_J5_r20 \
            --flux_lora ${CKPT6000} \
            --flux_cutoff_radius 20 \
            --flux_drop_ll --flux_J 5 \
            --prompt '${PROMPT}' --negative_prompt '${NEG}' \
            > /workspace/logs/hypersim_wpd_J5_r20_\${SLURM_JOB_ID}.out 2>&1
    " &
SRUN_PID=$!

while kill -0 $SRUN_PID 2>/dev/null; do
    wait $SRUN_PID 2>/dev/null || true
done
