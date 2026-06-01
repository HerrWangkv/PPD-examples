#!/bin/bash
#SBATCH --job-name=vkitti_ppd_r32
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00
#SBATCH --partition=dev_accelerated
#SBATCH --output=logs/vkitti_ppd_r32_%j.out
#SBATCH --error=logs/vkitti_ppd_r32_%j.err
#SBATCH --signal=B:USR1@300

# PPD r=32 on vKITTI clone frames, 4-GPU torchrun, with resume support.
# Input:  /hkfs/work/workspace/scratch/xw2723-generation/vkitti/clone_flat (2126 frames)
# Output: /hkfs/work/workspace/scratch/xw2723-generation/vkitti_wpd/ppd_r32

if [ -n "${ACCEL_JOB}" ] && squeue -j "${ACCEL_JOB}" --noheader -o "%T" 2>/dev/null | grep -q "RUNNING"; then
    echo "Accelerated job ${ACCEL_JOB} is RUNNING — skipping dev chain slot."
    exit 0
fi

WORKSPACE=/hkfs/home/project/hk-project-p0023969/xw2723/test/PPD-examples
SIF=${WORKSPACE}/wpd.sif
DATA=/hkfs/work/workspace/scratch/xw2723-generation

PROMPT="A photorealistic photograph taken from a forward-facing vehicle-mounted camera. Natural outdoor lighting, authentic surface textures, real-world colors."
NEG="ugly, low quality, CG, render, unreal, game, cartoon, blur, low res, dashboard, steering wheel, windshield frame, car interior, lens artifacts"

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
           "${WORKSPACE}/sbatch_inference_vkitti_ppd_r32.sh")
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
            batch_sim2real_image_ppd.py \
            --input_dir /data/vkitti/clone_flat \
            --output_dir /data/vkitti_wpd/ppd_r32 \
            --flux_lora ${PPD_LORA} \
            --flux_cutoff_radius 32 \
            --height 384 --width 1280 \
            --prompt '${PROMPT}' --negative_prompt '${NEG}' \
            > /workspace/logs/vkitti_ppd_r32_\${SLURM_JOB_ID}.out 2>&1
    " &
SRUN_PID=$!

while kill -0 $SRUN_PID 2>/dev/null; do
    wait $SRUN_PID 2>/dev/null || true
done
