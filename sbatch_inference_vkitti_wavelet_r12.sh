#!/bin/bash
#SBATCH --job-name=inference_vkitti_wpd_r12
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00
#SBATCH --partition=dev_accelerated
#SBATCH --output=logs/inference_vkitti_wpd_r12_%j.out
#SBATCH --error=logs/inference_vkitti_wpd_r12_%j.err
#SBATCH --signal=B:USR1@300

# J=3 and J=5 at r=12 on vKITTI clone frames (2126 frames), 2 GPUs each via torchrun.
# Variants:
#   GPU 0+1 — dropll_J3_r12  (step-6000, r=12, drop_ll, J=3)
#   GPU 2+3 — dropll_J5_r12  (step-6000, r=12, drop_ll, J=5)
#
# Usage: sbatch sbatch_inference_vkitti_wavelet_r12.sh
# For accelerated (48h): sed 's/00:10:00/48:00:00/;s/dev_accelerated/accelerated/' sbatch_inference_vkitti_wavelet_r12.sh | sbatch
# NOTE: mkdir -p logs  before submitting

# If ACCEL_JOB is set and already running, yield to it and exit cleanly.
if [ -n "${ACCEL_JOB}" ] && squeue -j "${ACCEL_JOB}" --noheader -o "%T" 2>/dev/null | grep -q "RUNNING"; then
    echo "Accelerated job ${ACCEL_JOB} is RUNNING — skipping dev chain slot."
    exit 0
fi

WORKSPACE=/hkfs/home/project/hk-project-p0023969/xw2723/test/PPD-examples
SIF=${WORKSPACE}/wpd.sif
DATA=/hkfs/work/workspace/scratch/xw2723-generation

PROMPT="A photorealistic photograph taken from a forward-facing vehicle-mounted camera. Natural outdoor lighting, authentic surface textures, real-world colors."
NEG="ugly, low quality, CG, render, unreal, game, cartoon, blur, low res, dashboard, steering wheel, windshield frame, car interior, lens artifacts"

CKPT6000=models/train/FLUX.1-dev_lora_wpd_dropll/step-6000.safetensors

chain_resubmit() {
    echo "USR1 received — approaching time limit. Resubmitting next chain slot."
    if [ -n "${ACCEL_JOB}" ]; then
        if squeue -j "${ACCEL_JOB}" --noheader -o "%T" 2>/dev/null | grep -q "RUNNING"; then
            echo "Accelerated job ${ACCEL_JOB} is now RUNNING — chain complete, not resubmitting."
        else
            NEXT=$(sbatch --parsable \
                   --time=01:00:00 \
                   --export=ALL,ACCEL_JOB=${ACCEL_JOB},HUGGING_FACE_TOKEN=${HUGGING_FACE_TOKEN} \
                   "${WORKSPACE}/sbatch_inference_vkitti_wavelet_r12.sh")
            echo "Resubmitted next chain slot: job ${NEXT}"
        fi
    fi
}
trap 'chain_resubmit' USR1

srun --label --export=ALL \
    apptainer exec --nv --writable-tmpfs \
    --bind ${WORKSPACE}:/workspace \
    --bind ${DATA}:/data \
    --env HF_TOKEN="${HUGGING_FACE_TOKEN}" \
    ${SIF} bash -c "
        cd /workspace

        # GPU 0+1: drop_ll J=3 r=12
        CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
            --master_port=29500 \
            batch_sim2real_image_wavelet.py \
            --input_dir /data/vkitti/clone_flat \
            --output_dir /data/vkitti_wpd/dropll_J3_r12 \
            --flux_lora ${CKPT6000} \
            --flux_cutoff_radius 12 \
            --flux_drop_ll --flux_J 3 \
            --height 384 --width 1280 \
            --prompt '${PROMPT}' --negative_prompt '${NEG}' \
            > /workspace/logs/vkitti_J3_r12_\${SLURM_JOB_ID}.out 2>&1 &

        # GPU 2+3: drop_ll J=5 r=12
        CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 \
            --master_port=29501 \
            batch_sim2real_image_wavelet.py \
            --input_dir /data/vkitti/clone_flat \
            --output_dir /data/vkitti_wpd/dropll_J5_r12 \
            --flux_lora ${CKPT6000} \
            --flux_cutoff_radius 12 \
            --flux_drop_ll --flux_J 5 \
            --height 384 --width 1280 \
            --prompt '${PROMPT}' --negative_prompt '${NEG}' \
            > /workspace/logs/vkitti_J5_r12_\${SLURM_JOB_ID}.out 2>&1 &

        wait
        echo 'J=3 r=12 and J=5 r=12 done.'
    " &
SRUN_PID=$!

while kill -0 $SRUN_PID 2>/dev/null; do
    wait $SRUN_PID 2>/dev/null || true
done
