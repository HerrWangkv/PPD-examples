#!/bin/bash
#SBATCH --job-name=inference_vkitti_baseline_radius
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00
#SBATCH --partition=dev_accelerated
#SBATCH --output=logs/inference_vkitti_baseline_radius_%j.out
#SBATCH --error=logs/inference_vkitti_baseline_radius_%j.err
#SBATCH --signal=B:USR1@300

# Baseline (flux.safetensors, no drop_ll) radius sweep on vKITTI clone frames, with resume support.
# Variants:
#   GPU 0 — baseline_r8  (flux.safetensors, r=8,  no drop_ll)
#   GPU 1 — baseline_r12 (flux.safetensors, r=12, no drop_ll)
#   GPU 2 — baseline_r20 (flux.safetensors, r=20, no drop_ll)
#   GPU 3 — baseline_r24 (flux.safetensors, r=24, no drop_ll)
#
# Usage: sbatch sbatch_inference_vkitti_baseline_radius.sh
# For accelerated (48h): sed 's/00:10:00/48:00:00/;s/dev_accelerated/accelerated/' sbatch_inference_vkitti_baseline_radius.sh | sbatch
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

# Trap USR1 (sent 5 min before time limit) to resubmit next chain slot before dying.
chain_resubmit() {
    echo "USR1 received — approaching time limit. Resubmitting next chain slot."
    if [ -n "${ACCEL_JOB}" ]; then
        if squeue -j "${ACCEL_JOB}" --noheader -o "%T" 2>/dev/null | grep -q "RUNNING"; then
            echo "Accelerated job ${ACCEL_JOB} is now RUNNING — chain complete, not resubmitting."
        else
            NEXT=$(sbatch --parsable \
                   --time=01:00:00 \
                   --export=ALL,ACCEL_JOB=${ACCEL_JOB},HUGGING_FACE_TOKEN=${HUGGING_FACE_TOKEN} \
                   "${WORKSPACE}/sbatch_inference_vkitti_baseline_radius.sh")
            echo "Resubmitted next chain slot: job ${NEXT}"
        fi
    fi
}
trap 'chain_resubmit' USR1

# Run srun in background so USR1 can interrupt the wait() call below
srun --label --export=ALL \
    apptainer exec --nv --writable-tmpfs \
    --bind ${WORKSPACE}:/workspace \
    --bind ${DATA}:/data \
    --env HF_TOKEN="${HUGGING_FACE_TOKEN}" \
    ${SIF} bash -c "
        cd /workspace

        # GPU 0: baseline r=8
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python batch_sim2real_image_wavelet.py \
            --input_dir /data/vkitti/clone_flat \
            --output_dir /data/vkitti_wpd/baseline_r8 \
            --flux_lora flux.safetensors \
            --flux_cutoff_radius 8 \
            --height 384 --width 1280 \
            --prompt '${PROMPT}' --negative_prompt '${NEG}' \
            > /workspace/logs/vkitti_baseline_r8_\${SLURM_JOB_ID}.out 2>&1 &

        # GPU 1: baseline r=12
        CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python batch_sim2real_image_wavelet.py \
            --input_dir /data/vkitti/clone_flat \
            --output_dir /data/vkitti_wpd/baseline_r12 \
            --flux_lora flux.safetensors \
            --flux_cutoff_radius 12 \
            --height 384 --width 1280 \
            --prompt '${PROMPT}' --negative_prompt '${NEG}' \
            > /workspace/logs/vkitti_baseline_r12_\${SLURM_JOB_ID}.out 2>&1 &

        # GPU 2: baseline r=20
        CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python batch_sim2real_image_wavelet.py \
            --input_dir /data/vkitti/clone_flat \
            --output_dir /data/vkitti_wpd/baseline_r20 \
            --flux_lora flux.safetensors \
            --flux_cutoff_radius 20 \
            --height 384 --width 1280 \
            --prompt '${PROMPT}' --negative_prompt '${NEG}' \
            > /workspace/logs/vkitti_baseline_r20_\${SLURM_JOB_ID}.out 2>&1 &

        # GPU 3: baseline r=24
        CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python batch_sim2real_image_wavelet.py \
            --input_dir /data/vkitti/clone_flat \
            --output_dir /data/vkitti_wpd/baseline_r24 \
            --flux_lora flux.safetensors \
            --flux_cutoff_radius 24 \
            --height 384 --width 1280 \
            --prompt '${PROMPT}' --negative_prompt '${NEG}' \
            > /workspace/logs/vkitti_baseline_r24_\${SLURM_JOB_ID}.out 2>&1 &

        wait
        echo 'All 4 baseline radius variants done.'
    " &
SRUN_PID=$!

# Loop so that USR1 can interrupt wait(), run chain_resubmit, then we resume waiting
while kill -0 $SRUN_PID 2>/dev/null; do
    wait $SRUN_PID 2>/dev/null || true
done
