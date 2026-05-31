#!/bin/bash
#SBATCH --job-name=vkitti_ablation_dropll
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00
#SBATCH --partition=dev_accelerated
#SBATCH --output=logs/vkitti_ablation_dropll_%j.out
#SBATCH --error=logs/vkitti_ablation_dropll_%j.err
#SBATCH --signal=B:USR1@300

# Ablation: baseline lora (flux.safetensors) + inference-time drop_ll J=4, radius sweep.
# Isolates the inference-time LL zeroing contribution (no drop_ll training).
#   GPU 0 — ablation_dropll_r8   (flux.safetensors, drop_ll J=4, r=8)
#   GPU 1 — ablation_dropll_r12  (flux.safetensors, drop_ll J=4, r=12)
#   GPU 2 — ablation_dropll_r20  (flux.safetensors, drop_ll J=4, r=20)
#   GPU 3 — ablation_dropll_r24  (flux.safetensors, drop_ll J=4, r=24)
#
# Usage: sbatch sbatch_inference_vkitti_ablation_dropll.sh
# NOTE: mkdir -p logs  before submitting

if [ -n "${ACCEL_JOB}" ] && squeue -j "${ACCEL_JOB}" --noheader -o "%T" 2>/dev/null | grep -q "RUNNING"; then
    echo "Accelerated job ${ACCEL_JOB} is RUNNING — skipping dev chain slot."
    exit 0
fi

WORKSPACE=/hkfs/home/project/hk-project-p0023969/xw2723/test/PPD-examples
SIF=${WORKSPACE}/wpd.sif
DATA=/hkfs/work/workspace/scratch/xw2723-generation

PROMPT="A photorealistic photograph taken from a forward-facing vehicle-mounted camera. Natural outdoor lighting, authentic surface textures, real-world colors."
NEG="ugly, low quality, CG, render, unreal, game, cartoon, blur, low res, dashboard, steering wheel, windshield frame, car interior, lens artifacts"

BASELINE_LORA=flux.safetensors

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
           "${WORKSPACE}/sbatch_inference_vkitti_ablation_dropll.sh")
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

        # GPU 0: baseline lora + drop_ll J=4 r=8
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python batch_sim2real_image_wavelet.py \
            --input_dir /data/vkitti/clone_flat \
            --output_dir /data/vkitti_wpd/ablation_dropll_r8 \
            --flux_lora ${BASELINE_LORA} \
            --flux_cutoff_radius 8 \
            --flux_drop_ll --flux_J 4 \
            --height 384 --width 1280 \
            --prompt '${PROMPT}' --negative_prompt '${NEG}' \
            > /workspace/logs/vkitti_ablation_dropll_r8_\${SLURM_JOB_ID}.out 2>&1 &

        # GPU 1: baseline lora + drop_ll J=4 r=12
        CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python batch_sim2real_image_wavelet.py \
            --input_dir /data/vkitti/clone_flat \
            --output_dir /data/vkitti_wpd/ablation_dropll_r12 \
            --flux_lora ${BASELINE_LORA} \
            --flux_cutoff_radius 12 \
            --flux_drop_ll --flux_J 4 \
            --height 384 --width 1280 \
            --prompt '${PROMPT}' --negative_prompt '${NEG}' \
            > /workspace/logs/vkitti_ablation_dropll_r12_\${SLURM_JOB_ID}.out 2>&1 &

        # GPU 2: baseline lora + drop_ll J=4 r=20
        CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python batch_sim2real_image_wavelet.py \
            --input_dir /data/vkitti/clone_flat \
            --output_dir /data/vkitti_wpd/ablation_dropll_r20 \
            --flux_lora ${BASELINE_LORA} \
            --flux_cutoff_radius 20 \
            --flux_drop_ll --flux_J 4 \
            --height 384 --width 1280 \
            --prompt '${PROMPT}' --negative_prompt '${NEG}' \
            > /workspace/logs/vkitti_ablation_dropll_r20_\${SLURM_JOB_ID}.out 2>&1 &

        # GPU 3: baseline lora + drop_ll J=4 r=24
        CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python batch_sim2real_image_wavelet.py \
            --input_dir /data/vkitti/clone_flat \
            --output_dir /data/vkitti_wpd/ablation_dropll_r24 \
            --flux_lora ${BASELINE_LORA} \
            --flux_cutoff_radius 24 \
            --flux_drop_ll --flux_J 4 \
            --height 384 --width 1280 \
            --prompt '${PROMPT}' --negative_prompt '${NEG}' \
            > /workspace/logs/vkitti_ablation_dropll_r24_\${SLURM_JOB_ID}.out 2>&1 &

        wait
        echo 'All 4 ablation dropll radius variants done.'
    " &
SRUN_PID=$!

while kill -0 $SRUN_PID 2>/dev/null; do
    wait $SRUN_PID 2>/dev/null || true
done
