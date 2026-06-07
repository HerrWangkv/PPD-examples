#!/bin/bash
#SBATCH --job-name=dnaedit_nucarla
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=96:00:00
#SBATCH --partition=accelerated-h100
#SBATCH --output=logs/dnaedit_nucarla_%j.out
#SBATCH --error=logs/dnaedit_nucarla_%j.err

mkdir -p logs

# Paths — adjust REPO_DIR and DATA_DIR to your HPC layout
REPO_DIR=/home/hk-project-p0023969/xw2723/test/PPD-examples
DATA_DIR=/home/hk-project-p0023969/xw2723/nucarla_videos/rgb
OUTPUT_DIR=/workspace/outputs/nucarla/dnaedit

srun --label --export=ALL --ntasks-per-node=1 --gres=gpu:4 \
    apptainer exec --nv --writable-tmpfs \
    --bind "$REPO_DIR:/workspace" \
    --bind "$(dirname $DATA_DIR):$(dirname $DATA_DIR)" \
    --env NCCL_P2P_DISABLE=1 \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$REPO_DIR/wpd-dnaedit.sif" bash -c "
        cd /workspace && \
        torchrun --nproc_per_node=4 --master_port=29500 batch_dnaedit_nucarla.py \
            --input_dir $DATA_DIR \
            --output_dir $OUTPUT_DIR \
            --ckpt_dir models/Wan-AI/Wan2.1-T2V-14B \
            --task t2v-14B \
            --size 1280*704 \
            --frame_num 49 \
            --guide_scale 1.0 \
            --tgt_guide_scale 5.0 \
            --jmp 12 \
            --seed 42
    "

echo "=== DNAEdit done. Results in $REPO_DIR/outputs/nucarla/dnaedit ==="
