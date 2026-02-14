#!/bin/bash
#SBATCH --job-name=inference_qwen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --partition=accelerated
#SBATCH --output=logs/inference_qwen_%j.out
#SBATCH --error=logs/inference_qwen_%j.err

# NOTE: You must run 'mkdir -p logs' manually in your terminal before sbatch!

# Check if an argument was provided
if [ -z "$1" ]; then
    echo "Error: No argument provided. Usage: sbatch sbatch_inference_qwen.sh <timestep>"
    exit 1
fi

# Cleaned up srun command
# Removed redundant --gres and --ntasks flags (inherited from SBATCH)
# Added safeguards for the argument variable
srun --label --export=ALL \
    apptainer exec --nv --writable-tmpfs \
    --bind /home/hk-project-p0023969/xw2723/test/PPD-examples:/workspace \
    --bind /hkfs/work/workspace/scratch/xw2723-generation/:/workspace/data/ \
    --env HF_TOKEN="$HUGGING_FACE_TOKEN" \
    wpd.sif bash -c "
        cd /workspace && \
        PYTHONPATH=. \
        torchrun --nproc_per_node=4 examples/qwen_image/model_inference/Qwen-Image-Edit_synthia.py \
        --synthia_folder /workspace/data/synthia \
        --output_folder /workspace/data/synthia_qwen"