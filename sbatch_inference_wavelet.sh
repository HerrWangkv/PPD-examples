#!/bin/bash
#SBATCH --job-name=inference_wpd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --partition=dev_accelerated
#SBATCH --output=logs/inference_wpd_%j.out
#SBATCH --error=logs/inference_wpd_%j.err

# NOTE: You must run 'mkdir -p logs' manually in your terminal before sbatch!

# Check if an argument was provided
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
    echo "Error: No argument provided. Usage: sbatch sbatch_inference_wavelet.sh <cutoff_radius> <maximal_radius> <gamma>"
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
        torchrun --nproc_per_node=4 examples/flux/model_inference/FLUX.1-dev_wavelet_synthia.py \
        --lora_checkpoint_path /workspace/models/train/FLUX.1-dev_lora_wpd/step-15000.safetensors \
        --synthia_folder /workspace/data/synthia \
        --cutoff_radius \"$1\" \
        --maximal_radius \"$2\" \
        --gamma \"$3\" \
        --output_folder /workspace/data/synthia_wavelet_${1}_${2}_${3}"