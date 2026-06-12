#!/bin/bash
# I2V translation: dropll r30 J5, new seed 123, all 160 nuCarla scenes, GPUs 0-3

REPO_DIR="$(pwd)"

CUDA_VISIBLE_DEVICES=0,1,2,3 apptainer exec --nv --writable-tmpfs \
    --bind "$REPO_DIR:/workspace" \
    --bind /mrtstorage:/mrtstorage \
    --bind /tmp:/tmp \
    --env HF_TOKEN=$HUGGING_FACE_TOKEN \
    --env NCCL_P2P_DISABLE=1 \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env HF_HOME=/tmp/hf_cache \
    "$REPO_DIR/wpd.sif" bash -c "
        cd /workspace && PYTHONPATH=. torchrun --nproc_per_node=4 batch_sim2real_video_wavelet.py \
            --input_dataset /tmp/nucarla_all_rgb \
            --output_dir outputs/nucarla/dropll_r16_J5 \
            --flux_lora models/train/FLUX.1-dev_lora_wpd_dropll/step-6000.safetensors \
            --flux_cutoff_radius 16 --flux_drop_ll --flux_J 5 \
            --wan_high_lora models/train/Wan2.2-I2V-A14B_high_lora_wpd_dropll/step-200.safetensors \
            --wan_low_lora models/train/Wan2.2-I2V-A14B_low_lora_wpd_dropll/step-400.safetensors \
            --wan_cutoff_radius 16 --wan_drop_ll --wan_J 5
    "
