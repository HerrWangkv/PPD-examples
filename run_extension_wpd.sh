#!/bin/bash
# WPD translation of 100 extension nuCarla videos (scene_0060-0159).
# r30, J=5, drop_ll, distributed across 4 GPUs via torchrun.

FLUX_LORA="models/train/FLUX.1-dev_lora_wpd_dropll/step-6000.safetensors"
WAN_HIGH="models/train/Wan2.2-I2V-A14B_high_lora_wpd_dropll/step-200.safetensors"
WAN_LOW="models/train/Wan2.2-I2V-A14B_low_lora_wpd_dropll/step-400.safetensors"
DATA="/mrtstorage/users/kwang/nucarla_videos_extension"
OUT="/workspace/outputs/nucarla/extension_dropll_r30_J5"

docker run --rm --gpus '"device=0,1,2,3"' \
    --name "extension_wpd_r30J5" \
    -v "$(pwd):/workspace" \
    -v /mrtstorage:/mrtstorage \
    -e HF_TOKEN=$HUGGING_FACE_TOKEN \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    wpd bash -c "
        cd /workspace && PYTHONPATH=. torchrun --nproc_per_node=4 \
            batch_sim2real_video_wavelet.py \
            --input_dataset '$DATA' \
            --output_dir '$OUT' \
            --flux_lora $FLUX_LORA \
            --flux_cutoff_radius 30 --flux_drop_ll --flux_J 5 \
            --wan_low_lora $WAN_LOW \
            --wan_high_lora $WAN_HIGH \
            --wan_cutoff_radius 30 --wan_drop_ll --wan_J 5
    "

echo "=== Done. Results in $OUT ==="
