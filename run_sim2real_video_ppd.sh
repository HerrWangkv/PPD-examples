#!/bin/bash

docker build -t wpd .

docker run -it --rm --gpus all \
    -v "$(pwd):/workspace" \
    -e HF_TOKEN=$HUGGING_FACE_TOKEN \
    wpd bash -c "
        cd /workspace && \
        CUDA_VISIBLE_DEVICES=0 \
        PYTHONPATH=. python sim2real_video_ppd.py \
        --rgb_video Town01_Rep0_ControlLoss_0.mp4 \
        --flux_lora models/ppd/flux1-dev_phipd_lora_302000.safetensors \
        --flux_cutoff_radius 30 \
        --wan_low_lora models/ppd/wan2.2-14b-low-step-12400.safetensors \
        --wan_high_lora models/ppd/wan2.2-14b-high-step-12400.safetensors \
        --wan_cutoff_radius 30
    "