#!/bin/bash

docker build -t wpd .

docker run -it --rm --gpus all \
    -v "$(pwd):/workspace" \
    -e HF_TOKEN=$HUGGING_FACE_TOKEN \
    wpd bash -c "
        cd /workspace && \
        CUDA_VISIBLE_DEVICES=7 \
        PYTHONPATH=. python sim2real_video.py \
        --rgb_video video_rgb.mp4 \
        --depth_video video_depth.mp4 \
        --flux_lora models/train/FLUX.1-dev_lora_wpd_openscene/step-7000.safetensors \
        --wan_low_lora models/ppd/wan2.2-14b-low-step-12400.safetensors \
        --wan_high_lora models/ppd/wan2.2-14b-high-step-12400.safetensors \
        --wan_maximal_radius 20
    "