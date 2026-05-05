#!/bin/bash

docker build -t wpd .

docker run -it --rm --gpus all \
    -v "$(pwd):/workspace" \
    -v "/mrtstorage/users/kwang/nucarla_videos/:/workspace/data/nucarla_videos" \
    -e HF_TOKEN=$HUGGING_FACE_TOKEN \
    wpd bash -c "
        cd /workspace && \
        CUDA_VISIBLE_DEVICES=4 \
        PYTHONPATH=. python sim2real_video_wavelet.py \
        --rgb_video data/nucarla_videos/rgb/scene_0000.mp4 \
        --flux_lora flux.safetensors \
        --flux_cutoff_radius 30 \
        --wan_low_lora wan_low.safetensors \
        --wan_high_lora wan_high.safetensors \
        --wan_cutoff_radius 30
    "