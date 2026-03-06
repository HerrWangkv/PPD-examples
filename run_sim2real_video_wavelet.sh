#!/bin/bash

docker build -t wpd .

docker run -it --rm --gpus all \
    -v "$(pwd):/workspace" \
    -v "/mrtstorage/users/kwang/my_sunny_videos/:/workspace/data/my_sunny_videos" \
    -e HF_TOKEN=$HUGGING_FACE_TOKEN \
    wpd bash -c "
        cd /workspace && \
        CUDA_VISIBLE_DEVICES=6 \
        PYTHONPATH=. python sim2real_video_wavelet.py \
        --rgb_video scene_0006.mp4 \
        --flux_lora flux.safetensors \
        --flux_cutoff_radius 30 \
        --flux_maximal_radius 30 \
        --flux_gamma 1 \
        --wan_low_lora wan_low.safetensors \
        --wan_high_lora wan_high.safetensors \
        --wan_cutoff_radius 10 \
        --wan_maximal_radius 30 \
        --wan_gamma 10 \
        --debug
    "