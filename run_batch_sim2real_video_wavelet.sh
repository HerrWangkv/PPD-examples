#!/bin/bash

# Default Arguments
INPUT_DATASET="data/my_sunny_videos"
OUTPUT_DIR="outputs/wavelet/flux_20_40_10_wan_20_40_10"

# Override if provided
if [ ! -z "$1" ]; then
    INPUT_DATASET=$1
fi
if [ ! -z "$2" ]; then
    OUTPUT_DIR=$2
fi

docker build -t wpd .

docker run -it --rm --gpus all \
    -v "$(pwd):/workspace" \
    -v /mrtstorage/users/kwang/my_sunny_videos:/workspace/data/my_sunny_videos \
    -e HF_TOKEN=$HUGGING_FACE_TOKEN \
    wpd bash -c "
        cd /workspace && \
        CUDA_VISIBLE_DEVICES=6 \
        PYTHONPATH=. python batch_sim2real_video_wavelet.py \
        --input_dataset '$INPUT_DATASET' \
        --output_dir '$OUTPUT_DIR' \
        --flux_lora models/train/FLUX.1-dev_lora_wpd/step-15000.safetensors \
        --flux_cutoff_radius 20 \
        --flux_maximal_radius 40 \
        --flux_gamma 10 \
        --wan_low_lora wan_low_wpd_400.safetensors \
        --wan_high_lora wan_high_wpd_600.safetensors \
        --wan_cutoff_radius 20 \
        --wan_maximal_radius 40 \
        --wan_gamma 10
    "
