#!/bin/bash

# Default Arguments
INPUT_DATASET="data/nucarla_videos"
OUTPUT_DIR="outputs/wavelet/flux_30_30_1_wan_30_30_1"

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
    -v /mrtstorage/users/kwang/nucarla_videos:/workspace/data/nucarla_videos \
    -e HF_TOKEN=$HUGGING_FACE_TOKEN \
    wpd bash -c "
        cd /workspace && \
        CUDA_VISIBLE_DEVICES=5 \
        PYTHONPATH=. python batch_sim2real_video_wavelet.py \
        --input_dataset '$INPUT_DATASET' \
        --output_dir '$OUTPUT_DIR' \
        --flux_lora models/train/FLUX.1-dev_lora_wpd/step-15000.safetensors \
        --flux_cutoff_radius 30 \
        --flux_maximal_radius 30 \
        --flux_gamma 1 \
        --wan_low_lora wan_low_wpd.safetensors \
        --wan_high_lora wan_high_wpd.safetensors \
        --wan_cutoff_radius 30 \
        --wan_maximal_radius 30 \
        --wan_gamma 1
    "
