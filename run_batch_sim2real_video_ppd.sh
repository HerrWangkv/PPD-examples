#!/bin/bash

# Default Arguments
INPUT_DATASET="my_sim2real_videos"
OUTPUT_DIR="outputs/ppd/flux_30_wan_30"

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
    -e HF_TOKEN=$HUGGING_FACE_TOKEN \
    wpd bash -c "
        cd /workspace && \
        CUDA_VISIBLE_DEVICES=0 \
        PYTHONPATH=. python batch_sim2real_video_ppd.py \
        --input_dataset '$INPUT_DATASET' \
        --output_dir '$OUTPUT_DIR' \
        --flux_lora models/ppd/flux1-dev_phipd_lora_302000.safetensors \
        --flux_cutoff_radius 30 \
        --wan_low_lora models/ppd/wan2.2-14b-low-step-12400.safetensors \
        --wan_high_lora models/ppd/wan2.2-14b-high-step-12400.safetensors \
        --wan_cutoff_radius 30
    "
