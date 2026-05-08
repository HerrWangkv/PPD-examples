#!/bin/bash

docker build -t wpd .

docker run --rm --gpus all \
    -v "$(pwd):/workspace" \
    -v "/mrtstorage/datasets_tmp/surreal:/workspace/data/surreal" \
    -v "/mrtstorage/users/kwang/surreal_wavelet:/workspace/data/surreal_wavelet" \
    -e HF_TOKEN=$HUGGING_FACE_TOKEN \
    -e NCCL_P2P_DISABLE=1 \
    wpd bash -c "
        cd /workspace && \
        PYTHONPATH=. torchrun --nproc_per_node=8 batch_sim2real_image_wavelet.py \
        --input_dir data/surreal \
        --output_dir data/surreal_wavelet \
        --flux_lora models/train/FLUX.1-dev_lora_wpd/step-20000.safetensors \
        --flux_cutoff_radius 20 \
        --prompt \"A real photograph of a person, natural skin texture, realistic clothing and fabric, photographic lighting and shadows, candid photography, high resolution.\" \
        --negative_prompt \"CGI, 3D render, synthetic body, smooth plastic skin, flat CG lighting, mannequin, artificial, unrealistic\"
    "
