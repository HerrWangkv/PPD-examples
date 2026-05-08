#!/bin/bash

docker build -t wpd .

docker run -it --rm --gpus all \
    -v "$(pwd):/workspace" \
    -v "/mrtstorage/datasets_tmp/hypersim:/workspace/data/hypersim" \
    -v "/mrtstorage/users/kwang/hypersim_wavelet:/workspace/data/hypersim_wavelet" \
    -e HF_TOKEN=$HUGGING_FACE_TOKEN \
    -e NCCL_P2P_DISABLE=1 \
    wpd bash -c "
        cd /workspace && \
        PYTHONPATH=. torchrun --nproc_per_node=8 batch_sim2real_image_wavelet.py \
        --input_dir data/hypersim \
        --output_dir data/hypersim_wavelet \
        --flux_lora models/train/FLUX.1-dev_lora_wpd/step-20000.safetensors \
        --flux_cutoff_radius 20 \
        --prompt \"A photorealistic indoor scene, natural and artificial lighting, real photograph, high resolution, realistic textures and materials.\"
    "
