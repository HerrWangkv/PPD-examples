#!/bin/bash

docker build -t wpd .

docker run -it --name generate_synthia_wavelet --rm --gpus all --ipc=host \
    -v "$(pwd):/workspace" \
    -e HF_TOKEN=$HUGGING_FACE_TOKEN \
    -e NCCL_P2P_DISABLE=1 \
    wpd bash -c "
        cd /workspace && \
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
        PYTHONPATH=. \
        torchrun --nproc_per_node=8 examples/flux/model_inference/FLUX.1-dev_wavelet_synthia.py \
        --lora_checkpoint_path /workspace/models/train/FLUX.1-dev_lora_wpd/step-15000.safetensors \
        --synthia_folder /workspace/data/synthia \
        --cutoff_radius 20 \
        --maximal_radius 40 \
        --gamma 1"