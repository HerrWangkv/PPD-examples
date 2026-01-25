#!/bin/bash

docker build -t wpd .

docker run -it --name generate_synthia_baseline --rm --gpus all --ipc=host \
    -v "$(pwd):/workspace" \
    -e HF_TOKEN=$HUGGING_FACE_TOKEN \
    -e NCCL_P2P_DISABLE=1 \
    wpd bash -c "
        cd /workspace && \
        CUDA_VISIBLE_DEVICES=4,5,6,7 \
        PYTHONPATH=. \
        torchrun --nproc_per_node=4 examples/flux/model_inference/FLUX.1-dev_ppd_synthia.py \
        --lora_checkpoint_path /workspace/models/ppd/flux1-dev_phipd_lora_302000.safetensors \
        --synthia_folder /workspace/data/synthia \
        --radius 30"