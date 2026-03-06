#!/bin/bash

docker build -t wpd .

docker run -it --name generate_synthia_qwen --rm --gpus all --ipc=host \
    -v "$(pwd):/workspace" \
    -e HF_TOKEN=$HUGGING_FACE_TOKEN \
    -e NCCL_P2P_DISABLE=1 \
    wpd bash -c "
        cd /workspace && \
        PYTHONPATH=. \
        torchrun --nproc_per_node=8 examples/qwen_image/model_inference/Qwen-Image-Edit_synthia.py \
        --synthia_folder /workspace/data/synthia \
        --output_folder data/synthia_qwen"