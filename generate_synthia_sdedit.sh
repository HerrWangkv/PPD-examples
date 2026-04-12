#!/bin/bash

docker build -t wpd .

docker run -it --name generate_synthia_sdedit --rm --gpus all --ipc=host \
    -v "$(pwd):/workspace" \
    -e HF_TOKEN=$HUGGING_FACE_TOKEN \
    -e NCCL_P2P_DISABLE=1 \
    wpd bash -c "
        cd /workspace && \
        CUDA_VISIBLE_DEVICES=0 \
        PYTHONPATH=. \
        torchrun --nproc_per_node=1 examples/flux/model_inference/FLUX.1-dev_sdedit_synthia.py \
        --synthia_folder /workspace/data/synthia \
        --timestep 0.5 \
        --output_folder data/synthia_sdedit_0.5"