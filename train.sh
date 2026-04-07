#!/bin/bash

docker build -t wpd .

docker run -it --rm --name train_wpd --gpus all --ipc=host \
    -v "$(pwd):/workspace" \
    -e HF_TOKEN=$HUGGING_FACE_TOKEN \
    -e NCCL_P2P_DISABLE=1 \
    wpd bash -c "
        cd /workspace && \
        PYTHONPATH=. bash examples/flux/model_training/lora/WPD-FLUX.1-dev.sh"