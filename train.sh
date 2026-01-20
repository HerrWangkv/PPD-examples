#!/bin/bash

docker build -t wpd .

docker run -it --rm --name train_wpd --gpus all \
    -v "$(pwd):/workspace" \
    -e HF_TOKEN=$HUGGING_FACE_TOKEN \
    wpd bash -c "
        cd /workspace && \
        CUDA_VISIBLE_DEVICES=3 \
        PYTHONPATH=. bash examples/flux/model_training/lora/WPD-FLUX.1-dev.sh"