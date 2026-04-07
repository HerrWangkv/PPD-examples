#!/bin/bash

docker build -t wpd .

docker run -it --rm --gpus all \
    -v "$(pwd):/workspace" \
    -e HF_TOKEN=$HUGGING_FACE_TOKEN \
    wpd bash -c "
        cd /workspace && \
        CUDA_VISIBLE_DEVICES=7 \
        PYTHONPATH=. python examples/flux/model_inference/FLUX.1-dev_wavelet.py \
        --lora_checkpoint_path flux.safetensors \
        --radius 20 \
        --input_image models/ppd/test1.jpg \
        --prompt \"$(cat models/ppd/test1.txt)\" \
        --output outputs/wavelet.png"