#!/bin/bash

docker build -t wpd .

docker run -it --rm --gpus all \
    -v "$(pwd):/workspace" \
    -e HF_TOKEN=$HUGGING_FACE_TOKEN \
    wpd bash -c "
        cd /workspace && \
        CUDA_VISIBLE_DEVICES=2 \
        PYTHONPATH=. python examples/flux/model_inference/FLUX.1-dev_wavelet.py \
        --lora_checkpoint_path models/train/FLUX.1-dev_lora_wpd/step-20000.safetensors \
        --radius 30 \
        --input_image data/synthia/RGB/0000820.png \
        --prompt \"A photorealistic driving scene in a European city. Natural lighting, detailed asphalt road, urban buildings, trees, cars on the street. High resolution, cinematic, realistic textures, automotive photography.\" \
        --output outputs/wavelet.png"