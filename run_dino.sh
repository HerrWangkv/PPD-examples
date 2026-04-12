#!/bin/bash

docker build -t wpd .

docker run -it --rm --gpus all \
    -v "$(pwd):/workspace" \
    -e HF_TOKEN=$HUGGING_FACE_TOKEN \
    wpd bash -c "
        cd /workspace && \
        CUDA_VISIBLE_DEVICES=0 \
        PYTHONPATH=. python examples/flux/model_inference/FLUX.1-dev_dino.py \
        --lora_checkpoint_path models/ppd/flux1-dev_lora_color_step=266000_biased.safetensors \
        --dino_opt_steps 300 \
        --input_image data/synthia/RGB/0000820.png \
        --prompt \"A photorealistic driving scene in a European city. Natural lighting, detailed asphalt road, urban buildings, trees, cars on the street. High resolution, cinematic, realistic textures, automotive photography.\" \
        --output_name outputs/dino.png"
