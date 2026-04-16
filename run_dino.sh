#!/bin/bash

docker build -t wpd .

mkdir -p "/storage_local/kwang/.cache/torch"

docker run -it --rm --gpus all \
    -v "$(pwd):/workspace" \
    -v "/storage_local/kwang/.cache/torch:/root/.cache/torch" \
    -e HF_TOKEN=$HUGGING_FACE_TOKEN \
    wpd bash -c "
        cd /workspace && \
        CUDA_VISIBLE_DEVICES=0 \
        PYTHONPATH=. python examples/flux/model_inference/FLUX.1-dev_dino.py \
        --lora_checkpoint_path models/train/FLUX.1-dev_lora_dino_pd/step-3000.safetensors \
        --dino_opt_steps 300 \
        --input_image models/ppd/test1.jpg \
        --prompt \"$(cat models/ppd/test1.txt)\" \
        --output_name outputs/dino.png"
