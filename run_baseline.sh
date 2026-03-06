#!/bin/bash

docker build -t wpd .

docker run -it --rm --gpus all \
    -v "$(pwd):/workspace" \
    -e HF_TOKEN=$HUGGING_FACE_TOKEN \
    wpd bash -c "
        cd /workspace && \
        CUDA_VISIBLE_DEVICES=3 \
        PYTHONPATH=. python examples/flux/model_inference/FLUX.1-dev_ppd.py \
        --lora_checkpoint_path /workspace/models/ppd/flux1-dev_phipd_lora_302000.safetensors \
        --input_image models/ppd/test1.jpg \
        --prompt \"$(cat models/ppd/test1.txt)\" \
        --output outputs/ppd.png --radius 30"