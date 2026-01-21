#!/bin/bash

docker build -t wpd .

docker run -it --rm --gpus all \
    -v "$(pwd):/workspace" \
    -e HF_TOKEN=$HUGGING_FACE_TOKEN \
    wpd bash -c "
        cd /workspace && \
        CUDA_VISIBLE_DEVICES=6 \
        PYTHONPATH=. python examples/flux/model_inference/FLUX.1-dev_wavelet.py \
        --lora_checkpoint_path /workspace/models/ppd/flux1-dev_phipd_lora_302000.safetensors \
        --J 7 \
        --input_image /workspace/data/synthia/RGB/0000000.png \
        --prompt \"\$(cat /workspace/data/synthia/RGB/0000000.txt)\" \
        --output outputs/synthia/0000000/wavelet/7.png"