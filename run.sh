#!/bin/bash

docker build -t wpd .

docker run -it --gpus all \
    -v "$(pwd):/workspace" \
    -e HF_TOKEN=$HUGGING_FACE_TOKEN \
    wpd bash -c "
        cd /workspace && \
        CUDA_VISIBLE_DEVICES=6 \
        PYTHONPATH=. python examples/flux/model_inference/FLUX.1-dev_wavelet.py \
        --lora_checkpoint_path /workspace/models/ppd/flux1-dev_phipd_lora_302000.safetensors \
        --threshold 0.6 \
        --input_image /workspace/models/ppd/test1.jpg \
        --prompt \"\$(cat /workspace/models/ppd/test1.txt)\" \
        --output outputs/test1/wavelet/test1.png"