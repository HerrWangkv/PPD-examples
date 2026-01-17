#!/bin/bash

docker build -t ppd .

docker run -it --gpus all \
    -v "$(pwd):/workspace" \
    -e HF_TOKEN=$HUGGING_FACE_TOKEN \
    ppd bash -c "
        cd /workspace && \
        CUDA_VISIBLE_DEVICES=7 \
        PYTHONPATH=. python examples/flux/model_inference/FLUX.1-dev_ppd.py \
        --lora_checkpoint_path /workspace/models/ppd/flux1-dev_phipd_lora_302000.safetensors \
        --input_image /workspace/models/ppd/test1.jpg \
        --prompt \"\$(cat /workspace/models/ppd/test1_jungle.txt)\" \
        --output output.png --radius 30"