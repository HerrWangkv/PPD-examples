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
        --input_image /workspace/models/ppd/test2.jpg \
        --prompt \"\$(cat /workspace/models/ppd/test2.txt)\" \
        --output output.png --radius 30 && \
        CUDA_VISIBLE_DEVICES=7 \
        PYTHONPATH=. python examples/wanvideo/model_inference/Wan2.2-I2V-A14B_ppd.py \
        --lora_low_checkpoint_path /workspace/models/ppd/wan2.2-14b-low-step-12400.safetensors \
        --lora_high_checkpoint_path /workspace/models/ppd/wan2.2-14b-high-step-12400.safetensors \
        --input_image output.png \
        --input_video /workspace/models/ppd/test2.mp4 \
        --prompt \"\$(cat /workspace/models/ppd/test2.txt)\" \
        --radius 30 --output output.mp4"