#!/bin/bash

docker build -t wpd .

docker run -it --rm --name train_wan_high --gpus all --ipc=host \
    -v "$(pwd):/workspace" \
    -e HF_TOKEN=$HUGGING_FACE_TOKEN \
    -e NCCL_P2P_DISABLE=1 \
    wpd bash -c "
        cd /workspace && \
        CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. bash examples/wanvideo/model_training/lora/WPD_Wan2.2-I2V-A14B_high.sh"