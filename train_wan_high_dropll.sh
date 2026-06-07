#!/bin/bash

docker build -t wpd .

docker run --rm --name train_wan_high_dropll --gpus '"device=0,1,2,3"' --ipc=host \
    -v "$(pwd):/workspace" \
    -e HF_TOKEN=$HUGGING_FACE_TOKEN \
    -e NCCL_P2P_DISABLE=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    wpd bash -c "
        cd /workspace && \
        PYTHONPATH=. bash examples/wanvideo/model_training/lora/WPD_Wan2.2-I2V-A14B_high_dropll.sh"
