#!/bin/bash

docker build -t wpd .

docker run -it --rm --name train_wpd --gpus all --ipc=host \
    -v "$(pwd):/workspace" \
    -v /mrtstorage/users/kwang/openscene_front_10hz/front_camera_10hz:/workspace/data/openscene \
    -e HF_TOKEN=$HUGGING_FACE_TOKEN \
    -e NCCL_P2P_DISABLE=1 \
    wpd bash -c "
        cd /workspace && \
        PYTHONPATH=. bash examples/flux/model_training/lora/WPD-FLUX.1-dev_openscene.sh"