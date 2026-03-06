#!/bin/bash

# Usage: ./eval_synthia.sh <gen_folder>
docker build -t wpd .

docker run -it --rm --gpus all --ipc=host \
    --device /dev/fuse \
    --cap-add SYS_ADMIN \
    --security-opt apparmor:unconfined \
    -v "$(pwd):/workspace" \
    -v /mrtstorage/users/kwang/synthia_sim2real/:/workspace/data/synthia_sim2real \
    -e HF_TOKEN=$HUGGING_FACE_TOKEN \
    -e NCCL_P2P_DISABLE=1 \
    -w /workspace \
    wpd bash -c "
        CUDA_VISIBLE_DEVICES=4 python calc_miou_synthia.py --gen_folder \"$1\" && \
        CUDA_VISIBLE_DEVICES=4 python calc_as_synthia.py --gen_folder \"$1\" && \
        CUDA_VISIBLE_DEVICES=4 python calc_depth_metrics_synthia.py --gen_folder \"$1\"
    "