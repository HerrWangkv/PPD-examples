#!/bin/bash

# Usage: ./eval_synthia.sh <gen_folder>
docker build -t wpd .

docker run -it --rm --gpus all --name eval_synthia --ipc=host \
    --device /dev/fuse \
    --cap-add SYS_ADMIN \
    --security-opt apparmor:unconfined \
    -v "$(pwd):/workspace" \
    -v /mrtstorage/datasets/public/cityscapes.sqfs:/data/cityscapes.sqfs \
    -e HF_TOKEN=$HUGGING_FACE_TOKEN \
    -e NCCL_P2P_DISABLE=1 \
    -w /workspace \
    wpd bash -c "
        mkdir -p /workspace/data/cityscapes && \
        squashfuse /data/cityscapes.sqfs /workspace/data/cityscapes && \
        echo 'Cityscapes mounted at /workspace/data/cityscapes' && \
        CUDA_VISIBLE_DEVICES=1 python calc_miou_synthia.py --gen_folder \"$1\" && \
        CUDA_VISIBLE_DEVICES=1 python calc_fid_synthia.py --gen_folder \"$1\" && \
        CUDA_VISIBLE_DEVICES=1 python calc_clip_synthia.py --gen_folder \"$1\" && \
        CUDA_VISIBLE_DEVICES=1 python calc_as_synthia.py --gen_folder \"$1\"
    "