#!/bin/bash

docker build -t wpd .

docker run -it --rm --gpus all --ipc=host \
    --device /dev/fuse \
    --cap-add SYS_ADMIN \
    --security-opt apparmor:unconfined \
    -v "$(pwd):/workspace" \
    -v /mrtstorage/datasets/public/cityscapes.sqfs:/data/cityscapes.sqfs \
    -v /mrtstorage/users/kwang/synthia_sim2real/:/workspace/data/synthia_sim2real \
    -e HF_TOKEN=$HUGGING_FACE_TOKEN \
    -e NCCL_P2P_DISABLE=1 \
    -w /workspace \
    wpd \
    /bin/bash -c "mkdir -p /workspace/data/cityscapes && squashfuse /data/cityscapes.sqfs /workspace/data/cityscapes && echo 'Cityscapes mounted at /workspace/data/cityscapes' && exec /bin/bash"