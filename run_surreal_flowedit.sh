#!/bin/bash

docker build -t wpd .

docker run --rm --gpus all \
    -v "$(pwd):/workspace" \
    -v "/mrtstorage/datasets_tmp/surreal:/workspace/data/surreal" \
    -v "/mrtstorage/users/kwang/surreal_flowedit:/workspace/data/surreal_flowedit" \
    -e HF_TOKEN=$HUGGING_FACE_TOKEN \
    -e NCCL_P2P_DISABLE=1 \
    wpd bash -c "
        cd /workspace && \
        pip install diffusers==0.30.1 -q && \
        PYTHONPATH=. torchrun --nproc_per_node=8 batch_sim2real_image_flowedit.py \
        --input_dir data/surreal \
        --output_dir data/surreal_flowedit \
        --src_prompt \"A synthetic 3D rendered human body, SMPL model, flat CG lighting, plastic-like skin, computer graphics.\" \
        --tar_prompt \"A real photograph of a person, natural skin texture, realistic clothing, photographic lighting, candid photography.\"
    "
