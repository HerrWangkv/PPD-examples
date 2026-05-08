#!/bin/bash

docker build -t wpd .

docker run -it --rm --gpus all \
    -v "$(pwd):/workspace" \
    -v "/mrtstorage/datasets_tmp/hypersim:/workspace/data/hypersim" \
    -v "/mrtstorage/users/kwang/hypersim_flowedit:/workspace/data/hypersim_flowedit" \
    -e HF_TOKEN=$HUGGING_FACE_TOKEN \
    -e NCCL_P2P_DISABLE=1 \
    wpd bash -c "
        cd /workspace && \
        pip install diffusers==0.30.1 -q && \
        PYTHONPATH=. torchrun --nproc_per_node=8 batch_sim2real_image_flowedit.py \
        --input_dir data/hypersim \
        --output_dir data/hypersim_flowedit \
        --src_prompt \"A synthetic rendered indoor scene, computer graphics, 3D rendering, artificial lighting, CG textures.\" \
        --tar_prompt \"A photorealistic indoor scene, natural and artificial lighting, real photograph, high resolution, realistic textures and materials.\"
    "
