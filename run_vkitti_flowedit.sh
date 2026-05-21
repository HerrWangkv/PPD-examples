#!/bin/bash
# FlowEdit baseline translation of Virtual KITTI → real
# 4 GPUs splitting 6378 frames via RANK/WORLD_SIZE (no NCCL, pure work-split)

INPUT_DIR=/mrtstorage/datasets_tmp/vkitti/neutral_flat
OUTPUT_DIR=outputs/vkitti/flowedit

SRC="A synthetic rendered outdoor driving scene, virtual world, computer graphics, CG textures, simulated environment."
TAR="A photorealistic photograph taken from a forward-facing vehicle-mounted camera. Natural outdoor lighting, authentic surface textures, real-world colors."

docker run --rm --name vkitti_flowedit_gpu0 --gpus '"device=0"' \
    -v "$(pwd):/workspace" -v "/mrtstorage:/mrtstorage" \
    -e HF_TOKEN=$HUGGING_FACE_TOKEN \
    -e RANK=0 -e WORLD_SIZE=4 -e LOCAL_RANK=0 \
    wpd bash -c "pip install 'diffusers==0.30.3' -q && cd /workspace && mkdir -p ${OUTPUT_DIR} && \
        PYTHONPATH=. python batch_sim2real_image_flowedit.py \
            --input_dir ${INPUT_DIR} --output_dir ${OUTPUT_DIR} \
            --src_prompt '${SRC}' --tar_prompt '${TAR}' \
            --height 384 --width 1280" &

docker run --rm --name vkitti_flowedit_gpu1 --gpus '"device=1"' \
    -v "$(pwd):/workspace" -v "/mrtstorage:/mrtstorage" \
    -e HF_TOKEN=$HUGGING_FACE_TOKEN \
    -e RANK=1 -e WORLD_SIZE=4 -e LOCAL_RANK=0 \
    wpd bash -c "pip install 'diffusers==0.30.3' -q && cd /workspace && mkdir -p ${OUTPUT_DIR} && \
        PYTHONPATH=. python batch_sim2real_image_flowedit.py \
            --input_dir ${INPUT_DIR} --output_dir ${OUTPUT_DIR} \
            --src_prompt '${SRC}' --tar_prompt '${TAR}' \
            --height 384 --width 1280" &

docker run --rm --name vkitti_flowedit_gpu2 --gpus '"device=2"' \
    -v "$(pwd):/workspace" -v "/mrtstorage:/mrtstorage" \
    -e HF_TOKEN=$HUGGING_FACE_TOKEN \
    -e RANK=2 -e WORLD_SIZE=4 -e LOCAL_RANK=0 \
    wpd bash -c "pip install 'diffusers==0.30.3' -q && cd /workspace && mkdir -p ${OUTPUT_DIR} && \
        PYTHONPATH=. python batch_sim2real_image_flowedit.py \
            --input_dir ${INPUT_DIR} --output_dir ${OUTPUT_DIR} \
            --src_prompt '${SRC}' --tar_prompt '${TAR}' \
            --height 384 --width 1280" &

docker run --rm --name vkitti_flowedit_gpu3 --gpus '"device=3"' \
    -v "$(pwd):/workspace" -v "/mrtstorage:/mrtstorage" \
    -e HF_TOKEN=$HUGGING_FACE_TOKEN \
    -e RANK=3 -e WORLD_SIZE=4 -e LOCAL_RANK=0 \
    wpd bash -c "pip install 'diffusers==0.30.3' -q && cd /workspace && mkdir -p ${OUTPUT_DIR} && \
        PYTHONPATH=. python batch_sim2real_image_flowedit.py \
            --input_dir ${INPUT_DIR} --output_dir ${OUTPUT_DIR} \
            --src_prompt '${SRC}' --tar_prompt '${TAR}' \
            --height 384 --width 1280" &

wait
echo "=== vKITTI FlowEdit done ==="
