#!/bin/bash
# DNAEdit baseline translation of Virtual KITTI → real
# 4 GPUs splitting 6378 frames via RANK/WORLD_SIZE

INPUT_DIR=/mrtstorage/datasets_tmp/vkitti/neutral_flat
OUTPUT_DIR=outputs/vkitti/dnaedit

SRC="A synthetic rendered outdoor driving scene, virtual world, computer graphics, CG textures, simulated environment."
TAR="A photorealistic photograph taken from a forward-facing vehicle-mounted camera. Natural outdoor lighting, authentic surface textures, real-world colors."

for GPU in 0 1 2 3; do
    docker run --rm --name vkitti_dnaedit_gpu${GPU} --gpus "\"device=${GPU}\"" \
        -v "$(pwd):/workspace" -v "/mrtstorage:/mrtstorage" \
        -e HF_TOKEN=$HUGGING_FACE_TOKEN \
        -e RANK=${GPU} -e WORLD_SIZE=4 -e LOCAL_RANK=0 \
        wpd bash -c "pip install 'diffusers==0.30.1' -q && cd /workspace && mkdir -p ${OUTPUT_DIR} && \
            PYTHONPATH=. python batch_sim2real_image_dnaedit.py \
                --input_dir ${INPUT_DIR} --output_dir ${OUTPUT_DIR} \
                --src_prompt '${SRC}' --tar_prompt '${TAR}' \
                --height 384 --width 1280" &
done

wait
echo "=== vKITTI DNAEdit done ==="
