#!/bin/bash
# FLUX.1-Kontext-dev sim2real translation of Virtual KITTI → real
# Requires ~23GB for flux1-kontext-dev.safetensors (downloaded on first run to ./models/)
# 4 GPUs splitting 2126 clone frames via RANK/WORLD_SIZE

INPUT_DIR=/mrtstorage/datasets_tmp/vkitti/neutral_flat
OUTPUT_DIR=/mrtstorage/users/kwang/vkitti_translated/kontext

# Pre-download weights with a single container to avoid parallel download races
echo "=== Pre-downloading FLUX.1-Kontext-dev weights ==="
docker run --rm --gpus '"device=0"' \
    -v "$(pwd):/workspace" -v "/mrtstorage:/mrtstorage" \
    -e HF_TOKEN=$HUGGING_FACE_TOKEN \
    wpd bash -c "cd /workspace && \
        PYTHONPATH=. python -c \"
from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig
FluxImagePipeline.from_pretrained(
    torch_dtype='bfloat16', device='cpu',
    model_configs=[
        ModelConfig(model_id='black-forest-labs/FLUX.1-Kontext-dev', origin_file_pattern='flux1-kontext-dev.safetensors'),
        ModelConfig(model_id='black-forest-labs/FLUX.1-dev', origin_file_pattern='text_encoder/model.safetensors'),
        ModelConfig(model_id='black-forest-labs/FLUX.1-dev', origin_file_pattern='text_encoder_2/'),
        ModelConfig(model_id='black-forest-labs/FLUX.1-dev', origin_file_pattern='ae.safetensors'),
    ],
)
print('Download complete.')
\""
echo "=== Weights ready, launching 4-GPU inference ==="

for GPU in 0 1 2 3; do
    docker run --rm --name vkitti_kontext_gpu${GPU} --gpus "\"device=${GPU}\"" \
        -v "$(pwd):/workspace" -v "/mrtstorage:/mrtstorage" \
        -e HF_TOKEN=$HUGGING_FACE_TOKEN \
        -e RANK=${GPU} -e WORLD_SIZE=4 -e LOCAL_RANK=0 \
        wpd bash -c "cd /workspace && \
            PYTHONPATH=. python batch_sim2real_image_kontext.py \
                --input_dir ${INPUT_DIR} \
                --output_dir ${OUTPUT_DIR} \
                --height 384 --width 1280 --clone_only" &
done

wait
echo "=== vKITTI Kontext done ==="
