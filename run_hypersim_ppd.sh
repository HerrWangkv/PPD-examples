#!/bin/bash
# PPD r8 batch inference on Hypersim (FLUX-only, original PPD lora)
# Usage: bash run_hypersim_ppd.sh [--gpus 0,1,2,3] [--radius 8]

GPUS="0,1,2,3"
RADIUS=8
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)   GPUS=$2;   shift 2 ;;
        --radius) RADIUS=$2; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

N_GPUS=$(echo $GPUS | tr ',' '\n' | wc -l)
VARIANT="hypersim_ppd_r${RADIUS}"
OUTPUT_DIR="/mrtstorage/users/kwang/${VARIANT}"
mkdir -p ${OUTPUT_DIR} && chmod 777 ${OUTPUT_DIR}
echo "=== PPD r${RADIUS} on Hypersim: ${N_GPUS} GPUs (${GPUS}), output → ${OUTPUT_DIR} ==="

docker build -t wpd . -q

docker run --rm --gpus "\"device=${GPUS}\"" \
    -v "$(pwd):/workspace" \
    -v "/mrtstorage/datasets_tmp/hypersim:/workspace/data/hypersim" \
    -v "${OUTPUT_DIR}:/workspace/data/${VARIANT}" \
    -e HF_TOKEN=$HUGGING_FACE_TOKEN \
    -e NCCL_P2P_DISABLE=1 \
    wpd bash -c "
        cd /workspace && \
        PYTHONPATH=. torchrun --nproc_per_node=${N_GPUS} batch_sim2real_image_ppd.py \
        --input_dir data/hypersim \
        --output_dir data/${VARIANT} \
        --flux_lora models/ppd/flux1-dev_phipd_lora_302000.safetensors \
        --flux_cutoff_radius ${RADIUS} \
        --prompt \"A photorealistic indoor scene, natural and artificial lighting, real photograph, high resolution, realistic textures and materials.\"
    "

ln -sfn ${OUTPUT_DIR} outputs/hypersim/ppd_r${RADIUS}
echo "=== Done. $(ls ${OUTPUT_DIR}/*.jpg 2>/dev/null | wc -l) images saved ==="
