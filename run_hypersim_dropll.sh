#!/bin/bash
# Translate Hypersim with WPD drop_ll variants
# Usage: bash run_hypersim_dropll.sh [--gpus 0,1,2,3] [--J 5] [--radius 24]

GPUS="0,1,2,3"
J=5
RADIUS=24

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)   GPUS=$2;   shift 2 ;;
        --J)      J=$2;      shift 2 ;;
        --radius) RADIUS=$2; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

N_GPUS=$(echo $GPUS | tr ',' '\n' | wc -l)
VARIANT="hypersim_dropll_J${J}_r${RADIUS}"
OUTPUT_DIR="/mrtstorage/users/kwang/${VARIANT}"

mkdir -p ${OUTPUT_DIR} && chmod 777 ${OUTPUT_DIR}
echo "=== Translating Hypersim: drop_ll J=${J} r=${RADIUS} on GPUs ${GPUS} ==="

docker run --rm --gpus "\"device=${GPUS}\"" \
    -v "$(pwd):/workspace" \
    -v "/mrtstorage/datasets_tmp/hypersim:/workspace/data/hypersim" \
    -v "${OUTPUT_DIR}:/workspace/data/${VARIANT}" \
    -e HF_TOKEN=$HUGGING_FACE_TOKEN \
    -e NCCL_P2P_DISABLE=1 \
    wpd bash -c "
        cd /workspace && \
        PYTHONPATH=. torchrun --nproc_per_node=${N_GPUS} batch_sim2real_image_wavelet.py \
        --input_dir data/hypersim \
        --output_dir data/${VARIANT} \
        --flux_lora models/train/FLUX.1-dev_lora_wpd_dropll/step-6000.safetensors \
        --flux_cutoff_radius ${RADIUS} \
        --flux_drop_ll --flux_J ${J} \
        --prompt \"A photorealistic indoor scene, natural and artificial lighting, real photograph, high resolution, realistic textures and materials.\"
    " 2>&1 | tee logs/${VARIANT}.log

echo "=== Done: ${VARIANT} ==="
