#!/bin/bash
# DNAEdit batch inference on Hypersim → /mrtstorage/users/kwang/hypersim_dnaedit/
# Usage: bash run_hypersim_dnaedit.sh [--gpus 0,1,2,3]

GPUS="0,1,2,3"
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus) GPUS=$2; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

N_GPUS=$(echo $GPUS | tr ',' '\n' | wc -l)
OUTPUT_DIR="/mrtstorage/users/kwang/hypersim_dnaedit"
mkdir -p ${OUTPUT_DIR} && chmod 777 ${OUTPUT_DIR}
echo "=== DNAEdit on Hypersim: ${N_GPUS} GPUs (${GPUS}), output → ${OUTPUT_DIR} ==="

docker build -t wpd . -q

docker run --rm --gpus "\"device=${GPUS}\"" \
    -v "$(pwd):/workspace" \
    -v "/mrtstorage/datasets_tmp/hypersim:/workspace/data/hypersim" \
    -v "${OUTPUT_DIR}:/workspace/data/hypersim_dnaedit" \
    -e HF_TOKEN=$HUGGING_FACE_TOKEN \
    -e NCCL_P2P_DISABLE=1 \
    wpd bash -c "
        cd /workspace && \
        pip install diffusers==0.30.1 -q && \
        PYTHONPATH=. torchrun --nproc_per_node=${N_GPUS} batch_sim2real_image_dnaedit.py \
        --input_dir data/hypersim \
        --output_dir data/hypersim_dnaedit \
        --src_prompt \"A synthetic rendered indoor scene, computer graphics, 3D rendering, path-traced lighting, CG textures, simulated environment.\" \
        --tar_prompt \"A photorealistic indoor scene, natural and artificial lighting, real photograph, high resolution, realistic textures and materials.\"
    "

ln -sfn ${OUTPUT_DIR} outputs/hypersim/dnaedit
echo "=== Done. $(ls ${OUTPUT_DIR}/*.jpg 2>/dev/null | wc -l) images saved ==="
