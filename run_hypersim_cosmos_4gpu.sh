#!/bin/bash
# Launch Cosmos Transfer on Hypersim across 4 GPUs.
# Specs must already be sharded: /tmp/cosmos_hypersim/specs/depth_edge_gpu{0..3}/
# Usage: bash run_hypersim_cosmos_4gpu.sh

set -euo pipefail

OUT=/storage_local/kwang/repos/PPD-examples/outputs/hypersim/cosmos_depth_edge
COSMOS=/storage_local/kwang/repos/PPD-examples/CosmosTransfer
SPECS=/tmp/cosmos_hypersim/specs

mkdir -p "$OUT"

for gpu in 0 1 2 3; do
    docker run --rm \
        --runtime=nvidia \
        --ipc=host \
        -e CUDA_VISIBLE_DEVICES="$gpu" \
        -e HF_TOKEN="${HUGGING_FACE_TOKEN:-}" \
        -e CUDA_NAME=cu128 \
        -v "$COSMOS":/workspace \
        -v /workspace/.venv \
        -v /root/.cache:/root/.cache \
        -v "$SPECS":/staging/specs \
        -v /mrtstorage:/mrtstorage \
        -v "$OUT":/output \
        cosmos-transfer25:latest \
        bash -c "python examples/inference.py \
            -i /staging/specs/depth_edge_gpu${gpu}/*.json \
            -o /output \
            --disable-guardrails" \
        > /tmp/cosmos_hypersim_gpu${gpu}.log 2>&1 &
    echo "Launched GPU $gpu (pid $!)"
done

wait
echo "All 4 GPUs done. $(ls "$OUT"/*.jpg 2>/dev/null | wc -l) images saved."

rsync -av --progress "$OUT/" /mrtstorage/users/kwang/hypersim_cosmos_depth_edge/
