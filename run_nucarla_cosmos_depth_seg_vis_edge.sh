#!/usr/bin/env bash
# Translate 60 nuCarla videos (scene_0000-0059) with Cosmos Transfer 2.5
# using depth+seg+vis+edge controls, 4-GPU torchrun.
#
# Usage:
#   export HUGGING_FACE_TOKEN=hf_...
#   bash run_nucarla_cosmos_depth_seg_vis_edge.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COSMOS_DIR="$SCRIPT_DIR/CosmosTransfer"
DATA_DIR="/mrtstorage/users/kwang/nucarla_videos/rgb"
CONFIGS_DIR="$SCRIPT_DIR/nucarla_cosmos_configs"
OUT_DIR="$SCRIPT_DIR/outputs/nucarla/cosmos_depth_seg_vis_edge"

mkdir -p "$OUT_DIR"

# Build list of specs for scenes not yet translated
REMAINING=()
for i in $(seq 0 59); do
    scene=$(printf "scene_%04d" $i)
    if [[ -f "$OUT_DIR/${scene}.mp4" ]]; then
        echo "Skip $scene (already done)"
    else
        REMAINING+=("/configs/depth_seg_vis_edge/${scene}.json")
    fi
done

if [[ ${#REMAINING[@]} -eq 0 ]]; then
    echo "All 60 scenes already done."
    exit 0
fi

echo "${#REMAINING[@]} scenes remaining."

docker run --rm --runtime=nvidia --ipc=host \
    --gpus all \
    -v "$COSMOS_DIR":/workspace \
    -v /root/.cache:/root/.cache \
    -v "$DATA_DIR":/data_input \
    -v "$CONFIGS_DIR":/configs \
    -v "$OUT_DIR":/output \
    -e HF_TOKEN="${HUGGING_FACE_TOKEN:-}" \
    -e NCCL_P2P_DISABLE=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    cosmos-transfer25 \
    bash -c "torchrun --nproc_per_node=4 examples/inference.py \
        -i ${REMAINING[*]} \
        -o /output"

echo "=== Done. Results in $OUT_DIR ==="
echo "$(ls "$OUT_DIR"/*.mp4 2>/dev/null | wc -l) mp4s saved."
