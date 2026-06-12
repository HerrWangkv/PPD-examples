#!/usr/bin/env bash
# Translate 60 nuCarla videos with Wan2.1-VACE-14B (zero-shot, gray control).
# Runs 4 parallel containers on GPU 0-3, 15 scenes each.
#
# Usage:
#   export HUGGING_FACE_TOKEN=hf_...
#   bash run_nucarla_vace.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="/mrtstorage/users/kwang/nucarla_videos/rgb"
OUT_DIR="$SCRIPT_DIR/outputs/nucarla/vace_gray"

mkdir -p "$OUT_DIR"

DONE=$(ls "$OUT_DIR"/scene_????.mp4 2>/dev/null | wc -l)
echo "Already done: $DONE / 60"
if [[ $DONE -ge 60 ]]; then
    echo "All 60 scenes done."
    exit 0
fi

run_gpu() {
    local GPU=$1
    local START=$2
    local END=$3
    docker run --rm --gpus "device=$GPU" --ipc=host \
        -v "$SCRIPT_DIR":/workspace \
        -v "$DATA_DIR":/mrtstorage/users/kwang/nucarla_videos/rgb \
        -v /root/.cache:/root/.cache \
        -e HF_TOKEN="${HUGGING_FACE_TOKEN:-}" \
        -e HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_TOKEN:-}" \
        -e PYTHONPATH=/workspace \
        -e CUDA_VISIBLE_DEVICES=0 \
        -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        -w /workspace \
        wpd \
        python batch_sim2real_vace_nucarla.py \
            --input_dir /mrtstorage/users/kwang/nucarla_videos/rgb \
            --output_dir outputs/nucarla/vace_gray \
            --scene_start "$START" \
            --scene_end "$END" 2>&1 | tee "logs/vace_nucarla_gpu${GPU}.log"
}

# 4 GPUs × 15 scenes each
run_gpu 0  0 14 &
run_gpu 1 15 29 &
run_gpu 2 30 44 &
run_gpu 3 45 59 &
wait

echo "All 4 GPU jobs finished."
DONE=$(ls "$OUT_DIR"/scene_????.mp4 2>/dev/null | wc -l)
echo "Total done: $DONE / 60"
