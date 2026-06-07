#!/bin/bash
# LightEMMA evaluation on nuCarla translated videos using gemini-2.5-flash.
# Usage: bash run_lightemma_eval.sh <videos_dir> <exp_name>
# Example: bash run_lightemma_eval.sh /mrtstorage/users/kwang/nucarla_videos_wavelet/dropll_r30_J5 dropll_r30_J5

VIDEOS_DIR="${1:-/mrtstorage/users/kwang/nucarla_videos_wavelet/dropll_r30_J5}"
EXP_NAME="${2:-dropll_r30_J5}"
DATASET_DIR="/mrtstorage/users/kwang/nucarla_dataset"
OUTPUT_DIR="$(pwd)/LightEMMA/output"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_FILE="$SCRIPT_DIR/LightEMMA/${EXP_NAME}.log"
cd "$SCRIPT_DIR/LightEMMA"

echo "Logging to $LOG_FILE"
python predict_and_eval_carla.py \
    --model gemini-2.5-flash \
    --dataset_dir "$DATASET_DIR" \
    --videos_dir "$VIDEOS_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --exp_name "$EXP_NAME" 2>&1 >> "$LOG_FILE"

echo "=== Done. Results in $OUTPUT_DIR/$EXP_NAME ==="
