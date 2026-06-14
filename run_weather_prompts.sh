#!/bin/bash
# Run weather/illumination controllability experiment inside Docker
# Usage: bash run_weather_prompts.sh [--gpu N]
set -e

GPU=0
while [[ $# -gt 0 ]]; do
    case $1 in --gpu) GPU=$2; shift 2;; *) shift;; esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

docker run --rm --gpus "\"device=$GPU\"" \
    -v "$SCRIPT_DIR":/workspace \
    -v /mrtstorage:/mrtstorage \
    -w /workspace \
    ppd:latest \
    bash -c "PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python exp_weather_prompts.py 2>&1 | tee logs/weather_prompts.log"

echo "Done. Figure saved to figures/weather_prompts.png"
