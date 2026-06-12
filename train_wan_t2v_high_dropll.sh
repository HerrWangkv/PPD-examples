#!/bin/bash

REPO_DIR="$(pwd)"

CUDA_VISIBLE_DEVICES=0,1,2,3 apptainer exec --nv --writable-tmpfs \
    --bind "$REPO_DIR:/workspace" \
    --env HF_TOKEN=$HUGGING_FACE_TOKEN \
    --env NCCL_P2P_DISABLE=1 \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$REPO_DIR/wpd.sif" bash -c "
        cd /workspace && \
        PYTHONPATH=. bash examples/wanvideo/model_training/lora/WPD_Wan2.2-T2V-A14B_high_dropll.sh"
