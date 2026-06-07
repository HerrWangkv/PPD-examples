#!/bin/bash
# DNAEdit translation of all 60 nuCarla videos using 4 GPUs per inference (FSDP).
# Videos processed sequentially; model sharded across GPUs 0-3.

OUTPUT_DIR="/workspace/outputs/nucarla/dnaedit"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

docker run --rm --gpus '"device=0,1,2,3"' \
    --name "dnaedit_nucarla" \
    --ipc=host \
    -v "$SCRIPT_DIR:/workspace" \
    -v /mrtstorage:/mrtstorage \
    -e HF_TOKEN=$HUGGING_FACE_TOKEN \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e NCCL_P2P_DISABLE=1 \
    wpd-dnaedit bash -c "
        cd /workspace && torchrun --nproc_per_node=4 batch_dnaedit_nucarla.py \
            --input_dir /mrtstorage/users/kwang/nucarla_videos/rgb \
            --output_dir $OUTPUT_DIR \
            --ckpt_dir models/Wan-AI/Wan2.1-T2V-14B \
            --task t2v-14B \
            --size 1280*704 \
            --frame_num 49
    "

echo "=== DNAEdit done. Results in $OUTPUT_DIR ==="
