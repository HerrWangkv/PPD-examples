#!/bin/bash

INPUT_VIDEO=/mrtstorage/users/kwang/nucarla_videos/rgb/scene_0032.mp4

docker run --rm --gpus '"device=7"' \
    -v "$(pwd):/workspace" \
    -v "/mrtstorage:/mrtstorage" \
    -e HF_TOKEN=$HUGGING_FACE_TOKEN \
    wpd bash -c "
        cd /workspace && \
        mkdir -p outputs/ckpt_scene0032 && \

        echo '=== step-6000, drop_ll J=4 r=30 ===' && \
        PYTHONPATH=. python sim2real_video_wavelet.py \
            --rgb_video $INPUT_VIDEO \
            --flux_lora models/train/FLUX.1-dev_lora_wpd_dropll/step-6000.safetensors \
            --flux_cutoff_radius 30 \
            --flux_drop_ll --flux_J 4 \
            --image-only \
            --output_video outputs/ckpt_scene0032/step6000_r30_J4.mp4 && \

        echo '=== Done. Results in outputs/ckpt_scene0032/ ==='
    "
