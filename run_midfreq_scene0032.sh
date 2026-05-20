#!/bin/bash

INPUT_VIDEO=/mrtstorage/users/kwang/nucarla_videos/rgb/scene_0032.mp4

docker run --rm --gpus all \
    -v "$(pwd):/workspace" \
    -v "/mrtstorage:/mrtstorage" \
    -e HF_TOKEN=$HUGGING_FACE_TOKEN \
    wpd bash -c "
        cd /workspace && \
        mkdir -p outputs/wpd_midfreq_scene0032 && \

        PYTHONPATH=. python sim2real_video_wavelet.py \
            --rgb_video $INPUT_VIDEO \
            --flux_lora flux.safetensors \
            --flux_cutoff_radius 30 \
            --flux_drop_ll --flux_J 3 \
            --wan_drop_ll --wan_J 3 \
            --output_video outputs/wpd_midfreq_scene0032/drop_ll_J3_video.mp4 && \

        echo '=== Done ==='
    "
