#!/bin/bash

INPUT_VIDEO=/mrtstorage/users/kwang/nucarla_videos/rgb/scene_0032.mp4

docker run -it --rm --gpus all \
    -v "$(pwd):/workspace" \
    -v "/mrtstorage:/mrtstorage" \
    -e HF_TOKEN=$HUGGING_FACE_TOKEN \
    wpd bash -c "
        cd /workspace && \
        mkdir -p outputs/wpd_midfreq_scene0032 && \

        echo '=== [1/4] Mid-freq WPD (min_radius=2, radius=30) ===' && \
        PYTHONPATH=. python sim2real_video_wavelet.py \
            --rgb_video $INPUT_VIDEO \
            --flux_lora flux.safetensors \
            --flux_cutoff_radius 30 \
            --flux_min_radius 2 \
            --image-only \
            --output_video outputs/wpd_midfreq_scene0032/min_r2.mp4 && \

        echo '=== [2/4] Mid-freq WPD (min_radius=3, radius=30) ===' && \
        PYTHONPATH=. python sim2real_video_wavelet.py \
            --rgb_video $INPUT_VIDEO \
            --flux_lora flux.safetensors \
            --flux_cutoff_radius 30 \
            --flux_min_radius 3 \
            --image-only \
            --output_video outputs/wpd_midfreq_scene0032/min_r3.mp4 && \

        echo '=== [3/4] Mid-freq WPD (min_radius=4, radius=30) ===' && \
        PYTHONPATH=. python sim2real_video_wavelet.py \
            --rgb_video $INPUT_VIDEO \
            --flux_lora flux.safetensors \
            --flux_cutoff_radius 30 \
            --flux_min_radius 4 \
            --image-only \
            --output_video outputs/wpd_midfreq_scene0032/min_r4.mp4 && \

        echo '=== [4/4] Mid-freq WPD (min_radius=5, radius=30) ===' && \
        PYTHONPATH=. python sim2real_video_wavelet.py \
            --rgb_video $INPUT_VIDEO \
            --flux_lora flux.safetensors \
            --flux_cutoff_radius 30 \
            --flux_min_radius 5 \
            --image-only \
            --output_video outputs/wpd_midfreq_scene0032/min_r5.mp4 && \

        echo '=== Done. Results in outputs/wpd_midfreq_scene0032/ ==='
    "
