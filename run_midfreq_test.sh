#!/bin/bash

docker run -it --rm --gpus all \
    -v "$(pwd):/workspace" \
    -e HF_TOKEN=$HUGGING_FACE_TOKEN \
    wpd bash -c "
        cd /workspace && \
        mkdir -p outputs/wpd_midfreq && \

        echo '=== [1/3] Baseline WPD (min_radius=None) ===' && \
        PYTHONPATH=. python examples/flux/model_inference/FLUX.1-dev_wavelet.py \
            --lora_checkpoint_path flux.safetensors \
            --input_image models/ppd/test1.jpg \
            --prompt \"\$(cat models/ppd/test1.txt)\" \
            --radius 15 \
            --output_name outputs/wpd_midfreq/baseline.png && \

        echo '=== [2/3] Mid-freq WPD (min_radius=5) ===' && \
        PYTHONPATH=. python examples/flux/model_inference/FLUX.1-dev_wavelet.py \
            --lora_checkpoint_path flux.safetensors \
            --input_image models/ppd/test1.jpg \
            --prompt \"\$(cat models/ppd/test1.txt)\" \
            --radius 15 --min_radius 5 \
            --output_name outputs/wpd_midfreq/min_r5.png && \

        echo '=== [3/3] Mid-freq WPD (min_radius=10) ===' && \
        PYTHONPATH=. python examples/flux/model_inference/FLUX.1-dev_wavelet.py \
            --lora_checkpoint_path flux.safetensors \
            --input_image models/ppd/test1.jpg \
            --prompt \"\$(cat models/ppd/test1.txt)\" \
            --radius 15 --min_radius 10 \
            --output_name outputs/wpd_midfreq/min_r10.png && \

        echo '=== Done. Results in outputs/wpd_midfreq/ ==='
    "
