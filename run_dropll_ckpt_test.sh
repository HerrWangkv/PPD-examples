#!/bin/bash

PROMPT="$(cat models/ppd/test1.txt)"
CKPT="models/train/FLUX.1-dev_lora_wpd_dropll/step-1000.safetensors"

docker run --rm --gpus '"device=4"' \
    -v "$(pwd):/workspace" \
    -e HF_TOKEN=$HUGGING_FACE_TOKEN \
    wpd bash -c "
        cd /workspace && \
        mkdir -p outputs/dropll_ckpt_test && \

        echo '=== [1/3] Old ckpt, drop_ll J=5 ===' && \
        PYTHONPATH=. python examples/flux/model_inference/FLUX.1-dev_wavelet.py \
            --lora_checkpoint_path flux.safetensors \
            --input_image models/ppd/test1.jpg \
            --prompt \"$PROMPT\" \
            --radius 30 --drop_ll --J 5 \
            --output_name outputs/dropll_ckpt_test/old_ckpt_r30_J5.png && \

        echo '=== [2/3] New ckpt step-1000, drop_ll J=5 ===' && \
        PYTHONPATH=. python examples/flux/model_inference/FLUX.1-dev_wavelet.py \
            --lora_checkpoint_path $CKPT \
            --input_image models/ppd/test1.jpg \
            --prompt \"$PROMPT\" \
            --radius 30 --drop_ll --J 5 \
            --output_name outputs/dropll_ckpt_test/new_ckpt_r30_J5.png && \

        echo '=== [3/3] New ckpt step-1000, drop_ll J=4 ===' && \
        PYTHONPATH=. python examples/flux/model_inference/FLUX.1-dev_wavelet.py \
            --lora_checkpoint_path $CKPT \
            --input_image models/ppd/test1.jpg \
            --prompt \"$PROMPT\" \
            --radius 30 --drop_ll --J 4 \
            --output_name outputs/dropll_ckpt_test/new_ckpt_r30_J4.png && \

        echo '=== Done. Results in outputs/dropll_ckpt_test/ ==='
    "
