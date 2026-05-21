#!/bin/bash

PROMPT="$(cat models/ppd/test1.txt)"

docker run --rm --gpus all \
    -v "$(pwd):/workspace" \
    -e HF_TOKEN=$HUGGING_FACE_TOKEN \
    wpd bash -c "
        cd /workspace && \
        mkdir -p outputs/dropll_test1 && \

        echo '=== [1/5] Baseline (no drop_ll) ===' && \
        PYTHONPATH=. python examples/flux/model_inference/FLUX.1-dev_wavelet.py \
            --lora_checkpoint_path flux.safetensors \
            --input_image models/ppd/test1.jpg \
            --prompt \"$PROMPT\" \
            --radius 20 \
            --output_name outputs/dropll_test1/baseline.png && \

        echo '=== [2/5] drop_ll J=3 ===' && \
        PYTHONPATH=. python examples/flux/model_inference/FLUX.1-dev_wavelet.py \
            --lora_checkpoint_path flux.safetensors \
            --input_image models/ppd/test1.jpg \
            --prompt \"$PROMPT\" \
            --radius 20 --drop_ll --J 3 \
            --output_name outputs/dropll_test1/drop_ll_J3.png && \

        echo '=== [3/5] drop_ll J=4 ===' && \
        PYTHONPATH=. python examples/flux/model_inference/FLUX.1-dev_wavelet.py \
            --lora_checkpoint_path flux.safetensors \
            --input_image models/ppd/test1.jpg \
            --prompt \"$PROMPT\" \
            --radius 20 --drop_ll --J 4 \
            --output_name outputs/dropll_test1/drop_ll_J4.png && \

        echo '=== [4/5] drop_ll J=5 ===' && \
        PYTHONPATH=. python examples/flux/model_inference/FLUX.1-dev_wavelet.py \
            --lora_checkpoint_path flux.safetensors \
            --input_image models/ppd/test1.jpg \
            --prompt \"$PROMPT\" \
            --radius 20 --drop_ll --J 5 \
            --output_name outputs/dropll_test1/drop_ll_J5.png && \

        echo '=== [5/5] drop_ll J=6 ===' && \
        PYTHONPATH=. python examples/flux/model_inference/FLUX.1-dev_wavelet.py \
            --lora_checkpoint_path flux.safetensors \
            --input_image models/ppd/test1.jpg \
            --prompt \"$PROMPT\" \
            --radius 20 --drop_ll --J 6 \
            --output_name outputs/dropll_test1/drop_ll_J6.png && \

        echo '=== Done. Results in outputs/dropll_test1/ ==='
    "
