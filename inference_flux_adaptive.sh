#!/bin/bash

docker build -t ppd .

docker run -it --gpus all \
    -v "$(pwd):/workspace" \
    -e HF_TOKEN=$HUGGING_FACE_TOKEN \
    ppd bash -c "
        cd /workspace && \
        CUDA_VISIBLE_DEVICES=7 \
        PYTHONPATH=. python examples/flux/model_inference/FLUX.1-dev_adaptive_ppd.py \
        --lora_checkpoint_path /workspace/models/ppd/flux1-dev_phipd_lora_302000.safetensors \
        --input_image /workspace/models/ppd/test1.jpg \
        --old_prompt \"\$(cat /workspace/models/ppd/test1.txt)\" \
        --new_prompt \"\$(cat /workspace/models/ppd/test1.txt)\" \
        --output outputs/test1/adaptive/original.png --radius 30 && \
        CUDA_VISIBLE_DEVICES=7 \
        PYTHONPATH=. python examples/flux/model_inference/FLUX.1-dev_adaptive_ppd.py \
        --lora_checkpoint_path /workspace/models/ppd/flux1-dev_phipd_lora_302000.safetensors \
        --input_image /workspace/models/ppd/test1.jpg \
        --old_prompt \"\$(cat /workspace/models/ppd/test1.txt)\" \
        --new_prompt \"\$(cat /workspace/test1_jungle.txt)\" \
        --output outputs/test1/adaptive/jungle.png --radius 30 && \
        CUDA_VISIBLE_DEVICES=7 \
        PYTHONPATH=. python examples/flux/model_inference/FLUX.1-dev_adaptive_ppd.py \
        --lora_checkpoint_path /workspace/models/ppd/flux1-dev_phipd_lora_302000.safetensors \
        --input_image /workspace/models/ppd/test1.jpg \
        --old_prompt \"\$(cat /workspace/models/ppd/test1.txt)\" \
        --new_prompt \"\$(cat /workspace/test1_desert.txt)\" \
        --output outputs/test1/adaptive/desert.png --radius 30 && \
        CUDA_VISIBLE_DEVICES=7 \
        PYTHONPATH=. python examples/flux/model_inference/FLUX.1-dev_adaptive_ppd.py \
        --lora_checkpoint_path /workspace/models/ppd/flux1-dev_phipd_lora_302000.safetensors \
        --input_image /workspace/models/ppd/test1.jpg \
        --old_prompt \"\$(cat /workspace/models/ppd/test1.txt)\" \
        --new_prompt \"\$(cat /workspace/test1_space.txt)\" \
        --output outputs/test1/adaptive/space.png --radius 30 && \
        CUDA_VISIBLE_DEVICES=7 \
        PYTHONPATH=. python examples/flux/model_inference/FLUX.1-dev_adaptive_ppd.py \
        --lora_checkpoint_path /workspace/models/ppd/flux1-dev_phipd_lora_302000.safetensors \
        --input_image /workspace/models/ppd/test1.jpg \
        --old_prompt \"\$(cat /workspace/models/ppd/test1.txt)\" \
        --new_prompt \"\$(cat /workspace/test1_cyberpunk.txt)\" \
        --output outputs/test1/adaptive/cyberpunk.png --radius 30 && \
        CUDA_VISIBLE_DEVICES=7 \
        PYTHONPATH=. python examples/flux/model_inference/FLUX.1-dev_adaptive_ppd.py \
        --lora_checkpoint_path /workspace/models/ppd/flux1-dev_phipd_lora_302000.safetensors \
        --input_image /workspace/models/ppd/test1.jpg \
        --old_prompt \"\$(cat /workspace/models/ppd/test1.txt)\" \
        --new_prompt \"\$(cat /workspace/test1_add_person.txt)\" \
        --output outputs/test1/adaptive/add_person.png --radius 30"