#!/bin/bash

VIDEO_DIR=/mrtstorage/users/kwang/nucarla_videos/rgb

# GPU 4: baseline (flux.safetensors, no drop_ll, new prompt)
docker run --rm --name batch_baseline_newprompt --gpus '"device=4"' \
    -v "$(pwd):/workspace" \
    -v "/mrtstorage:/mrtstorage" \
    -e HF_TOKEN=$HUGGING_FACE_TOKEN \
    wpd bash -c "
        cd /workspace && \
        mkdir -p outputs/batch_scene0000_0059/baseline_newprompt && \
        for i in \$(seq 0 59); do
            SCENE=scene_\$(printf '%04d' \$i)
            echo \"=== Baseline new prompt: \$SCENE ===\" && \
            PYTHONPATH=. python sim2real_video_wavelet.py \
                --rgb_video $VIDEO_DIR/\${SCENE}.mp4 \
                --flux_lora flux.safetensors \
                --flux_cutoff_radius 30 \
                --image-only \
                --output_video outputs/batch_scene0000_0059/baseline_newprompt/\${SCENE}.mp4 || \
            echo \"FAILED: \$SCENE\"
        done && \
        echo '=== Baseline new prompt done ==='
    " &

# GPU 5: step-6000 drop_ll J=4 new prompt
docker run --rm --name batch_dropll_step6000_newprompt --gpus '"device=5"' \
    -v "$(pwd):/workspace" \
    -v "/mrtstorage:/mrtstorage" \
    -e HF_TOKEN=$HUGGING_FACE_TOKEN \
    wpd bash -c "
        cd /workspace && \
        mkdir -p outputs/batch_scene0000_0059/dropll_step6000_J4_newprompt && \
        for i in \$(seq 0 59); do
            SCENE=scene_\$(printf '%04d' \$i)
            echo \"=== drop_ll step-6000 J=4 new prompt: \$SCENE ===\" && \
            PYTHONPATH=. python sim2real_video_wavelet.py \
                --rgb_video $VIDEO_DIR/\${SCENE}.mp4 \
                --flux_lora models/train/FLUX.1-dev_lora_wpd_dropll/step-6000.safetensors \
                --flux_cutoff_radius 30 \
                --flux_drop_ll --flux_J 4 \
                --image-only \
                --output_video outputs/batch_scene0000_0059/dropll_step6000_J4_newprompt/\${SCENE}.mp4 || \
            echo \"FAILED: \$SCENE\"
        done && \
        echo '=== drop_ll step-6000 J=4 new prompt done ==='
    " &

wait
echo "=== Both done ==="
