#!/bin/bash
# Translate Virtual KITTI (clone + overcast + morning) → real using WPD
# vKITTI native res: 1242×375 → run at 384×1280
# r=16 (equivalent to r=30 on nuCarla, scaled for 48px latent height)

VKITTI_DIR=/mrtstorage/datasets_tmp/vkitti/vkitti_1.3.1_rgb
INPUT_DIR=/mrtstorage/datasets_tmp/vkitti/neutral_flat

PROMPT="A photorealistic photograph taken from a forward-facing vehicle-mounted camera. Natural outdoor lighting, authentic surface textures, real-world colors."
NEG="ugly, low quality, CG, render, unreal, game, cartoon, blur, low res, dashboard, steering wheel, windshield frame, car interior, lens artifacts"

CKPT6000=models/train/FLUX.1-dev_lora_wpd_dropll/step-6000.safetensors

# --- Flatten clone + overcast + morning frames (idempotent) ---
mkdir -p $INPUT_DIR
for SEQ_DIR in $VKITTI_DIR/*/; do
    SEQ=$(basename $SEQ_DIR)
    for COND in clone overcast morning; do
        for IMG in $SEQ_DIR/$COND/*.png; do
            [ -f "$IMG" ] || continue
            ln -sf $IMG $INPUT_DIR/${SEQ}_${COND}_$(basename $IMG)
        done
    done
done
echo "$(ls $INPUT_DIR | wc -l) frames in $INPUT_DIR"

# --- GPU 0: baseline ---
docker run --rm --name vkitti_baseline_newprompt --gpus '"device=0"' \
    -v "$(pwd):/workspace" -v "/mrtstorage:/mrtstorage" \
    -e HF_TOKEN=$HUGGING_FACE_TOKEN \
    wpd bash -c "cd /workspace && mkdir -p outputs/vkitti/baseline_newprompt && \
        PYTHONPATH=. python batch_sim2real_image_wavelet.py \
            --input_dir $INPUT_DIR --output_dir outputs/vkitti/baseline_newprompt \
            --flux_lora flux.safetensors --flux_cutoff_radius 16 \
            --prompt '$PROMPT' --negative_prompt '$NEG' \
            --height 384 --width 1280" &

# --- GPU 1: drop_ll J=3 ---
docker run --rm --name vkitti_dropll_step6000_J3 --gpus '"device=1"' \
    -v "$(pwd):/workspace" -v "/mrtstorage:/mrtstorage" \
    -e HF_TOKEN=$HUGGING_FACE_TOKEN \
    wpd bash -c "cd /workspace && mkdir -p outputs/vkitti/dropll_step6000_J3 && \
        PYTHONPATH=. python batch_sim2real_image_wavelet.py \
            --input_dir $INPUT_DIR --output_dir outputs/vkitti/dropll_step6000_J3 \
            --flux_lora $CKPT6000 --flux_cutoff_radius 16 --flux_drop_ll --flux_J 3 \
            --prompt '$PROMPT' --negative_prompt '$NEG' \
            --height 384 --width 1280" &

# --- GPU 2: drop_ll J=4 ---
docker run --rm --name vkitti_dropll_step6000_J4 --gpus '"device=2"' \
    -v "$(pwd):/workspace" -v "/mrtstorage:/mrtstorage" \
    -e HF_TOKEN=$HUGGING_FACE_TOKEN \
    wpd bash -c "cd /workspace && mkdir -p outputs/vkitti/dropll_step6000_J4 && \
        PYTHONPATH=. python batch_sim2real_image_wavelet.py \
            --input_dir $INPUT_DIR --output_dir outputs/vkitti/dropll_step6000_J4 \
            --flux_lora $CKPT6000 --flux_cutoff_radius 16 --flux_drop_ll --flux_J 4 \
            --prompt '$PROMPT' --negative_prompt '$NEG' \
            --height 384 --width 1280" &

# --- GPU 3: drop_ll J=5 ---
docker run --rm --name vkitti_dropll_step6000_J5 --gpus '"device=3"' \
    -v "$(pwd):/workspace" -v "/mrtstorage:/mrtstorage" \
    -e HF_TOKEN=$HUGGING_FACE_TOKEN \
    wpd bash -c "cd /workspace && mkdir -p outputs/vkitti/dropll_step6000_J5 && \
        PYTHONPATH=. python batch_sim2real_image_wavelet.py \
            --input_dir $INPUT_DIR --output_dir outputs/vkitti/dropll_step6000_J5 \
            --flux_lora $CKPT6000 --flux_cutoff_radius 16 --flux_drop_ll --flux_J 5 \
            --prompt '$PROMPT' --negative_prompt '$NEG' \
            --height 384 --width 1280" &

wait
echo "=== vKITTI translation done ==="
