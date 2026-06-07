#!/bin/bash
# Compare new dropll Wan checkpoints across radius=16/30 x J=4/5 on all 60 nuCarla videos.
# 4 variants, one per GPU (0-3). Each GPU processes all 60 videos.
#
# Checkpoints: flux dropll step-6000, wan high dropll step-200, wan low dropll step-400
# Radius applies to both flux and wan. drop_ll=True for all.

FLUX_LORA="models/train/FLUX.1-dev_lora_wpd_dropll/step-6000.safetensors"
WAN_HIGH="models/train/Wan2.2-I2V-A14B_high_lora_wpd_dropll/step-200.safetensors"
WAN_LOW="models/train/Wan2.2-I2V-A14B_low_lora_wpd_dropll/step-400.safetensors"
DATA="/mrtstorage/users/kwang/nucarla_videos"
OUT="outputs/wan_ckpt_compare"

run_variant() {
    local GPU=$1
    local RADIUS=$2
    local J=$3
    local TAG="dropll_r${RADIUS}_J${J}"
    echo "=== [GPU $GPU] $TAG ==="
    docker run --rm --gpus "\"device=$GPU\"" \
        --name "wan_compare_${TAG}" \
        -v "$(pwd):/workspace" \
        -v /mrtstorage:/mrtstorage \
        -e HF_TOKEN=$HUGGING_FACE_TOKEN \
        -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        wpd bash -c "
            cd /workspace && PYTHONPATH=. python batch_sim2real_video_wavelet.py \
                --input_dataset '$DATA' \
                --output_dir '$OUT/$TAG' \
                --flux_lora $FLUX_LORA \
                --flux_cutoff_radius $RADIUS --flux_drop_ll --flux_J $J \
                --wan_low_lora $WAN_LOW \
                --wan_high_lora $WAN_HIGH \
                --wan_cutoff_radius $RADIUS --wan_drop_ll --wan_J $J
        " &
}

run_variant 0 16 4
run_variant 1 16 5
run_variant 2 30 4
run_variant 3 30 5

wait
echo "=== All done. Results in $OUT/ ==="
