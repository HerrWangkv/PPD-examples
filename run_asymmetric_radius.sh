#!/bin/bash
# Asymmetric FLUX/Wan radius experiments on 60 nuCarla videos.
# FLUX radius (structure lock) and Wan radius (realism) varied independently.
# J=4, drop_ll=True for all. 2x2 grid: {flux_r24, flux_r30} x {wan_r8, wan_r16}

FLUX_LORA="models/train/FLUX.1-dev_lora_wpd_dropll/step-6000.safetensors"
WAN_HIGH="models/train/Wan2.2-I2V-A14B_high_lora_wpd_dropll/step-200.safetensors"
WAN_LOW="models/train/Wan2.2-I2V-A14B_low_lora_wpd_dropll/step-400.safetensors"
DATA="/mrtstorage/users/kwang/nucarla_videos"
OUT="outputs/nucarla/asymmetric_radius"

run_variant() {
    local GPU=$1
    local FLUX_R=$2
    local WAN_R=$3
    local TAG="fr${FLUX_R}_wr${WAN_R}_J4"
    echo "=== [GPU $GPU] $TAG ==="
    docker run -it --rm --gpus "\"device=$GPU\"" \
        --name "asym_${TAG}" \
        -v "$(pwd):/workspace" \
        -v /mrtstorage:/mrtstorage \
        -e HF_TOKEN=$HUGGING_FACE_TOKEN \
        -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        wpd bash -c "
            cd /workspace && PYTHONPATH=. python batch_sim2real_video_wavelet.py \
                --input_dataset '$DATA' \
                --output_dir '$OUT/$TAG' \
                --flux_lora $FLUX_LORA \
                --flux_cutoff_radius $FLUX_R --flux_drop_ll --flux_J 4 \
                --wan_low_lora $WAN_LOW \
                --wan_high_lora $WAN_HIGH \
                --wan_cutoff_radius $WAN_R --wan_drop_ll --wan_J 4
        "
}

mkdir -p "$OUT"

run_variant 1 22 22

wait
echo "=== All done. Results in $OUT/ ==="
