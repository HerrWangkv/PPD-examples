#!/bin/bash
# Evaluate all vKITTI translated variants under outputs/vkitti/
# FID reference: KITTI tracking sequences 0001/0002/0006/0018/0020 (2126 frames,
#                scene-matched 1:1 with vKITTI clone condition)
# Clone-only evaluation: only {scene}_clone_{frame}.png images are used
# Results saved to logs/vkitti_eval/<variant>_{miou,depth,fid}.log
#
# Usage: bash run_eval_vkitti_all.sh [--gpus 0,1,2,3]

VKITTI_DIR=outputs/vkitti
RESULTS_DIR=logs/vkitti_eval
GPUS=(4 5 6 7)           # default GPUs to use

# Parse --gpus argument
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus) IFS=',' read -ra GPUS <<< "$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p $RESULTS_DIR
source .venv/bin/activate

# Collect all variant dirs under VKITTI_DIR
VARIANTS=()
for d in $VKITTI_DIR/*/; do
    VARIANTS+=("$(basename "$d")")
done
echo "Variants: ${VARIANTS[*]}"
echo "GPUs: ${GPUS[*]}"


run_variant() {
    local VARIANT=$1
    local GPU=$2
    local DIR=$VKITTI_DIR/$VARIANT
    echo "[GPU $GPU] Starting $VARIANT ..."

    CUDA_VISIBLE_DEVICES=$GPU python calc_miou_vkitti.py \
        --gen_folder $DIR --clone_only 2>&1 | tee $RESULTS_DIR/${VARIANT}_miou.log

    CUDA_VISIBLE_DEVICES=$GPU python calc_depth_metrics_vkitti.py \
        --gen_folder $DIR --clone_only 2>&1 | tee $RESULTS_DIR/${VARIANT}_depth.log

    CUDA_VISIBLE_DEVICES=$GPU python calc_fid_vkitti.py \
        --gen_folder $DIR --clone_only 2>&1 | tee $RESULTS_DIR/${VARIANT}_fid.log

    CUDA_VISIBLE_DEVICES=$GPU python calc_lpips_vkitti.py \
        --gen_folder $DIR 2>&1 | tee $RESULTS_DIR/${VARIANT}_lpips.log

    echo "[GPU $GPU] Done: $VARIANT"
}

# Distribute variants across GPUs in round-robin, N at a time
N=${#GPUS[@]}
i=0
for V in "${VARIANTS[@]}"; do
    GPU=${GPUS[$((i % N))]}
    run_variant "$V" "$GPU" &
    i=$((i + 1))
    # Wait every N jobs to avoid overloading
    if (( i % N == 0 )); then wait; fi
done
wait

# Summary
echo ""
echo "=============================================="
echo " vKITTI Eval Summary (clone-only)"
echo " FID ref: KITTI tracking scenes 0001/0002/0006/0018/0020 (2126 frames)"
echo "=============================================="
printf "%-28s %8s %8s %8s %8s %10s %8s\n" "Variant" "mIoU%" "DepSSIM" "AbsRel" "FID" "KID" "LPIPS"
echo "----------------------------------------------"
for V in "${VARIANTS[@]}"; do
    MIOU=$(grep -oP 'mIoU: \K[\d.]+' $RESULTS_DIR/${V}_miou.log 2>/dev/null | tail -1)
    DSSIM=$(grep -oP 'Depth SSIM:\s+\K[\d.]+' $RESULTS_DIR/${V}_depth.log 2>/dev/null | tail -1)
    ABSREL=$(grep -oP 'AbsRel:\s+\K[\d.]+' $RESULTS_DIR/${V}_depth.log 2>/dev/null | tail -1)
    FID=$(grep -oP 'FID:\s+\K[\d.]+' $RESULTS_DIR/${V}_fid.log 2>/dev/null | tail -1)
    KID=$(grep -oP 'KID:\s+\K[\d.]+' $RESULTS_DIR/${V}_fid.log 2>/dev/null | tail -1)
    LPIPS=$(grep -oP 'LPIPS \(alex\): \K[\d.]+' $RESULTS_DIR/${V}_lpips.log 2>/dev/null | tail -1)
    printf "%-28s %8s %8s %8s %8s %10s %8s\n" "$V" "${MIOU:--}" "${DSSIM:--}" "${ABSREL:--}" "${FID:--}" "${KID:--}" "${LPIPS:--}"
done
echo "=============================================="
