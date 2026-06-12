#!/bin/bash
# Run patch-sFID for all nuCarla variants. Real-side VGG cache is built on first run
# and reused for subsequent variants.

REAL_DIR=/tmp/nuscenes
LOG=logs/patch_sfid_nucarla.log

mkdir -p logs

declare -A VARIANTS=(
    ["input"]="outputs/nucarla/input/rgb"
    ["cosmos"]="outputs/nucarla/cosmos"
    ["ditto"]="outputs/nucarla/ditto"
    ["ppd_r30"]="outputs/nucarla/ppd/flux_30_wan_30"
    ["wavelet_r30"]="outputs/nucarla/wavelet/flux_30_30_1_wan_30_30_1"
    ["vace_gray"]="outputs/nucarla/vace_gray"
    ["dnaedit"]="outputs/nucarla/dnaedit"
    ["dropll_r30_J5"]="outputs/nucarla/wavelet/dropll_r30_J5"
    ["cosmos_depth_edge"]="outputs/nucarla/cosmos_depth_edge_imgs"
    ["cosmos_depth_seg_vis_edge"]="outputs/nucarla/cosmos_depth_seg_vis_edge_imgs"
)

ORDER=(input cosmos ditto ppd_r30 wavelet_r30 vace_gray dnaedit dropll_r30_J5 cosmos_depth_edge cosmos_depth_seg_vis_edge)

for name in "${ORDER[@]}"; do
    dir="${VARIANTS[$name]}"
    echo "=== $name ===" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=0 conda run -n sim2real_eval python eval_video/eval_patch_sfid.py \
        --real_dir "$REAL_DIR" \
        --fake_dir "$dir" 2>&1 | tee -a "$LOG"
    echo "" | tee -a "$LOG"
done

echo "=== ALL DONE ===" | tee -a "$LOG"
grep -E "(=== |Score:)" "$LOG"
