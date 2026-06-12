#!/bin/bash
set -e
REAL_DIR=/tmp/nuscenes
LOG=logs/patch_sfid_nucarla.log

mkdir -p logs
> "$LOG"

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

run_variant() {
    local name=$1 gpu=$2
    local dir="${VARIANTS[$name]}"
    echo "=== $name (GPU $gpu) ===" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=$gpu conda run -n sim2real_eval python -u \
        eval_video/eval_patch_sfid.py --real_dir "$REAL_DIR" --fake_dir "$dir" --tag "$name" 2>&1 \
        | tee -a "$LOG"
    echo "" | tee -a "$LOG"
}

i=0
while [ $i -lt ${#ORDER[@]} ]; do
    name0="${ORDER[$i]}"
    name1="${ORDER[$((i+1))]:-}"

    if [ -n "$name1" ]; then
        echo ">>> Launching $name0 (GPU 0) + $name1 (GPU 1) in parallel"
        run_variant "$name0" 0 &
        PID0=$!
        run_variant "$name1" 1 &
        PID1=$!
        wait $PID0 $PID1
    else
        echo ">>> Launching $name0 (GPU 0)"
        run_variant "$name0" 0
    fi

    i=$((i+2))
done

echo "=== ALL DONE ===" | tee -a "$LOG"
echo ""
echo "=== SUMMARY ==="
grep -E "^=== |^Score:|^KID:" "$LOG"
