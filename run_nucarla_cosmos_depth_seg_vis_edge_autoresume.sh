#!/usr/bin/env bash
# Auto-resuming wrapper for cosmos depth+seg+vis+edge nuCarla translation.
# Keeps retrying until all 60 scenes are done.

set -euo pipefail

OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/outputs/nucarla/cosmos_depth_seg_vis_edge"
TOTAL=60
ATTEMPT=0

while true; do
    DONE=$(ls "$OUT_DIR"/*.mp4 2>/dev/null | grep -v control | wc -l)
    echo "[$(date '+%H:%M:%S')] Attempt $((++ATTEMPT)): $DONE/$TOTAL scenes done."

    if [[ "$DONE" -ge "$TOTAL" ]]; then
        echo "All $TOTAL scenes complete. Done."
        break
    fi

    bash "$(dirname "${BASH_SOURCE[0]}")/run_nucarla_cosmos_depth_seg_vis_edge.sh" \
        2>&1 | tee -a /tmp/cosmos_depth_seg_vis_edge.log || true

    DONE_AFTER=$(ls "$OUT_DIR"/*.mp4 2>/dev/null | grep -v control | wc -l)
    echo "[$(date '+%H:%M:%S')] After attempt $ATTEMPT: $DONE_AFTER/$TOTAL done."

    if [[ "$DONE_AFTER" -ge "$TOTAL" ]]; then
        echo "All $TOTAL scenes complete. Done."
        break
    fi

    if [[ "$DONE_AFTER" -le "$DONE" ]]; then
        echo "No progress made (stuck at $DONE_AFTER). Waiting 30s before retry..."
        sleep 30
    else
        echo "Progress: $((DONE_AFTER - DONE)) new scenes. Retrying immediately..."
    fi
done

echo "=== Final: $(ls "$OUT_DIR"/*.mp4 2>/dev/null | grep -v control | wc -l)/$TOTAL mp4s in $OUT_DIR ==="
