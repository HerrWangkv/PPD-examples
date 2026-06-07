#!/bin/bash
# Convert .npy frame arrays saved by batch_dnaedit_nucarla.py to H.264 mp4 on the host.
# Run from repo root with .venv activated (or without, just needs ffmpeg + numpy/cv2 on PATH).
set -e

OUTPUT_DIR="${1:-outputs/nucarla/dnaedit}"
TMPDIR_BASE=$(mktemp -d)

source .venv/bin/activate

for npy in "$OUTPUT_DIR"/*.npy; do
    [ -f "$npy" ] || { echo "No .npy files found in $OUTPUT_DIR"; break; }
    name=$(basename "$npy" .npy)
    mp4="$OUTPUT_DIR/$name.mp4"
    if [ -f "$mp4" ]; then
        echo "Skip $name (mp4 exists)"
        continue
    fi
    tmpdir="$TMPDIR_BASE/$name"
    mkdir -p "$tmpdir"
    echo "Encoding $name..."
    python - <<EOF
import numpy as np, cv2, sys
frames = np.load("$npy")  # (T, H, W, C) uint8 RGB
for i, f in enumerate(frames):
    cv2.imwrite(f"$tmpdir/{i:04d}.png", cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
print(f"  wrote {len(frames)} frames")
EOF
    ffmpeg -y -r 10 -i "$tmpdir/%04d.png" \
        -c:v libx264 -pix_fmt yuv420p -crf 18 \
        "$mp4" -loglevel error
    rm -rf "$tmpdir"
    rm "$npy"
    echo "  -> $mp4"
done

rm -rf "$TMPDIR_BASE"
echo "Done."
