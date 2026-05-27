#!/bin/bash
# Download Virtual KITTI 1.3.1
# Source: https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-1/
#
# Usage:
#   bash download_vkitti.sh          # download all (RGB + semantic + depth)
#   bash download_vkitti.sh rgb      # RGB only
#   bash download_vkitti.sh semantic # semantic/instance segmentation only
#   bash download_vkitti.sh depth    # depth only

set -e

OUTPUT_DIR=/mrtstorage/datasets_tmp/vkitti
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

BASE_URL="https://download.europe.naverlabs.com/virtual-kitti-1.3.1"

download_and_extract() {
    local filename="$1"
    echo "==> Downloading $filename ..."
    wget -c "$BASE_URL/$filename"
    echo "==> Extracting $filename ..."
    tar xf "$filename"
    echo "==> Done: $filename"
}

MODE="${1:-all}"

case "$MODE" in
    rgb)
        download_and_extract "vkitti_1.3.1_rgb.tar"
        ;;
    semantic)
        # Contains per-pixel semantic + instance segmentation (488 MB)
        download_and_extract "vkitti_1.3.1_scenegt.tar"
        ;;
    depth)
        # Contains depth ground truth maps (5.1 GB)
        download_and_extract "vkitti_1.3.1_depthgt.tar"
        ;;
    all)
        download_and_extract "vkitti_1.3.1_rgb.tar"
        download_and_extract "vkitti_1.3.1_scenegt.tar"
        download_and_extract "vkitti_1.3.1_depthgt.tar"
        ;;
    *)
        echo "Unknown mode: $MODE. Use: rgb | semantic | depth | all"
        exit 1
        ;;
esac

echo ""
echo "=== Download complete ==="
echo "  RGB:      $OUTPUT_DIR/vkitti_1.3.1_rgb/Scene{01,02,06,18,20}/<condition>/frames/rgb/Camera_0/"
echo "  Semantic: $OUTPUT_DIR/vkitti_1.3.1_scenegt/Scene{01,02,06,18,20}/<condition>/frames/classSegmentation/Camera_0/"
echo "  Depth:    $OUTPUT_DIR/vkitti_1.3.1_depthgt/Scene{01,02,06,18,20}/<condition>/frames/depth/Camera_0/"
