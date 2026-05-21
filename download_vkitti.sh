#!/bin/bash
# Download Virtual KITTI 1.3.1 RGB images (clone condition only)
# Source: https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-1/

OUTPUT_DIR=/mrtstorage/datasets_tmp/vkitti
mkdir -p $OUTPUT_DIR
cd $OUTPUT_DIR

wget -c "https://download.europe.naverlabs.com/virtual-kitti-1.3.1/vkitti_1.3.1_rgb.tar"

echo "Extracting..."
tar xf vkitti_1.3.1_rgb.tar

echo "Done. RGB frames at: $OUTPUT_DIR/vkitti_1.3.1_rgb/Scene*/clone/frames/rgb/Camera_0/"
