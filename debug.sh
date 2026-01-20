#!/bin/bash

docker build -t wpd .

docker run -it --rm --name debug --gpus all \
    -v "$(pwd):/workspace" \
    -e HF_TOKEN=$HUGGING_FACE_TOKEN \
    -w /workspace \
    wpd