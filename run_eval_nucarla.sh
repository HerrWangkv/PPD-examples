#!/bin/bash

BATCH_DIR=outputs/batch_scene0000_0059

docker run --rm --name eval_nucarla --gpus '"device=6"' \
    -v "$(pwd):/workspace" \
    wpd bash -c "
        cd /workspace && \
        for DIR in baseline baseline_newprompt dropll_step3000_J4 dropll_step6000_J4_newprompt; do
            echo \"--- \$DIR ---\" && \
            PYTHONPATH=. python calc_as_synthia.py \
                --gen_folder $BATCH_DIR/\$DIR
        done && \
        echo '=== Done ==='
    "
