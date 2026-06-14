#!/bin/bash
# Run E3 phase trajectory experiment inside Docker
# Usage: bash run_phase_trajectory.sh
set -e

cd /workspace
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python exp_phase_trajectory.py 2>&1 | tee logs/phase_trajectory.log
echo "Done. Figure saved to figures/phase_trajectory.png"
