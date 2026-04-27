set -e
PROMPT="$(cat models/ppd/test1.txt)"
INPUT=models/ppd/test1.jpg

run() {
  local name=$1; shift
  local out=outputs/dino_std_check/$name
  mkdir -p "$out"
  echo "=== $name ==="
  PYTHONPATH=. python validate_dino_trajectory.py \
    --input_image "$INPUT" --prompt "$PROMPT" \
    --output_dir "$out" --num_inference_steps 50 --save_frames_at 0 \
    --seed 0 "$@" 2>&1 | tee "$out/log.txt" | grep -E "noise stats|DINO opt step"
}

run gaussian        --noise_mode gaussian
run ppd_r20         --noise_mode ppd     --ppd_radius 20
run wpd_r20         --noise_mode wavelet --wpd_radius 20
run dino_baseline   --noise_mode dino    --dino_opt_steps 300
run dino_layer4     --noise_mode dino    --dino_opt_steps 300 --dino_loss_layers 4
run dino_4_11_23    --noise_mode dino    --dino_opt_steps 300 --dino_loss_layers 4 11 23
