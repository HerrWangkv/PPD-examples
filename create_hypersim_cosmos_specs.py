"""
Create Cosmos Transfer 2.5 inference JSON specs for Hypersim frames.

Hypersim is a flat directory of JPGs: ai_XXX_XXX_camYY_frameZZZZ.jpg
Depth and edge are auto-computed by Cosmos from the input image.

Usage:
    python create_hypersim_cosmos_specs.py [--controls depth edge] [--out_dir /tmp/cosmos_hypersim/specs]
    python create_hypersim_cosmos_specs.py --controls depth edge --guidance 3.0

Run with:
    bash run_hypersim_cosmos.sh --control depth_edge --gpu 0
"""

import argparse
import json
import os
import glob

HYPERSIM_DIR = "/mrtstorage/datasets_tmp/hypersim"

PROMPT = (
    "A photorealistic indoor scene photographed with a real camera. "
    "Natural and artificial lighting, realistic materials, authentic textures."
)
NEGATIVE_PROMPT = (
    "ugly, low quality, CG, render, unreal, game, cartoon, blur, low res, "
    "3D rendering, computer graphics"
)

CONTROL_DEFAULTS: dict[str, dict] = {
    "depth": {"control_weight": 1.0},
    "edge":  {"control_weight": 0.2},
}


def make_image_spec(frame_path: str, input_dir_docker: str, controls: list[str],
                    guidance: float) -> dict:
    fname = os.path.basename(frame_path)
    name = os.path.splitext(fname)[0]
    spec: dict = {
        "name": f"hypersim_{name}",
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "video_path": f"{input_dir_docker}/{fname}",
        "num_video_frames_per_chunk": 1,
        "max_frames": 1,
        "guidance": guidance,
    }
    for ctrl in controls:
        spec[ctrl] = CONTROL_DEFAULTS[ctrl]
    return spec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controls", nargs="+", default=["depth", "edge"],
                        choices=list(CONTROL_DEFAULTS))
    parser.add_argument("--out_dir", default="/tmp/cosmos_hypersim/specs")
    parser.add_argument("--input_dir", default=HYPERSIM_DIR,
                        help="Hypersim flat JPG dir (host path, for listing)")
    parser.add_argument("--input_dir_docker", default="/data/hypersim",
                        help="Same dir as seen inside Docker container")
    parser.add_argument("--guidance", type=float, default=3.0)
    args = parser.parse_args()

    combo_name = "_".join(args.controls)
    out_dir = os.path.join(args.out_dir, combo_name)
    os.makedirs(out_dir, exist_ok=True)

    frames = sorted(glob.glob(os.path.join(args.input_dir, "*.jpg")) +
                    glob.glob(os.path.join(args.input_dir, "*.png")))
    print(f"Found {len(frames)} frames in {args.input_dir}")

    for frame_path in frames:
        spec = make_image_spec(frame_path, args.input_dir_docker, args.controls, args.guidance)
        out_path = os.path.join(out_dir, spec["name"] + ".json")
        with open(out_path, "w") as f:
            json.dump(spec, f, indent=2)

    print(f"{len(frames)} specs written to {out_dir}/")
    print("\nRun inference with:")
    print(f"  bash run_hypersim_cosmos.sh --control {combo_name} --gpu <N>")


if __name__ == "__main__":
    main()
