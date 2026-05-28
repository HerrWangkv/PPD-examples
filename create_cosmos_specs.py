"""
Create Cosmos Transfer 2.5 inference JSON specs for vKITTI clone sequences.

Modes:
  video       — one spec per scene, input is an MP4 (single control)
  image       — one spec per frame, input is a PNG (img2img, max_frames=1, single control)
  multicontrol — one spec per scene, MP4 input, depth+edge+seg auto-computed (video2video)

Mirrors the structure of CosmosTransfer/assets/robot_example/*.json and
CosmosTransfer/assets/car_example/multicontrol/car_multicontrol_spec.json.

Usage:
    # Multi-control video2video (recommended for sim2real quality)
    python create_cosmos_specs.py --mode multicontrol

    # Single-control video mode
    python create_cosmos_specs.py --mode video --controls depth

    # Image mode (one spec per PNG frame)
    python create_cosmos_specs.py --mode image --controls depth seg

Run with:
    # Multicontrol video (one spec per scene)
    docker run ... python examples/inference.py -i /staging/specs/multicontrol/*.json -o /out

    # Image — pass all specs at once so they batch on one model load
    docker run ... python examples/inference.py \\
        -i /staging/specs/depth/image/*.json -o /out
"""

import argparse
import json
import os
import glob

SCENES = ["0001", "0002", "0006", "0018", "0020"]
VKITTI_RGB = "/mrtstorage/datasets_tmp/vkitti/vkitti_1.3.1_rgb"

PROMPT = (
    "A photorealistic driving scene filmed from a forward-facing camera mounted on a vehicle. "
    "The road is flanked by buildings, trees, and other cars under realistic natural lighting."
)

NEGATIVE_PROMPT = (
    "ugly, low quality, CG, render, unreal, game, cartoon, blur, low res, "
    "dashboard, steering wheel, windshield frame, car interior, lens artifacts"
)

CONTROL_DEFAULTS: dict[str, dict] = {
    "depth": {"control_weight": 1.0},
    "seg":   {"control_weight": 1.0},
    "edge":  {"control_weight": 0.2},
    "vis":   {"control_weight": 0.5, "preset_blur_strength": "high"},
}

# Controls combined in multicontrol mode (auto-computed, no control_path needed)
MULTICONTROL_CONTROLS = ["depth", "edge", "seg"]


def make_video_spec(scene: str, control: str, video_dir: str, guidance: float) -> dict:
    return {
        "name": f"vkitti_{scene}_clone_{control}",
        "prompt": PROMPT,
        "video_path": f"{video_dir}/{scene}_clone.mp4",
        "guidance": guidance,
        control: CONTROL_DEFAULTS[control],
    }


def make_multicontrol_spec(scene: str, video_dir: str, guidance: float) -> dict:
    spec: dict = {
        "name": f"vkitti_{scene}_clone",
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "video_path": f"{video_dir}/{scene}_clone.mp4",
        "guidance": guidance,
    }
    for ctrl in MULTICONTROL_CONTROLS:
        spec[ctrl] = CONTROL_DEFAULTS[ctrl]
    return spec


def make_image_spec(scene: str, frame: str, controls: list[str], input_dir: str,
                    guidance: float, sigma_max: float | None) -> dict:
    frame_id = os.path.splitext(frame)[0]  # e.g. "00042"
    spec: dict = {
        "name": f"{scene}_clone_{frame_id}",
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "video_path": f"{input_dir}/{scene}/clone/{frame}",
        "num_video_frames_per_chunk": 1,
        "max_frames": 1,
        "guidance": guidance,
    }
    for ctrl in controls:
        spec[ctrl] = CONTROL_DEFAULTS[ctrl]
    if sigma_max is not None:
        spec["sigma_max"] = sigma_max
    return spec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["video", "image", "multicontrol"], default="multicontrol")
    parser.add_argument("--out_dir", default="/tmp/cosmos_vkitti/specs")
    parser.add_argument(
        "--video_dir_docker", default="/staging/videos",
        help="(video/multicontrol mode) path to MP4s as seen inside Docker",
    )
    parser.add_argument(
        "--input_dir_docker", default="/mrtstorage/datasets_tmp/vkitti/vkitti_1.3.1_rgb",
        help="(image mode) path to vKITTI RGB root as seen inside Docker",
    )
    parser.add_argument(
        "--controls", nargs="+", default=["depth", "seg"],
        choices=list(CONTROL_DEFAULTS),
        help="(video/image mode only) which controls to generate specs for",
    )
    parser.add_argument("--scenes", nargs="+", default=SCENES)
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--sigma_max", type=float, default=None)
    args = parser.parse_args()

    created = 0

    if args.mode == "multicontrol":
        ctrl_dir = os.path.join(args.out_dir, "multicontrol")
        os.makedirs(ctrl_dir, exist_ok=True)
        for scene in args.scenes:
            spec = make_multicontrol_spec(scene, args.video_dir_docker, args.guidance)
            out_path = os.path.join(ctrl_dir, f"{scene}_spec.json")
            with open(out_path, "w") as f:
                json.dump(spec, f, indent=4)
            print(f"  {out_path}")
            created += 1
        print(f"\n{created} specs written to {ctrl_dir}/")
        print("\nRun inference with:")
        print(f"  python examples/inference.py -i /staging/specs/multicontrol/*.json -o /out/multicontrol")

    elif args.mode == "video":
        for control in args.controls:
            ctrl_dir = os.path.join(args.out_dir, control)
            os.makedirs(ctrl_dir, exist_ok=True)
            for scene in args.scenes:
                spec = make_video_spec(scene, control, args.video_dir_docker, args.guidance)
                out_path = os.path.join(ctrl_dir, f"{scene}_spec.json")
                with open(out_path, "w") as f:
                    json.dump(spec, f, indent=4)
                print(f"  {out_path}")
                created += 1
        print(f"\n{created} specs written to {args.out_dir}/")

    else:  # image mode — all --controls combined into one spec per frame
        combo_name = "_".join(args.controls)
        ctrl_dir = os.path.join(args.out_dir, combo_name, "image")
        os.makedirs(ctrl_dir, exist_ok=True)
        for scene in args.scenes:
            frame_dir = os.path.join(VKITTI_RGB, scene, "clone")
            frames = sorted(os.path.basename(p) for p in glob.glob(f"{frame_dir}/*.png"))
            if not frames:
                print(f"  [warn] no frames found in {frame_dir}")
                continue
            for frame in frames:
                spec = make_image_spec(scene, frame, args.controls, args.input_dir_docker,
                                       args.guidance, args.sigma_max)
                frame_id = os.path.splitext(frame)[0]
                out_path = os.path.join(ctrl_dir, f"{scene}_clone_{frame_id}.json")
                with open(out_path, "w") as f:
                    json.dump(spec, f, indent=4)
                created += 1
            print(f"  scene {scene}: {len(frames)} specs → {ctrl_dir}/")
        print(f"\n{created} specs written to {ctrl_dir}/")
        print("\nRun inference with:")
        print(f"  python examples/inference.py -i /staging/specs/{combo_name}/image/*.json -o /out/{combo_name}")


if __name__ == "__main__":
    main()
