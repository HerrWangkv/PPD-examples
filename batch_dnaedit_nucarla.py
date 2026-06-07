"""Batch DNAEdit translation for nuCarla videos. One process per GPU,
each handles a GPU-rank slice of the video list."""
import argparse
import glob
import os
import sys
import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "DNAEdit/wan-DNAEdit"))
import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS
from wan.utils.utils import cache_video


SRC_PROMPT = (
    "A synthetic driving simulation video with unrealistic rendering, "
    "computer graphics, game-like appearance, CG environment."
)
TGT_PROMPT = (
    "A photorealistic driving scene filmed from a moving vehicle. "
    "Natural lighting, real urban buildings, trees, cars on the street. "
    "High resolution, realistic textures."
)


def load_video_frames(video_path, frame_num, target_size):
    """Load frames from mp4, resize, return (C,T,H,W) float tensor in [-1,1]."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, frame_num, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            frame = frames[-1] if frames else np.zeros((*target_size[::-1], 3), np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, target_size)  # (W, H)
        frames.append(frame)
    cap.release()
    arr = np.stack(frames, axis=0).astype(np.float32) / 127.5 - 1.0  # (T,H,W,C)
    tensor = torch.from_numpy(arr).permute(3, 0, 1, 2).unsqueeze(0)   # (1,C,T,H,W)
    return tensor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str,
                        default="/mrtstorage/users/kwang/nucarla_videos/rgb")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--ckpt_dir", type=str,
                        default="models/Wan-AI/Wan2.1-T2V-14B")
    parser.add_argument("--task", type=str, default="t2v-14B")
    parser.add_argument("--size", type=str, default="1280*704")
    parser.add_argument("--frame_num", type=int, default=49)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--guide_scale", type=float, default=1.0)
    parser.add_argument("--tgt_guide_scale", type=float, default=5.0)
    parser.add_argument("--jmp", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_denoise", action="store_true",
                        help="VAE encode+decode only, skip denoising (for testing save pipeline)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Init distributed (torchrun sets RANK/WORLD_SIZE/LOCAL_RANK)
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    if world_size > 1:
        import torch.distributed as dist
        dist.init_process_group(backend="nccl", init_method="env://",
                                rank=rank, world_size=world_size)

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)

    video_files = sorted(glob.glob(os.path.join(args.input_dir, "*.mp4")))
    if rank == 0:
        print(f"Processing {len(video_files)} videos with {world_size} GPUs each")

    w, h = map(int, args.size.split("*"))
    size_tuple = (w, h)

    cfg = WAN_CONFIGS[args.task]
    if rank == 0:
        print(f"Loading WanT2V from {args.ckpt_dir}...")
    model = wan.WanT2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=local_rank,
        rank=rank,
        dit_fsdp=(world_size > 1),
    )

    for video_path in video_files:
        name = os.path.splitext(os.path.basename(video_path))[0]
        out_path = os.path.join(args.output_dir, f"{name}.mp4")
        if os.path.exists(out_path):
            if rank == 0:
                print(f"Skipping {name} (exists)")
            continue

        if rank == 0:
            print(f"Editing {name}...")
        video_tensor = load_video_frames(video_path, args.frame_num, (w, h))

        if args.skip_denoise:
            # VAE roundtrip only — test save pipeline without 50-step denoising
            if rank == 0:
                vt = video_tensor.squeeze(0).to(model.device)  # (C,T,H,W)
                with torch.no_grad():
                    z = model.vae.encode([vt])
                    result = model.vae.decode(z)[0]  # (C,T,H,W)
            else:
                result = None
        else:
            result = model.edit_DNAEdit(
                video_tensor,
                SRC_PROMPT,
                TGT_PROMPT,
                size=size_tuple,
                frame_num=args.frame_num,
                sampling_steps=args.sample_steps,
                guide_scale=args.guide_scale,
                tgt_guide_scale=args.tgt_guide_scale,
                jmp=args.jmp,
                seed=args.seed,
                offload_model=False,
            )

        if rank == 0:
            print(f"[rank0] result type={type(result)}, is None={result is None}", flush=True)
        if rank == 0 and result is not None:
            # result: (C, T, H, W) float in [-1, 1]
            frames = result.clamp(-1, 1).cpu().float()
            frames = ((frames + 1) / 2 * 255).to(torch.uint8)
            frames = frames.permute(1, 2, 3, 0).numpy()  # (T, H, W, C) uint8 RGB
            import subprocess, tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                for i, frame in enumerate(frames):
                    cv2.imwrite(f"{tmpdir}/{i:04d}.png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                subprocess.run([
                    "/usr/bin/ffmpeg", "-y", "-r", "10",
                    "-i", f"{tmpdir}/%04d.png",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                    out_path
                ], check=True, capture_output=True)
            print(f"Saved {out_path}")

        # barrier so all ranks stay in sync before the next scene
        if world_size > 1:
            import torch.distributed as dist
            dist.barrier()


if __name__ == "__main__":
    main()
