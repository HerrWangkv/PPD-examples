"""
Debug script for DNAEdit save pipeline.
Tests VAE decode + ffmpeg save in the exact FSDP context that edit_DNAEdit uses,
without running the 50-step denoising loop.

Run: torchrun --nproc_per_node=4 debug_dnaedit_save.py
"""
import os, sys, subprocess, tempfile
import torch
import torch.distributed as dist
from contextlib import contextmanager

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "DNAEdit/wan-DNAEdit"))
import wan
from wan.configs import WAN_CONFIGS

try:
    import torch.cuda.amp as amp
except ImportError:
    import torch.amp as amp

rank       = int(os.environ.get("RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)

if world_size > 1:
    dist.init_process_group(backend="nccl", init_method="env://",
                            rank=rank, world_size=world_size)

if rank == 0:
    print(f"[rank0] loading model...", flush=True)

cfg = WAN_CONFIGS["t2v-14B"]
model = wan.WanT2V(
    config=cfg,
    checkpoint_dir="models/Wan-AI/Wan2.1-T2V-14B",
    device_id=local_rank,
    rank=rank,
    dit_fsdp=(world_size > 1),
)

if rank == 0:
    print(f"[rank0] model loaded. vae z_dim={model.vae.model.z_dim}", flush=True)

# Build dummy latent of the same shape edit_DNAEdit would produce
# vae_stride=(4,8,8), size=(1280,704), frame_num=49
F, W, H = 49, 1280, 704
z_dim = model.vae.model.z_dim
latent_shape = (
    z_dim,
    (F - 1) // cfg.vae_stride[0] + 1,   # 13
    H // cfg.vae_stride[1],               # 88
    W // cfg.vae_stride[2],               # 160
)
if rank == 0:
    print(f"[rank0] dummy latent shape: {latent_shape}", flush=True)

dummy_latent = torch.randn(*latent_shape, dtype=torch.float32, device=f"cuda:{local_rank}")

# Replicate the exact context used in edit_DNAEdit
@contextmanager
def noop_no_sync():
    yield

no_sync = getattr(model.model, 'no_sync', noop_no_sync)

if rank == 0:
    print(f"[rank0] calling vae.decode inside FSDP no_sync context...", flush=True)

videos = None
with amp.autocast(dtype=model.param_dtype), torch.no_grad(), no_sync():
    if rank == 0:
        videos = model.vae.decode([dummy_latent])

if dist.is_initialized():
    dist.barrier()

result = videos[0] if rank == 0 else None

if rank == 0:
    print(f"[rank0] result type={type(result)}, is None={result is None}", flush=True)
    if result is not None:
        print(f"[rank0] result shape={result.shape}, dtype={result.dtype}, "
              f"min={result.min():.3f}, max={result.max():.3f}", flush=True)

# Test save pipeline
if rank == 0 and result is not None:
    print(f"[rank0] testing save pipeline...", flush=True)
    out_path = "outputs/nucarla/dnaedit/debug_save_test.mp4"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        frames = result.clamp(-1, 1).cpu().float()
        frames = ((frames + 1) / 2 * 255).to(torch.uint8)
        frames = frames.permute(1, 2, 3, 0).numpy()  # (T, H, W, C)
        print(f"[rank0] frames shape={frames.shape}, dtype={frames.dtype}", flush=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            import cv2
            for i, frame in enumerate(frames):
                cv2.imwrite(f"{tmpdir}/{i:04d}.png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            ret = subprocess.run([
                "/usr/bin/ffmpeg", "-y", "-r", "10",
                "-i", f"{tmpdir}/%04d.png",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                out_path
            ], capture_output=True)
            if ret.returncode != 0:
                print(f"[rank0] ffmpeg FAILED: {ret.stderr.decode()[:500]}", flush=True)
            else:
                size = os.path.getsize(out_path)
                print(f"[rank0] SAVE OK: {out_path} ({size} bytes)", flush=True)
    except Exception as e:
        import traceback
        print(f"[rank0] save EXCEPTION: {e}", flush=True)
        traceback.print_exc()
elif rank == 0:
    print(f"[rank0] result is None — save skipped", flush=True)

if dist.is_initialized():
    dist.barrier()
if rank == 0:
    print("[rank0] debug complete", flush=True)
