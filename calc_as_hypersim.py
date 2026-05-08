"""
Calculate NeuralRemaster Appearance Score (AS) for Hypersim sim2real experiments.
Compares: original Hypersim, WPD-translated, FlowEdit-translated.
"""

import os
import argparse
import torch
import clip
from PIL import Image
from tqdm import tqdm

EXPERIMENTS = {
    "hypersim_original": "/mrtstorage/datasets_tmp/hypersim",
    "hypersim_wavelet":  "/mrtstorage/users/kwang/hypersim_wavelet",
    "hypersim_flowedit": "/mrtstorage/users/kwang/hypersim_flowedit",
}

POSITIVE_PROMPTS = [
    "Photorealistic",
    "Real-world",
    "Natural illumination",
    "Realistic textures and materials",
]

NEGATIVE_PROMPTS = [
    "Unrealistic",
    "Simulated",
    "Flat lighting",
    "Computer graphics rendering",
    "Artifacts",
]


def get_image_paths(folder):
    exts = ('.png', '.jpg', '.jpeg')
    paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(exts):
                paths.append(os.path.join(root, f))
    return sorted(paths)


def compute_as(image_paths, model, preprocess, tp, tn, batch_size, device):
    scores = []
    for i in tqdm(range(0, len(image_paths), batch_size)):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                batch_images.append(preprocess(img).unsqueeze(0))
            except Exception:
                print(f"Warning: could not read {p}, skipping.")
        if not batch_images:
            continue
        image_input = torch.cat(batch_images).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            sim_pos = image_features @ tp
            sim_neg = image_features @ tn
            sim_neg = torch.clamp(sim_neg, min=1e-6)
            scores.extend((sim_pos / sim_neg).cpu().tolist())
    return sum(scores) / len(scores) if scores else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None,
                        help="Run only one experiment: original / wavelet / flowedit")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--backbone", type=str, default="ViT-B/32")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading CLIP ({args.backbone})...")
    model, preprocess = clip.load(args.backbone, device=device)
    model.eval()

    print("Positive prompts:", POSITIVE_PROMPTS)
    print("Negative prompts:", NEGATIVE_PROMPTS)

    with torch.no_grad():
        tp = model.encode_text(clip.tokenize(POSITIVE_PROMPTS).to(device)).mean(dim=0)
        tn = model.encode_text(clip.tokenize(NEGATIVE_PROMPTS).to(device)).mean(dim=0)
        tp = tp / tp.norm()
        tn = tn / tn.norm()

    exps = {k: v for k, v in EXPERIMENTS.items()
            if args.only is None or args.only in k}

    results = {}
    for name, path in exps.items():
        if not os.path.exists(path):
            print(f"[SKIP] {name}: path not found ({path})")
            continue
        print(f"\n{'='*60}\n Experiment: {name}\n{'='*60}")
        paths = get_image_paths(path)
        print(f"  Found {len(paths)} images")
        score = compute_as(paths, model, preprocess, tp, tn, args.batch_size, device)
        results[name] = score
        print(f"  AS: {score:.4f}")

    print(f"\n{'='*60}")
    print(f"{'Experiment':<30} {'AS':>10}")
    print(f"{'-'*60}")
    for name, score in results.items():
        print(f"{name:<30} {score:>10.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
