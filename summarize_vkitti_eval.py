"""
Read vKITTI eval logs and print a summarized table.

Usage:
    python summarize_vkitti_eval.py
    python summarize_vkitti_eval.py --sort fid
    python summarize_vkitti_eval.py --variants dropll baseline
"""

import argparse
import os
import re
import glob

LOG_DIR = "logs/vkitti_eval"

VARIANTS_ORDER = [
    "input",
    "flowedit",
    "dnaedit",
    "kontext",
    "cosmos_depth_imgs",
    "cosmos_depth_edge_imgs",
    "cosmos_depth_vis_imgs",
    "cosmos_depth_seg_imgs",
    "cosmos_seg_edge_imgs",
    "cosmos_depth_seg_edge_imgs",
    "cosmos_depth_seg_vis_imgs",
    "cosmos_depth_seg_vis_edge_imgs",
    "ppd_r8",
    "ppd_r12",
    "ppd_r16",
    "ppd_r20",
    "ppd_r24",
    "ppd_r32",
    "wpd_ppd_ckpt_r8",
    "wpd_ppd_ckpt_r12",
    "wpd_ppd_ckpt_r16",
    "wpd_ppd_ckpt_r20",
    "wpd_ppd_ckpt_r24",
    "baseline_r4",
    "baseline_r8",
    "baseline_r10",
    "baseline_r12",
    "baseline_r20",
    "baseline_r24",
    "baseline_newprompt",
    "dropll_step6000_J3",
    "dropll_step6000_J4",
    "dropll_step6000_J5",
    "dropll_J4_r8",
    "dropll_J3_r12",
    "dropll_J4_r12",
    "dropll_J5_r12",
    "dropll_J4_r20",
    "dropll_J4_r24",
    "ablation_dropll_r8",
    "ablation_dropll_r12",
    "ablation_dropll_r20",
    "ablation_dropll_r24",
]

DISPLAY_NAMES = {
    "input":                        "input (raw sim)",
    "flowedit":                     "FlowEdit",
    "dnaedit":                      "DNAEdit",
    "kontext":                      "FLUX.1-Kontext",
    "cosmos_depth_imgs":            "Cosmos depth",
    "cosmos_depth_edge_imgs":       "Cosmos depth+edge",
    "cosmos_depth_vis_imgs":        "Cosmos depth+vis",
    "cosmos_depth_seg_imgs":        "Cosmos depth+seg",
    "cosmos_seg_edge_imgs":         "Cosmos seg+edge",
    "cosmos_depth_seg_edge_imgs":   "Cosmos depth+seg+edge",
    "cosmos_depth_seg_vis_imgs":    "Cosmos depth+seg+vis",
    "cosmos_depth_seg_vis_edge_imgs": "Cosmos depth+seg+vis+edge",
    "ppd_r8":                                "PPD r8",
    "ppd_r12":                               "PPD r12",
    "ppd_r16":                               "PPD r16",
    "ppd_r20":                               "PPD r20",
    "ppd_r24":                               "PPD r24",
    "ppd_r32":                               "PPD r32",
    "wpd_ppd_ckpt_r8":                       "WPD (PPD ckpt) r8",
    "wpd_ppd_ckpt_r12":                      "WPD (PPD ckpt) r12",
    "wpd_ppd_ckpt_r16":                      "WPD (PPD ckpt) r16",
    "wpd_ppd_ckpt_r20":                      "WPD (PPD ckpt) r20",
    "wpd_ppd_ckpt_r24":                      "WPD (PPD ckpt) r24",
    "baseline_r4":                  "WPD baseline r4",
    "baseline_r8":                  "WPD baseline r8",
    "baseline_r10":                 "WPD baseline r10",
    "baseline_r12":                 "WPD baseline r12",
    "baseline_r20":                 "WPD baseline r20",
    "baseline_r24":                 "WPD baseline r24",
    "baseline_newprompt":           "WPD baseline r16",
    "dropll_step6000_J3":           "WPD J=3 r16",
    "dropll_step6000_J4":           "WPD J=4 r16",
    "dropll_step6000_J5":           "WPD J=5 r16",
    "dropll_J4_r8":                 "WPD J=4 r8",
    "dropll_J3_r12":                "WPD J=3 r12",
    "dropll_J4_r12":                "WPD J=4 r12",
    "dropll_J5_r12":                "WPD J=5 r12",
    "dropll_J4_r20":                "WPD J=4 r20",
    "ablation_dropll_r8":           "Ablation: infer drop_ll r8",
    "ablation_dropll_r12":          "Ablation: infer drop_ll r12",
    "ablation_dropll_r20":          "Ablation: infer drop_ll r20",
    "ablation_dropll_r24":          "Ablation: infer drop_ll r24",
    "dropll_J4_r24":                "WPD J=4 r24",
}


def extract(path, pattern):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        text = f.read()
    m = re.findall(pattern, text)
    return float(m[-1]) if m else None


def load_variant(v):
    base = os.path.join(LOG_DIR, v)
    # Prefer clean-mode FID log if available, fall back to legacy
    fid_log = f"{base}_fid_clean.log" if os.path.exists(f"{base}_fid_clean.log") else f"{base}_fid.log"
    return {
        "clip_iqa": extract(f"{base}_clipiqa.log", r"CLIP-IQA: ([\d.]+)"),
        "clip_res": extract(f"{base}_clipres.log",  r"CLIP-Residual: ([\d.]+)"),
        "fid":      extract(fid_log,               r"FID:\s+([\d.]+)"),
        "kid":      extract(fid_log,               r"KID:\s+([\d.]+)"),
        "miou":     extract(f"{base}_miou.log",    r"mIoU: ([\d.]+)"),
        "dep_ssim": extract(f"{base}_depth.log",   r"Depth SSIM:\s+([\d.]+)"),
        "abs_rel":  extract(f"{base}_depth.log",   r"AbsRel:\s+([\d.]+)"),
        "lpips":    extract(f"{base}_lpips.log",   r"LPIPS \(alex\): ([\d.]+)"),
    }


def fmt(val, decimals=4):
    return f"{val:.{decimals}f}" if val is not None else "-"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sort", choices=["clip_res", "clip_iqa", "fid", "kid", "miou", "dep_ssim", "abs_rel", "lpips"], default=None)
    parser.add_argument("--variants", nargs="+", default=None, help="Filter variants by substring")
    args = parser.parse_args()

    # Discover all variants from logs
    all_logs = (glob.glob(os.path.join(LOG_DIR, "*_fid.log")) +
                glob.glob(os.path.join(LOG_DIR, "*_fid_clean.log")))
    discovered = sorted(set(
        os.path.basename(p).replace("_fid_clean.log", "").replace("_fid.log", "")
        for p in all_logs
        if "kitti_real" not in p
    ))

    # Use predefined order, append any discovered variants not in order
    ordered = [v for v in VARIANTS_ORDER if v in discovered]
    ordered += [v for v in discovered if v not in VARIANTS_ORDER]

    if args.variants:
        ordered = [v for v in ordered if any(f in v for f in args.variants)]

    rows = []
    for v in ordered:
        metrics = load_variant(v)
        rows.append((v, metrics))

    if args.sort:
        reverse = args.sort in ("clip_iqa", "miou", "dep_ssim")
        rows.sort(key=lambda x: x[1][args.sort] if x[1][args.sort] is not None else (float('-inf') if reverse else float('inf')), reverse=reverse)

    # Find best and second-best (excluding input)
    non_input = [(v, m) for v, m in rows if v != "input"]
    bests, seconds = {}, {}
    for metric, higher_better in [("clip_res", True), ("clip_iqa", True), ("fid", False), ("kid", False),
                                   ("miou", True), ("dep_ssim", True), ("abs_rel", False), ("lpips", False)]:
        vals = sorted(
            [(v, m[metric]) for v, m in non_input if m[metric] is not None],
            key=lambda x: x[1], reverse=higher_better
        )
        if vals:
            bests[metric] = vals[0][0]
        if len(vals) > 1:
            seconds[metric] = vals[1][0]

    # Print table
    col_w = 26
    header = f"{'Variant':<{col_w}} {'KID':>8} {'FID':>8} {'CLIP-Res':>9} {'CLIP-IQA':>9} {'mIoU':>7} {'DepSSIM':>8} {'AbsRel':>8} {'LPIPS':>7}"
    print(header)
    print("-" * len(header))

    for v, m in rows:
        name = DISPLAY_NAMES.get(v, v)

        def mark(metric, val, decimals=4):
            s = fmt(val, decimals)
            if val is None:
                return s
            if bests.get(metric) == v:
                return f"**{s}**"
            if seconds.get(metric) == v:
                return f"*{s}*"
            return s

        print(f"{name:<{col_w}} "
              f"{mark('kid', m['kid']):>8} "
              f"{mark('fid', m['fid'], 2):>8} "
              f"{mark('clip_res', m['clip_res']):>9} "
              f"{mark('clip_iqa', m['clip_iqa']):>9} "
              f"{mark('miou', m['miou'], 2):>7} "
              f"{mark('dep_ssim', m['dep_ssim']):>8} "
              f"{mark('abs_rel', m['abs_rel']):>8} "
              f"{mark('lpips', m['lpips']):>7}")

    print("-" * len(header))
    print("** = best, * = 2nd best (excl. raw sim input)")


if __name__ == "__main__":
    main()
