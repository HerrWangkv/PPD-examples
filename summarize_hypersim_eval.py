"""
Summarize Hypersim eval results from logs/hypersim_eval/.

Usage:
    python summarize_hypersim_eval.py
"""
import os
import re

LOG_DIR = "logs/hypersim_eval"

VARIANTS_ORDER = [
    "input",
    "flowedit",
    "dnaedit",
    "cosmos_depth_edge_imgs",
    "ppd_r20",
    "wpd_r20",
    "dropll_J5_r24",
]

DISPLAY_NAMES = {
    "input":              "input (raw sim)",
    "flowedit":           "FlowEdit",
    "dnaedit":            "DNAEdit",
    "cosmos_depth_edge_imgs":  "Cosmos depth+edge",
    "ppd_r20":            "PPD r20",
    "wpd_r20":            "WPD baseline r20",
    "dropll_J5_r24":      "WPD J=5 r24 (drop_ll)",
}


def extract(path, pattern):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        text = f.read()
    m = re.findall(pattern, text)
    return float(m[-1]) if m else None


def load(v):
    base = os.path.join(LOG_DIR, v)
    fid_log = f"{base}_fid_clean.log" if os.path.exists(f"{base}_fid_clean.log") else f"{base}_fid.log"
    return {
        "fid":     extract(fid_log,             r"FID:\s+([\d.]+)"),
        "kid":     extract(fid_log,             r"KID:\s+([\d.]+)"),
        "dep_ssim":extract(f"{base}_depth.log", r"Depth SSIM:\s+([\d.]+)"),
        "abs_rel": extract(f"{base}_depth.log", r"AbsRel:\s+([\d.]+)"),
        "miou":     extract(f"{base}_miou.log",    r"mIoU:\s+([\d.]+)"),
        "clipiqa":  extract(f"{base}_clipiqa.log", r"CLIP-IQA:\s+([\d.]+)"),
    }


def fmt(val, d=4):
    return f"{val:.{d}f}" if val is not None else "-"


def main():
    import glob
    all_logs = (glob.glob(os.path.join(LOG_DIR, "*_fid.log")) +
                glob.glob(os.path.join(LOG_DIR, "*_fid_clean.log")))
    discovered = sorted(set(
        os.path.basename(p).replace("_fid_clean.log", "").replace("_fid.log", "")
        for p in all_logs
    ))
    ordered = [v for v in VARIANTS_ORDER if v in discovered]
    ordered += [v for v in discovered if v not in VARIANTS_ORDER]

    rows = [(v, load(v)) for v in ordered]

    # Find best and second-best per metric, excluding input
    non_input = [(v, m) for v, m in rows if v != "input"]
    METRICS = [
        ("clipiqa",  True),
        ("fid",      False),
        ("kid",      False),
        ("miou",     True),
        ("dep_ssim", True),
        ("abs_rel",  False),
    ]
    bests, seconds = {}, {}
    for metric, higher in METRICS:
        vals = sorted(
            [(v, m[metric]) for v, m in non_input if m[metric] is not None],
            key=lambda x: x[1], reverse=higher
        )
        if vals:
            bests[metric] = vals[0][0]
        if len(vals) > 1:
            seconds[metric] = vals[1][0]

    def mark(v, metric, val, decimals=4):
        s = fmt(val, decimals)
        if val is None:
            return s
        if bests.get(metric) == v:
            return f"**{s}**"
        if seconds.get(metric) == v:
            return f"*{s}*"
        return s

    header = f"{'Variant':<28} {'CLIP-IQA':>11} {'FID':>8} {'KID':>8} {'mIoU':>9} {'DepSSIM':>10} {'AbsRel':>10}"
    print(header)
    print("-" * len(header))
    for v, m in rows:
        name = DISPLAY_NAMES.get(v, v)
        print(f"{name:<28} "
              f"{mark(v, 'clipiqa',  m['clipiqa'],  4):>11} "
              f"{mark(v, 'fid',      m['fid'],      2):>8} "
              f"{mark(v, 'kid',      m['kid'],      4):>8} "
              f"{mark(v, 'miou',     m['miou'],     4):>9} "
              f"{mark(v, 'dep_ssim', m['dep_ssim'], 4):>10} "
              f"{mark(v, 'abs_rel',  m['abs_rel'],  4):>10}")
    print("-" * len(header))
    print("** = best, * = 2nd best (excl. raw sim input)")


if __name__ == "__main__":
    main()
