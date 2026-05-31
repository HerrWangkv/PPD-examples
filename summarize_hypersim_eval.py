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
    "wpd_r20",
    "dropll_J5_r24",
]

DISPLAY_NAMES = {
    "input":          "input (raw sim)",
    "flowedit":       "FlowEdit",
    "wpd_r20":        "WPD baseline r20",
    "dropll_J5_r24":  "WPD J=5 r24",
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

    header = f"{'Variant':<28} {'FID':>8} {'KID':>8} {'DepSSIM':>8} {'AbsRel':>8}"
    print(header)
    print("-" * len(header))
    for v in ordered:
        m = load(v)
        name = DISPLAY_NAMES.get(v, v)
        print(f"{name:<28} {fmt(m['fid'], 2):>8} {fmt(m['kid']):>8} "
              f"{fmt(m['dep_ssim']):>8} {fmt(m['abs_rel']):>8}")
    print("-" * len(header))


if __name__ == "__main__":
    main()
