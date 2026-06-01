"""
Check all *_fid_clean.log files to verify they used CleanFID 'clean' mode.
Prints a table: variant | mode | FID | KID | status
"""
import glob
import re
import os

LOG_DIRS = ["logs/vkitti_eval", "logs/hypersim_eval"]

def check_log(path):
    try:
        text = open(path).read()
    except Exception:
        return None, None, None

    mode_m = re.search(r"Mode:\s+(\S+)", text)
    fid_m  = re.search(r"FID:\s+([\d.]+)", text)
    kid_m  = re.search(r"KID:\s+([\d.]+)", text)

    mode = mode_m.group(1) if mode_m else "unknown"
    fid  = fid_m.group(1)  if fid_m  else "-"
    kid  = kid_m.group(1)  if kid_m  else "-"
    return mode, fid, kid


for log_dir in LOG_DIRS:
    logs = sorted(glob.glob(f"{log_dir}/*_fid_clean.log"))
    if not logs:
        continue
    print(f"\n{'='*70}")
    print(f"  {log_dir}")
    print(f"{'='*70}")
    print(f"  {'Variant':<42} {'Mode':<18} {'FID':>8} {'KID':>8}  Status")
    print(f"  {'-'*42} {'-'*18} {'-'*8} {'-'*8}  ------")

    bad = []
    for log in logs:
        variant = os.path.basename(log).replace("_fid_clean.log", "")
        mode, fid, kid = check_log(log)
        ok = mode == "clean"
        status = "✓" if ok else "✗ WRONG MODE"
        print(f"  {variant:<42} {mode:<18} {fid:>8} {kid:>8}  {status}")
        if not ok:
            bad.append(variant)

    if bad:
        print(f"\n  ⚠️  Needs rerun ({len(bad)}): {', '.join(bad)}")
    else:
        print(f"\n  All clean ✓")
