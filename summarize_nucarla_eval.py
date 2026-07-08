"""
Read nuCarla eval logs and print a summarized table (analogue of
summarize_vkitti_eval.py).

Sources:
  KID/FID            logs/kid_nucarla.log              (sections "=== name ===", "FID = x  KID = y")
  sFID/sKID          logs/nucarla_eval/<v>_sfid.log    ("sFID: x" / "sKID: y", EPE/DwD protocol)
  CMMD               logs/cmmd_nucarla.log             ("name  CMMD = x")
  CLIP-Residual      logs/nucarla_eval/<v>_clipres.log ("CLIP-Residual: x", residual_v7)
  Motion smoothness  eval_video/evaluation_results/<alias>_results_*_eval_results.json
                     and eval_video/ms_<v>/evaluation_results/*_eval_results.json
                     (VBench, scene_0000-0059 mean)
  LightEMMA          LightEMMA/output/<alias>/scene_*.json
                     (frame metrics -> scene mean -> macro mean over scenes
                      common to all methods; ADE_1s/2s/3s/avg, FDE; % vs carla)

Usage:
    python summarize_nucarla_eval.py [--sort kid|fid|skid|sfid|cmmd|clip_res|ms|ade_avg|fde]
"""

import argparse
import glob
import json
import os
import re

VARIANTS = [
    # (canonical, display, lightemma_dir, vbench_alias)
    ("input",                     "carla (raw sim)",        "carla",                     "carla"),
    ("cosmos",                    "Cosmos (no controls)",   "cosmos",                    "cosmos"),
    ("cosmos_depth_edge",         "Cosmos depth+edge",      "cosmos_depth_edge",         "cosmos_depth_edge"),
    ("cosmos_depth_seg_vis_edge", "Cosmos d+s+v+e",         "cosmos_depth_seg_vis_edge", "cosmos_depth_seg_vis_edge"),
    ("dnaedit",                   "DNAEdit",                "dnaedit",                   "dnaedit"),
    ("ditto",                     "Ditto",                  "ditto",                     "ditto"),
    ("ppd_r30",                   "PPD r30",                "ppd",                       "ppd"),
    ("vace_gray",                 "VACE gray",              "vace_gray",                 "vace_gray"),
    ("dropll_r30_J5",             "WPD drop_ll r30 J=5 (ours)", "dropll_r30_J5",         "dropll_r30_J5"),
    ("dropll_r22_J4",             "WPD drop_ll r22 J=4 (ours)", "dropll_r22_J4",         "dropll_r22_J4"),
    ("wavelet_r30",               "WPD baseline r30 (ours)", "wavelet",                  "wavelet"),
]

METRICS = [
    # key, header, higher_better, decimals
    ("kid",      "KID↓",      False, 4),
    ("fid",      "FID↓",      False, 2),
    ("skid",     "sKID↓",     False, 4),
    ("sfid",     "sFID↓",     False, 2),
    ("cmmd",     "CMMD↓",     False, 3),
    ("clip_res", "CLIP-Res↑", True,  4),
    ("ms",       "MotSmooth↑", True, 4),
    ("ade_1s",   "ADE_1s↓",   False, 4),
    ("ade_2s",   "ADE_2s↓",   False, 4),
    ("ade_3s",   "ADE_3s↓",   False, 4),
    ("ade_avg",  "ADE_avg↓",  False, 4),
    ("fde",      "FDE↓",      False, 4),
]
LIGHTEMMA_KEYS = {"ade_1s": "ADE_1s", "ade_2s": "ADE_2s", "ade_3s": "ADE_3s",
                  "ade_avg": "ADE_avg", "fde": "FDE"}


def extract(path, pattern):
    if not os.path.exists(path):
        return None
    m = re.findall(pattern, open(path).read())
    return float(m[-1]) if m else None


def load_kid_fid():
    """Parse sequential sections of logs/kid_nucarla.log."""
    out = {}
    path = "logs/kid_nucarla.log"
    if not os.path.exists(path):
        return out
    section = None
    for line in open(path):
        if "RESULTS" in line:  # trailing sorted-summary section repeats all values
            break
        m = re.match(r"=== (\S+) ===", line.strip())
        if m:
            section = m.group(1)
        m = re.search(r"FID = ([\d.]+)\s+KID = ([\d.]+)", line)
        if m and section:
            out[section] = (float(m.group(1)), float(m.group(2)))
    return out


CMMD_SUPPLEMENTAL = {
    # Values appended separately (log file had tqdm noise preventing regex parse)
    "dropll_r22_J4": 2.9844,
}

def load_cmmd():
    out = {}
    path = "logs/cmmd_nucarla.log"
    if not os.path.exists(path):
        return {**CMMD_SUPPLEMENTAL}
    for line in open(path):
        m = re.match(r"\s*(\S+)\s+CMMD = ([\d.]+)", line)
        if m:
            out[m.group(1)] = float(m.group(2))
    out.update(CMMD_SUPPLEMENTAL)
    return out


def load_motion_smoothness(alias):
    """VBench eval_results JSON; mean over scene_0000-0059."""
    paths = (glob.glob(f"eval_video/evaluation_results/{alias}_results_*_eval_results.json") +
             glob.glob(f"eval_video/ms_{alias}/evaluation_results/*_eval_results.json"))
    if not paths:
        return None
    data = json.load(open(sorted(paths)[-1]))["motion_smoothness"]
    vals = [v["video_results"] for v in data[1]
            if (m := re.search(r"scene_(\d{4})", v["video_path"])) and int(m.group(1)) < 60]
    return sum(vals) / len(vals) if vals else None


def load_lightemma():
    """Per-method scene means over scenes common to all methods."""
    root = "LightEMMA/output"
    scenes_per = {}
    for _, _, le_dir, _ in VARIANTS:
        files = glob.glob(os.path.join(root, le_dir, "scene_*.json"))
        if files:
            scenes_per[le_dir] = {os.path.basename(f).replace(".json", ""): f for f in files}
    if not scenes_per:
        return {}
    common = set.intersection(*(set(s) for s in scenes_per.values()))
    out = {}
    for le_dir, files in scenes_per.items():
        agg = {k: [] for k in LIGHTEMMA_KEYS.values()}
        for sc in common:
            frames = json.load(open(files[sc]))["frames"]
            for k in agg:
                vals = [f["metrics"][k] for f in frames
                        if isinstance(f.get("metrics"), dict) and k in f["metrics"]]
                if vals:
                    agg[k].append(sum(vals) / len(vals))
        out[le_dir] = {k: sum(v) / len(v) for k, v in agg.items() if v}
    out["_n_common"] = len(common)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sort", choices=[m[0] for m in METRICS], default=None)
    parser.add_argument("--pct", action="store_true",
                        help="show LightEMMA columns as 'value (±x%%)' vs carla")
    args = parser.parse_args()

    kidfid = load_kid_fid()
    cmmd = load_cmmd()
    le = load_lightemma()

    rows = []
    for canon, display, le_dir, vb in VARIANTS:
        base = f"logs/nucarla_eval/{canon}"
        m = {
            "fid":  kidfid.get(canon, (None, None))[0],
            "kid":  kidfid.get(canon, (None, None))[1],
            "sfid": (extract(f"{base}_patchfid.log", r"Score: ([\d.]+)") or
                     extract(f"{base}_sfid.log",    r"sFID: ([\d.]+)")),
            "skid": (extract(f"{base}_patchfid.log", r"KID: ([\d.]+)") or
                     extract(f"{base}_sfid.log",    r"sKID: ([\d.]+)")),
            "cmmd": cmmd.get(canon),
            "clip_res": (extract(f"{base}_clipres.log", r"CLIP-Residual: ([\d.]+)") or
                         extract(f"{base}_clipiqa.log", r"CLIP-IQA: ([\d.]+)")),
            "ms":   load_motion_smoothness(vb),
        }
        for key, lk in LIGHTEMMA_KEYS.items():
            m[key] = le.get(le_dir, {}).get(lk)
        rows.append((canon, display, m))

    if args.sort:
        hb = dict((k, h) for k, _, h, _ in METRICS)[args.sort]
        rows.sort(key=lambda r: r[2][args.sort] if r[2][args.sort] is not None
                  else (float("-inf") if hb else float("inf")), reverse=hb)

    # best / 2nd best excluding raw sim
    non_input = [(c, m) for c, _, m in rows if c != "input"]
    bests, seconds = {}, {}
    for key, _, hb, _ in METRICS:
        vals = sorted([(c, m[key]) for c, m in non_input if m[key] is not None],
                      key=lambda x: x[1], reverse=hb)
        if vals:
            bests[key] = vals[0][0]
        if len(vals) > 1:
            seconds[key] = vals[1][0]

    carla = next(m for c, _, m in rows if c == "input")
    col_w = 28
    header = f"{'Variant':<{col_w}}" + "".join(f" {h:>10}" for _, h, _, _ in METRICS)
    print(header)
    print("-" * len(header))
    for canon, display, m in rows:
        cells = []
        for key, _, _, dec in METRICS:
            v = m[key]
            if v is None:
                cells.append(f"{'-':>10}")
                continue
            s = f"{v:.{dec}f}"
            if args.pct and key in LIGHTEMMA_KEYS and canon != "input" and carla.get(key):
                s += f"({(v - carla[key]) / carla[key] * 100:+.1f}%)"
            if bests.get(key) == canon:
                s = f"**{s}**"
            elif seconds.get(key) == canon:
                s = f"*{s}*"
            cells.append(f"{s:>10}")
        print(f"{display:<{col_w}}" + " ".join([""] + cells))
    print("-" * len(header))
    n = le.get("_n_common")
    print(f"** = best, * = 2nd best (excl. raw sim). LightEMMA over {n} common scenes." if n
          else "** = best, * = 2nd best (excl. raw sim).")


if __name__ == "__main__":
    main()
