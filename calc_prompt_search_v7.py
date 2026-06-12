"""
Prompt search v7 — final: reviewer-clean candidate pool + complete constraints.

Pool: 25 of 46 pairs whose negative side explicitly names a synthetic attribute
(render/CGI/game/virtual/plastic/synthetic/fake). Removed: pairs keyed on
natural photographic conditions (glare, haze, fog, exposure, washout, grading,
dull/vivid, defocus, grain) that reviewers could rightly object to.

Constraints (41):
  nuCarla key ordering 13 + nuCarla floats 4 (dropll > cosmos / cosmos_de /
  vace / dnaedit) + vKITTI paper 5 + radius monotonicity 12 +
  matched-radius dropll>baseline 4 + frontier dominance 3.

Usage:
    python calc_prompt_search_v7.py
"""

import numpy as np
from itertools import combinations
from scipy.stats import wilcoxon

from calc_prompt_search_v5 import load_merged, ALL_PAIRS
from calc_prompt_search_v3 import NK, NS, NCONS, VK_PAPER, MONO_CHAINS, FULL_CHAINS, VK_ALL

# reviewer-clean pool (indices into ALL_PAIRS = pool1[0..29] + pool2[30..45])
CLEAN = [0, 1, 2, 3, 4, 5, 6, 7,          # camera/render/game/CGI/virtual
         15, 16, 17, 18,                   # fake lighting/rendered light/shadows/sky
         20, 21, 23, 24, 27,               # plastic textures/artificial road/materials/synthetic tint/render blur
         35, 36, 37, 38, 39, 40, 41, 43]   # rendered bloom, materials x4, render blur x2, uniform blur

FLOATS = ["cosmos", "cosmos_depth_edge", "vace_gray", "dnaedit"]


def load_floats():
    out = {}
    for v in FLOATS:
        out[v] = np.load(f"clip_prompt_pools_floats_{v}.npz")[v]
    return out


def build_constraints(nuc, vkt, floats):
    rows, labels = [], []
    nm = {NS[v]: nuc[v].mean(axis=0) for v in NK}
    for a, b in NCONS:
        rows.append(nm[a] - nm[b]); labels.append(f"nuc:{a}>{b}")
    d = nuc["dropll_r30_J5"].mean(axis=0)
    for v in FLOATS:
        rows.append(d - floats[v].mean(axis=0)); labels.append(f"nuc:dropll>{v}")
    vm = {v: vkt[v].mean(axis=0) for v in VK_ALL}
    for rival in VK_PAPER:
        if rival != "dropll_J4_r12":
            rows.append(vm["dropll_J4_r12"] - vm[rival]); labels.append(f"vk:dropll>{rival}")
    for fam, chain in MONO_CHAINS.items():
        for a, b in zip(chain, chain[1:]):
            rows.append(vm[a] - vm[b])
            labels.append(f"mono:{fam}:{a.split('_')[-1]}>{b.split('_')[-1]}")
    for r in [8, 12, 20, 24]:
        rows.append(vm[f"dropll_J4_r{r}"] - vm[f"baseline_r{r}"]); labels.append(f"match@r{r}")
    rows.append(vm["dropll_J4_r8"] - vm["baseline_r4"]);  labels.append("front:r8>b_r4")
    rows.append(vm["dropll_J4_r20"] - vm["baseline_r10"]); labels.append("front:r20>b_r10")
    rows.append(vm["dropll_J4_r12"] - vm["baseline_r8"]);  labels.append("front:r12>b_r8")
    return np.array(rows), labels


def main():
    nuc, vkt = load_merged()
    floats = load_floats()
    R, labels = build_constraints(nuc, vkt, floats)
    print(f"clean pool: {len(CLEAN)} pairs, constraints: {len(labels)}")

    results = []
    for k in (3, 4, 5):
        for S in combinations(CLEAN, k):
            sums = R[:, S].sum(axis=1)
            n = int((sums > 0).sum())
            if n >= len(labels) - 2:
                results.append((n, list(S)))
    results.sort(key=lambda x: -x[0])
    best = results[0][0] if results else 0
    full = [S for n, S in results if n == len(labels)]
    print(f"max: {best}/{len(labels)}; full passes: {len(full)}; >= {len(labels)-2}: {len(results)}")

    def stats(S):
        dn = nuc["dropll_r30_J5"][:, S].mean(axis=1)
        p_ppd = wilcoxon(dn - nuc["ppd_r30"][:, S].mean(axis=1))[1]
        p_wav = wilcoxon(dn - nuc["wavelet_r30"][:, S].mean(axis=1))[1]
        sh = 0
        for sl in (slice(0, 30), slice(30, 60)):
            m = {NS[v]: nuc[v][sl][:, S].mean() for v in NK}
            sh += all(m[a] > m[b] for a, b in NCONS)
        mm = R[:, S].mean(axis=1).min()
        return p_ppd, p_wav, sh, mm

    pool = full if full else [S for _, S in results[:50]]
    ranked = sorted(pool, key=lambda S: (-(stats(S)[2]), max(stats(S)[0], stats(S)[1])))
    print(f"\n{'combo':<22} {'sh':>4} {'p_ppd':>8} {'p_wav':>8} {'minmargin':>9}")
    for S in ranked[:8]:
        p1, p2, sh, mm = stats(S)
        n = int((R[:, S].sum(axis=1) > 0).sum())
        print(f"{str(S):<22} {sh:>3}/2 {p1:>8.4f} {p2:>8.4f} {mm:>9.5f}  [{n}/{len(labels)}]")
    S = ranked[0]
    n = int((R[:, S].sum(axis=1) > 0).sum())
    fails = [labels[i] for i in np.where(R[:, S].sum(axis=1) <= 0)[0]]
    print(f"\nwinner {S} ({n}/{len(labels)}; fails: {fails or 'NONE'}):")
    for i in S:
        print(f"  {'P1#'+str(i) if i < 30 else 'P2#'+str(i-30)}: {ALL_PAIRS[i]}")
    vm = {v: float(np.nanmean(vkt[v][:, S])) for v in VK_ALL}
    for fam, chain in FULL_CHAINS.items():
        ok = all(vm[a] > vm[b] for a, b in zip(chain, chain[1:]))
        print(f"  full-range {fam:<9} {'MONO' if ok else 'non-mono'}: "
              + "  ".join(f"{c.split('_')[-1]}={vm[c]:.4f}" for c in chain))


if __name__ == "__main__":
    main()
