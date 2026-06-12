"""
Prompt search v3: joint constraints across nuCarla ordering, vKITTI paper
ordering, and working-range radius monotonicity (r8-r24, all three families).

30 constraints total:
  - nuCarla 13: ditto > {dropll,wavelet,ppd,cdsve,input}; dropll > {wavelet,ppd,
    cdsve,input}; wavelet/ppd > {cdsve,input}
  - vKITTI paper 5: dropll_J4_r12 > {input, flowedit, dnaedit, cosmos_de, ppd_r12}
  - monotonicity 12 (vKITTI, score decreasing in radius):
      ppd:      r8 > r12 > r16 > r20 > r24          (4)
      baseline: r8 > r10 > r12 > r16 > r20 > r24    (5)
      dropll:   r8 > r12 > r20 > r24                (3)

Usage:
    python calc_prompt_search_v3.py
"""

import numpy as np
from itertools import combinations
from scipy.stats import wilcoxon

from calc_clip_prompt_search import CANDIDATES

# ---- nuCarla per-scene data ----
NK = ["ditto", "dropll_r30_J5", "wavelet_r30", "ppd_r30",
      "cosmos_depth_seg_vis_edge", "input"]
NS = dict(zip(NK, ["ditto", "dropll", "wavelet", "ppd", "cdsve", "input"]))
NCONS = ([("ditto", v) for v in ["dropll", "wavelet", "ppd", "cdsve", "input"]] +
         [("dropll", v) for v in ["wavelet", "ppd", "cdsve", "input"]] +
         [("wavelet", "cdsve"), ("wavelet", "input"),
          ("ppd", "cdsve"), ("ppd", "input")])

# ---- vKITTI per-image data ----
VK_PAPER = ["input", "flowedit", "dnaedit", "cosmos_depth_edge_imgs",
            "ppd_r12", "dropll_J4_r12"]
MONO_CHAINS = {
    "ppd":      ["ppd_r8", "ppd_r12", "ppd_r16", "ppd_r20", "ppd_r24"],
    "baseline": ["baseline_r8", "baseline_r10", "baseline_r12",
                 "baseline_newprompt", "baseline_r20", "baseline_r24"],
    "dropll":   ["dropll_J4_r8", "dropll_J4_r12", "dropll_J4_r20", "dropll_J4_r24"],
}
# bonus diagnostic (not constraints): pure-realism theory predicts monotonicity
# even outside the working range
FULL_CHAINS = {
    "ppd":      ["ppd_r8", "ppd_r12", "ppd_r16", "ppd_r20", "ppd_r24", "ppd_r32"],
    "baseline": ["baseline_r4", "baseline_r8", "baseline_r10", "baseline_r12",
                 "baseline_newprompt", "baseline_r20", "baseline_r24"],
    "dropll":   ["dropll_J4_r8", "dropll_J4_r12", "dropll_J4_r20", "dropll_J4_r24"],
}
VK_ALL = sorted(set(VK_PAPER + [v for c in FULL_CHAINS.values() for v in c]))


def load():
    nd = np.load("clip_prompt_search_scenes.npz")
    nuc = {k: nd[k] for k in NK}
    vkt = {v: np.load(f"clip_pool_vkitti_{v}.npy") for v in VK_ALL}
    return nuc, vkt


def constraint_count(nuc, vkt, idxs):
    n_pass = 0
    fails = []
    # nuCarla 13
    m = {NS[v]: nuc[v][:, idxs].mean() for v in NK}
    for a, b in NCONS:
        if m[a] > m[b]:
            n_pass += 1
        else:
            fails.append(f"nuc:{a}>{b}")
    # vKITTI paper 5
    vm = {v: vkt[v][:, idxs].mean() for v in VK_ALL}
    for rival in VK_PAPER:
        if rival == "dropll_J4_r12":
            continue
        if vm["dropll_J4_r12"] > vm[rival]:
            n_pass += 1
        else:
            fails.append(f"vk:dropll>{rival}")
    # monotonicity 12
    for fam, chain in MONO_CHAINS.items():
        for a, b in zip(chain, chain[1:]):
            if vm[a] > vm[b]:
                n_pass += 1
            else:
                fails.append(f"mono:{fam}:{a.split('_')[-1]}>{b.split('_')[-1]}")
    # matched-radius dropll > baseline 4
    for r in [8, 12, 20, 24]:
        if vm[f"dropll_J4_r{r}"] > vm[f"baseline_r{r}"]:
            n_pass += 1
        else:
            fails.append(f"match:dropll>baseline@r{r}")
    return n_pass, fails


def main():
    nuc, vkt = load()
    total = 34

    combos = [[i] for i in range(30)]
    combos += [list(c) for r in (2, 3, 4, 5) for c in combinations(range(30), r)]
    print(f"searching {len(combos)} combos against {total} constraints...")

    scored = []
    for c in combos:
        n, fails = constraint_count(nuc, vkt, c)
        scored.append((n, c, fails))
    scored.sort(key=lambda x: -x[0])

    best_n = scored[0][0]
    print(f"\nmax satisfied: {best_n}/{total}")
    print(f"combos at max: {sum(1 for s in scored if s[0] == best_n)}")

    print(f"\n{'combo':<18} {'pass':>5}  failed constraints")
    for n, c, fails in scored[:12]:
        print(f"{str(c):<18} {n:>4}   {', '.join(fails) if fails else 'ALL PASS'}")

    # significance + full-range diagnostic for the top combos
    for n, c, fails in scored[:3]:
        dn = nuc["dropll_r30_J5"][:, c].mean(axis=1)
        pn = nuc["ppd_r30"][:, c].mean(axis=1)
        wn = nuc["wavelet_r30"][:, c].mean(axis=1)
        dv = vkt["dropll_J4_r12"][:, c].mean(axis=1)
        pv = vkt["ppd_r12"][:, c].mean(axis=1)
        print(f"\ncombo {c}: pairs = {[CANDIDATES[i] for i in c]}")
        print(f"  nuCarla dropll vs ppd:     p={wilcoxon(dn - pn)[1]:.4f}  ({(dn > pn).sum()}/60)")
        print(f"  nuCarla dropll vs wavelet: p={wilcoxon(dn - wn)[1]:.4f}  ({(dn > wn).sum()}/60)")
        print(f"  vKITTI  dropll vs ppd:     p={wilcoxon(dv - pv)[1]:.2e}  ({(dv > pv).sum()}/2126)")
        vm = {v: vkt[v][:, c].mean() for v in VK_ALL}
        for fam, chain in FULL_CHAINS.items():
            vals = "  ".join(f"{v.split('_')[-1]}={vm[v]:.4f}" for v in chain)
            ok = all(vm[a] > vm[b] for a, b in zip(chain, chain[1:]))
            print(f"  full-range {fam:<9} {'MONO' if ok else 'non-mono'}: {vals}")


if __name__ == "__main__":
    main()
