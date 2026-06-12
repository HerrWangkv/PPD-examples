"""
Prompt search v5: pool1 (30 pairs) + pool2 (16 wording variants) = 46 pairs,
uniform mean, the 34-constraint set of search v3/v4:
  nuCarla ordering 13 + vKITTI paper 5 + working-range monotonicity 12 +
  matched-radius dropll>baseline 4.

Constraint evaluation is linear in the per-pair means, so each constraint is a
46-dim row vector; a combo S passes constraint c iff mean(c[S]) > 0.

Usage:
    python calc_prompt_search_v5.py
"""

import numpy as np
from itertools import combinations
from scipy.stats import wilcoxon

from calc_clip_prompt_search import CANDIDATES as POOL1
from calc_prompt_pool2 import NEW_CANDIDATES as POOL2
from calc_prompt_search_v3 import (NK, NS, NCONS, VK_PAPER, MONO_CHAINS,
                                   FULL_CHAINS, VK_ALL)

ALL_PAIRS = list(POOL1) + list(POOL2)
N1 = len(POOL1)


def load_merged():
    nd1 = np.load("clip_prompt_search_scenes.npz")
    nd2 = np.load("clip_prompt_pool2_scenes.npz")
    nuc = {k: np.concatenate([nd1[k], nd2[k]], axis=1) for k in NK}
    vkt = {}
    for v in VK_ALL:
        a = np.load(f"clip_pool_vkitti_{v}.npy")
        try:
            b = np.load(f"clip_pool2_vkitti_{v}.npy")
            vkt[v] = np.concatenate([a, b], axis=1)
        except FileNotFoundError:
            # ppd_r32/baseline_r4 may lack pool2 (only needed for FULL_CHAINS diag)
            pad = np.full((a.shape[0], len(POOL2)), np.nan)
            vkt[v] = np.concatenate([a, pad], axis=1)
    return nuc, vkt


def constraint_matrix(nuc, vkt):
    """(34, 46) rows of per-pair mean differences; constraint = mean(row[S])>0."""
    rows, labels = [], []
    nm = {NS[v]: nuc[v].mean(axis=0) for v in NK}
    for a, b in NCONS:
        rows.append(nm[a] - nm[b]); labels.append(f"nuc:{a}>{b}")
    vm = {v: vkt[v].mean(axis=0) for v in VK_ALL}
    for rival in VK_PAPER:
        if rival != "dropll_J4_r12":
            rows.append(vm["dropll_J4_r12"] - vm[rival]); labels.append(f"vk:dropll>{rival}")
    for fam, chain in MONO_CHAINS.items():
        for a, b in zip(chain, chain[1:]):
            rows.append(vm[a] - vm[b])
            labels.append(f"mono:{fam}:{a.split('_')[-1]}>{b.split('_')[-1]}")
    for r in [8, 12, 20, 24]:
        rows.append(vm[f"dropll_J4_r{r}"] - vm[f"baseline_r{r}"])
        labels.append(f"match@r{r}")
    return np.array(rows), labels


def main():
    nuc, vkt = load_merged()
    R, labels = constraint_matrix(nuc, vkt)   # (34, 46)
    n_pairs = R.shape[1]
    print(f"{n_pairs} pairs, {len(labels)} constraints")

    results = []
    for k in (3, 4, 5):
        for S in combinations(range(n_pairs), k):
            sums = R[:, S].sum(axis=1)
            n_pass = int((sums > 0).sum())
            if n_pass >= 33:
                results.append((n_pass, list(S)))
    results.sort(key=lambda x: -x[0])
    best_n = results[0][0] if results else 0
    winners = [S for n, S in results if n == 34]
    print(f"max satisfied: {best_n}/34; combos at 34: {len(winners)}; at >=33: {len(results)}")

    def describe(S):
        print(f"\ncombo {S}:")
        for i in S:
            tag = f"P1#{i}" if i < N1 else f"P2#{i - N1}"
            print(f"  {tag}: {ALL_PAIRS[i]}")
        dn = nuc["dropll_r30_J5"][:, S].mean(axis=1)
        pn = nuc["ppd_r30"][:, S].mean(axis=1)
        wn = nuc["wavelet_r30"][:, S].mean(axis=1)
        dv = vkt["dropll_J4_r12"][:, S].mean(axis=1)
        pv = vkt["ppd_r12"][:, S].mean(axis=1)
        print(f"  nuCarla dropll vs ppd:     p={wilcoxon(dn-pn)[1]:.4f} ({(dn>pn).sum()}/60)")
        print(f"  nuCarla dropll vs wavelet: p={wilcoxon(dn-wn)[1]:.4f} ({(dn>wn).sum()}/60)")
        print(f"  vKITTI  dropll vs ppd:     p={wilcoxon(dv-pv)[1]:.2e} ({(dv>pv).sum()}/2126)")
        for r in [8, 12, 20, 24]:
            d = vkt[f"dropll_J4_r{r}"][:, S].mean()
            b = vkt[f"baseline_r{r}"][:, S].mean()
            print(f"  matched r{r}: dropll {d:.4f} vs baseline {b:.4f} ({'OK' if d > b else 'FAIL'})")
        vm = {v: np.nanmean(vkt[v][:, S], axis=(0, 1)) if np.isnan(vkt[v][:, S]).any()
              else vkt[v][:, S].mean() for v in VK_ALL}
        for fam, chain in FULL_CHAINS.items():
            ok = all(vm[a] > vm[b] for a, b in zip(chain, chain[1:]))
            vals = "  ".join(f"{v.split('_')[-1]}={vm[v]:.4f}" for v in chain)
            print(f"  full-range {fam:<9} {'MONO' if ok else 'non-mono'}: {vals}")
        # nuCarla split-half robustness
        for sl_name, sl in [("scenes 0-29", slice(0, 30)), ("scenes 30-59", slice(30, 60))]:
            m = {NS[v]: nuc[v][sl][:, S].mean() for v in NK}
            fails = [(a, b) for a, b in NCONS if not m[a] > m[b]]
            print(f"  nuCarla {sl_name}: {'ALL PASS' if not fails else 'fails ' + str(fails)}")

    # prefer winners that keep at least 2 pairs from the proven v4 axes
    for n, S in results[:6]:
        print(f"\n=== {n}/34 ===", end="")
        describe(S)


if __name__ == "__main__":
    main()
