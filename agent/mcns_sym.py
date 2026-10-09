"""FULL_FLY U1b: complete the male CNS left visual path, which is under-reconstructed (left L1 inputs 101k vs 116k, LPLC2
82k vs 111k; FlyWire the same side; Schlegel et al. 2024: left-right completion rates differ). Two passes, both on the
left weights of existing edges only (no new edges, left topology kept), scope = visual superclasses:
  A structure: per (pre type, post type) the left synapses per post cell raised to the right's, ratio clipped [1, 2].
  B activity: symmetric spontaneous drive (all R1-6 + L1-L3, both eyes, never a test stimulus) on the GPU; left inputs of
    each visual type firing under 0.95 of its right homolog scaled by sqrt(rate ratio), clipped [1, 1.5], <= 5 rounds.
Output .work/mcns/graph-sym-t5.npz, read by mcns.graph(sym=True).
  python -m agent.mcns_sym check                      A + G1 (CPU)
  gpu-wait run --shared --vram 8 --preemptible --timeout 1800 --env PYTHONPATH=$PWD -- .venv/bin/python -m agent.mcns_sym grow    B + G2
"""
import sys

import numpy as np
import pandas as pd

from agent import mcns

SCOPE = ('ol_intrinsic', 'visual_projection', 'ol_sensory', 'visual_centrifugal')
OUT = mcns.DATA / 'graph-sym-t5.npz'
DRIVE_HZ, ROUNDS, MIN_SPIKES = 40.0, 5, 100


def cells():
    """per neuron: type code (0 = untyped), side (0 L, 1 R, -1 other), in visual scope; graph() order."""
    a = pd.read_feather(mcns.DATA / 'body-annotations-male-cns-v1.0-minconf-0.5.feather', columns=['status', 'type', 'superclass', 'somaSide'])
    a = a[a['status'].isin(['Traced', 'Anchor'])].reset_index(drop=True)
    names, t = np.unique(a['type'].fillna('').to_numpy(), return_inverse=True)  # '' sorts first: code 0
    side = a['somaSide'].fillna('').str[:1].map({'L': 0, 'R': 1}).fillna(-1).to_numpy(np.int64)
    return t.astype(np.int64), side, a['superclass'].isin(SCOPE).to_numpy() & (side >= 0) & (t > 0), names


def pairs(pre, post, t, side, scope):
    """typed in-scope edges: their indices, pair id per edge (pre type, post type; sparse, only pairs that occur), post side,
    and cells per (post type of each pair, side): (P, 2)."""
    e = np.flatnonzero(scope[post] & (t[pre] > 0))
    nt = int(t.max()) + 1
    up, pid = np.unique(t[pre[e]] * nt + t[post[e]], return_inverse=True)
    n = np.bincount(t[scope] * 2 + side[scope], minlength=nt * 2).reshape(nt, 2)
    return e, pid, side[post[e]], n[up % nt]


def per_cell(w, e, pid, s, n):
    """(P, 2): synapses per post cell for each pair and side."""
    syn = np.bincount(pid * 2 + s, np.abs(w[e]).astype(np.float64), minlength=len(n) * 2).reshape(-1, 2)
    return syn / np.maximum(n, 1)


def complete(w, e, pid, s, n):
    """A, in place on w: left edges of each pair scaled by r = right/left synapses per cell, clipped [1, 2]; returns r (P,)."""
    per = per_cell(w, e, pid, s, n)
    r = np.where((per[:, 0] > 0) & (per[:, 1] > 0), np.clip(per[:, 1] / np.maximum(per[:, 0], 1e-9), 1, 2), 1)
    left = s == 0
    w[e[left]] = np.round(w[e[left]] * r[pid[left]]).astype(w.dtype)
    return r


def check():
    pre, post, w, *_ = mcns.graph()
    t, side, scope, _ = cells()
    w0 = w.copy()
    e, pid, s, n = pairs(pre, post, t, side, scope)
    r = complete(w, e, pid, s, n)
    mid = (r > 1) & (r < 2)  # completed, not capped
    after = per_cell(w, e, pid, s, n)
    dev = np.abs(after[mid, 0] / after[mid, 1] - 1)
    out = ~(scope[post] & (side[post] == 0))
    ok = [(f"completed pairs {int(mid.sum())} (capped at 2: {int((r == 2).sum())}): left per cell within 10 % of right, max dev {dev.max():.3f}", dev.max() <= 0.10),
          (f"edges outside the left visual scope byte-identical ({int(out.sum())})", np.array_equal(w[out], w0[out])),
          (f"edge count unchanged {len(w)}", len(w) == len(w0))]
    sc = scope[post] & (side[post] == 0)
    print(f"left visual inputs: {np.abs(w0[sc]).sum() / 1e6:.2f}M -> {np.abs(w[sc]).sum() / 1e6:.2f}M synapses; right {np.abs(w[scope[post] & (side[post] == 1)]).sum() / 1e6:.2f}M")
    for k, v in ok:
        print(('PASS ' if v else 'FAIL ') + k)
    if not all(v for _, v in ok):
        sys.exit('FAIL G1 structure')
    np.savez(OUT, pre=pre.astype(np.int32), post=post.astype(np.int32), w=w.astype(np.int32))
    print(f'PASS G1 structure -> {OUT} (A only; grow adds B)')


def grow(flies=4, seed=61):
    from agent import fly_vision as fv, shiu_g1
    from agent.shiu import ShiuGPU
    z = np.load(OUT)
    pre, post, w = z['pre'].astype(np.int64), z['post'].astype(np.int64), z['w'].astype(np.int64)
    names, kc_orn = mcns.graph()[3:]
    t, side, scope, tn = cells()
    nt = int(t.max()) + 1
    cg = np.where(scope, t * 2 + side, -1).astype(np.int32)
    ids, _, _ = fv.receptors(fv.LAMINA)
    n = len(t)
    win = np.array([300, 5300])  # 30 ms settle, then 0.5 s
    for rnd in range(ROUNDS + 1):
        b = ShiuGPU(*next(shiu_g1.arms((pre, post, w, names, kc_orn), 'mcns'))[1], n=n, flies=flies, seed=seed + rnd)
        for c in range(flies):
            b.stimulate(ids, DRIVE_HZ, fly=c, key=c)
        sums, _ = b.run_groups(win, cg)
        s = np.diff(np.asarray(sums, np.int64), axis=0)[0].sum(1)  # spikes per group over all flies
        s = np.pad(s, (0, nt * 2 - len(s))).reshape(nt, 2).astype(np.float64)
        cnt = np.bincount(t[scope] * 2 + side[scope], minlength=nt * 2).reshape(nt, 2)
        rate = s / np.maximum(cnt, 1)
        live = (s >= MIN_SPIKES).all(1)
        ratio = np.where(live, rate[:, 0] / np.maximum(rate[:, 1], 1e-9), 1)
        within = np.abs(ratio[live] - 1) <= 0.05
        print(f"round {rnd}: {int(live.sum())} visual types with >= {MIN_SPIKES} spikes per side; L/R within 5 %: {int(within.sum())}/{int(live.sum())}; "
              f"median |L/R-1| {np.median(np.abs(ratio[live] - 1)):.3f}; LC4 {ratio[list(tn).index('LC4')]:.2f} LPLC2 {ratio[list(tn).index('LPLC2')]:.2f}", flush=True)
        low = live & (ratio < 0.95)
        if rnd == ROUNDS or not low.any():
            break
        g = np.clip(np.sqrt(1 / np.maximum(ratio, 1e-3)), 1, 1.5)
        e = scope[post] & (side[post] == 0) & low[t[post]]
        w[e] = np.round(w[e] * g[t[post[e]]]).astype(w.dtype)
    np.savez(OUT, pre=pre.astype(np.int32), post=post.astype(np.int32), w=w.astype(np.int32))
    ok = within.all()
    print(('PASS' if ok else 'FAIL') + f" G2 activity: every live visual type L/R within 5 % ({int(within.sum())}/{int(live.sum())}) -> {OUT}")
    if not ok:
        sys.exit('FAIL G2 activity')


if __name__ == '__main__':
    {'check': check, 'grow': grow}[sys.argv[1]]()
