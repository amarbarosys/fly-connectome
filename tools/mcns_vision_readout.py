"""Fly vision steering, step 1 (body-free): a readout from the male CNS descending neurons to a turn command, fitted on
rendered discs at TRAINING positions only; the closed loop (agent/body_engine/mcns_link.py steer) tests it at held-out
positions on a new seed. The leg-MN readout of unit 3b cannot do this: it was fitted on forward-DN drive and the discs' DNs
do not lateralize the MNs (mcns_link vision, 9ed333f, NEGATIVE).

Conditions (one column each, K repeats, 1 s): forward DNs 80 Hz (as in the loop) plus an image through the lamina input
(agent/fly_vision.py LAMINA): blank, or a 40 deg dark disc at az +-40, +-55, +-85, +-100 (training; +-70 is held out).
Teacher per 100 ms window after 200 ms: T = +1 for a left disc (az > 0), -1 right, 0 blank: turn AWAY from the looming
side (escape). T > 0 raises the left drive, which turns the fly right (3b: left command -51.8 deg).
Fit: ridge on z-scored per-(type, side) DN counts of the 471 DN types with both a left and a right cell (1,299 cells), each
sample joined by its mirror (left and right sums swapped, T negated): the readout is antisymmetric, so the weaker left eye
(fewer placed L3, 6a71943) cannot bias it. Leave-one-azimuth-out (the held-out condition's mirror is held out too).
Gates: held-out T sign right in every disc condition; held-out R^2 >= 0.5; held-out |T| left / right discs in 0.5-2;
mutant: condition-permuted teacher R^2 < 0.3. (Before: all 1,316 DNs, no mirror: R^2 0.20, left |T| 0.19 vs right 0.27.)
usage: gpu-wait run --shared --vram 4 --priority 60 --timeout 900 --env PYTHONPATH=$PWD -- .venv/bin/python tools/mcns_vision_readout.py
"""
import json
import sys
from pathlib import Path

import numpy as np

from agent import fly_vision as fv, mcns, shiu_g1
from agent.mcns_mapping import annotation_table, descending_name_sets
from agent.shiu import ShiuGPU

READOUT = Path('.work/agent/mcns/vision_readout.npz')
K, FWD_HZ, AZ = 16, 80, (40, -40, 55, -55, 85, -85, 100, -100)
COND = [None] + list(AZ)
data = mcns.graph()
n = data[4][0]
a = annotation_table()
dn = np.flatnonzero(a.superclass.fillna('').str.startswith('descending').to_numpy())
typ, side = a.type.fillna('').to_numpy()[dn], a.somaSide.fillna('').str[:1].str.upper().to_numpy()[dn]
pairs = sorted(t for t in set(typ) if t and ((typ == t) & (side == 'L')).any() and ((typ == t) & (side == 'R')).any())
keep = np.isin(typ, pairs) & np.isin(side, ['L', 'R'])
dn, typ, side = dn[keep], typ[keep], side[keep]
G = np.zeros((len(dn), 2 * len(pairs)))  # cell -> feature: (type, L) first, then (type, R)
G[np.arange(len(dn)), np.searchsorted(pairs, typ) + len(pairs) * (side == 'R')] = 1
fwd, _ = descending_name_sets()
ids, az, el = fv.receptors(fv.LAMINA)
brain = ShiuGPU(*next(shiu_g1.arms(data, 'mcns'))[1], n=n, flies=len(COND) * K, seed=31)
for c, a in enumerate(COND):
    img = np.full((90, 180), fv.BG) if a is None else fv.disc(a)
    for r in range(K):
        col = c * K + r
        brain.stimulate(fwd['forward'], FWD_HZ, fly=col, key=col)
        for cells, hz in fv.drive(img, ids, az, el):
            brain.stimulate(cells, hz, fly=col, key=col)
win = np.arange(2000, 10001, 1000)  # 200 ms onset, then 8 windows of 100 ms
cg = np.full(n, -1, np.int32); cg[dn] = 0
_, got = brain.run_groups(win, cg, np.concatenate([f * n + dn for f in range(len(COND) * K)]).astype(np.int32))
x = np.diff(got.astype(np.int64), axis=0).reshape(len(win) - 1, len(COND), K, len(dn)).astype(float) @ G  # (w, cond, rep, feature)
mirror = lambda X: np.concatenate([X[..., len(pairs):], X[..., :len(pairs)]], -1)
np.save(READOUT.with_suffix('.x.npy'), x)
teach = np.array([0.0 if a is None else np.sign(a) for a in COND])


def fit(X, y, lam=30.0):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    return mu, sd, y.mean(), np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))


def held_out(t):
    """leave-one-azimuth-out (blank always in training): predicted T per held-out condition, all windows and repeats."""
    pred = {}
    for c, a in enumerate(COND):
        if a is None:
            continue
        tr = [k for k in range(len(COND)) if k != c]
        X = x[:, tr].reshape(-1, x.shape[-1]); y = np.broadcast_to(t[tr][None, :, None], (x.shape[0], len(tr), K)).ravel()
        mu, sd, y0, w = fit(np.r_[X, mirror(X)], np.r_[y, -y])
        pred[a] = ((x[:, c].reshape(-1, x.shape[-1]) - mu) / sd) @ w + y0
    return pred


def r2(pred, t):
    y = np.concatenate([np.full(len(p), t[COND.index(a)]) for a, p in pred.items()]); p = np.concatenate(list(pred.values()))
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


p = held_out(teach)
perm = teach[np.random.default_rng(3).permutation(len(COND))]
q = held_out(perm)
signs = {str(a): bool(np.sign(v.mean()) == np.sign(a)) for a, v in p.items()}
side_ratio = float(np.mean([abs(v.mean()) for a, v in p.items() if a > 0]) / np.mean([abs(v.mean()) for a, v in p.items() if a < 0]))
out = {'dn_spikes': {str(a): float(x[:, c].sum((0, 2)).mean() / K) for c, a in enumerate(COND)}, 'held_out_mean_T': {str(a): float(v.mean()) for a, v in p.items()},
       'r2': r2(p, teach), 'mutant_permuted_r2': r2(q, perm), 'signs': signs, 'side_ratio_left_right': side_ratio}
print(json.dumps(out))
X, y = x.reshape(-1, x.shape[-1]), np.broadcast_to(teach[None, :, None], (x.shape[0], len(COND), K)).ravel()
mu, sd, y0, w = fit(np.r_[X, mirror(X)], np.r_[y, -y])
wc = G @ (w / sd)  # per-cell weights on raw counts (mcns_link applies ((x - mu) / sd) @ w + y0 per cell): mu 0, sd 1
READOUT.parent.mkdir(parents=True, exist_ok=True)
np.savez(READOUT, dn=dn, mu=np.zeros(len(dn)), sd=np.ones(len(dn)), y0=y0 - (mu / sd) @ w, w=wc)
ok = [('G1 held-out turn sign right at every azimuth', all(signs.values())), ('G2 held-out R^2 >= 0.5', out['r2'] >= 0.5),
      ('G3 held-out |T| left / right discs in 0.5-2', 0.5 <= side_ratio <= 2), ('M1 permuted-teacher R^2 < 0.3', out['mutant_permuted_r2'] < 0.3)]
for k, v in ok:
    print(('PASS ' if v else 'FAIL ') + k)
if not all(v for _, v in ok):
    sys.exit('FAIL mcns vision readout')
print('PASS mcns vision readout')
