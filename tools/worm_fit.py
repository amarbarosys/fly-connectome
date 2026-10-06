"""Fit the worm touch reflex (agent/worm.py) as Wicks 1996 did, then test on behaviour it never saw.

Parameters per graph: one reversal E_i in [-48, 0] mV per presynaptic neuron (a continuous Dale sign)
and one global log10 chemical conductance. Antithetic evolution strategy; each candidate is one arm, so
a population is one batched step. Objective: anterior touch (ALM, AVM) raises AVA over AVB, posterior
(PLM) lowers it, margin MARGIN mV. Holdout: FLP (nose touch) and ASH (nociception) must raise it.
Real graph vs 2 degree-preserving shuffles fitted with the same budget.
Run: .venv/bin/python -I tools/worm_fit.py  -> .work/worm_fit.json
"""
import json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from agent import worm
from agent.shiu_shuffle import shuffle_edges

GENS_OVERRIDE = int(os.environ.get("WORM_GENS", 0))
POP, GENS, SECONDS, SIGMA, LR, MARGIN, AMP = 16, 60, 0.3, 0.3, 0.1, 0.5, 1e4
GENS = GENS_OVERRIDE or GENS
names, chem, gap, rev0 = worm.load(); n = len(names); ix = {k: i for i, k in enumerate(names)}
STIM = {'ant': ('ALML', 'ALMR', 'AVM'), 'post': ('PLML', 'PLMR'), 'flp': ('FLPL', 'FLPR'), 'ash': ('ASHL', 'ASHR')}
cols = ['rest'] + list(STIM)
I = np.zeros((1, n, len(cols)), np.float32)
for j, k in enumerate(STIM, 1):
    for nm in STIM[k]: I[0, ix[nm], j] = AMP

def graph(seed):
    if seed is None: return worm.dense(n, *chem), worm.dense(n, *gap)
    (cp, cq, cw), (gp, gq, gw) = chem, gap
    return worm.dense(n, cp, shuffle_edges(cp, cq, seed)[0], cw), worm.dense(n, gp, shuffle_edges(gp, gq, seed)[0], gw)

def evaluate(c, g, theta):
    """theta (K, n + 1) unconstrained -> per column AVA-AVB minus rest, (K, cols)."""
    e = worm.E_INH / (1 + np.exp(-theta[:, :n]))  # sigmoid into [-48, 0]
    gs = 10 ** np.clip(theta[:, n], -3, 0)  # chem scale 0.001..1 of Gs
    b = worm.WormBrain([c * s for s in gs], [g] * len(theta), e, len(cols))
    v = b.run(I, SECONDS)
    d = v[:, [ix['AVAL'], ix['AVAR']]].mean(1) - v[:, [ix['AVBL'], ix['AVBR']]].mean(1)
    return d - d[:, :1]

def loss(d):
    return np.maximum(0, MARGIN - d[:, 1]) + np.maximum(0, MARGIN + d[:, 2])  # train on ant, post only

def fit(seed, rng):
    c, g = graph(seed)
    theta = np.concatenate([np.where(rev0 < 0, 3.0, -3.0), [-1.0]])  # start: GABA-only signs, Gs x 0.1
    for gen in range(GENS):
        eps = rng.standard_normal((POP // 2, n + 1)); eps = np.concatenate([eps, -eps])
        L = loss(evaluate(c, g, theta + SIGMA * eps))
        rank = (np.argsort(np.argsort(L)) / (POP - 1) - .5)  # rank-shaped, low loss -> negative
        theta -= LR / SIGMA * (rank @ eps) / POP
    d = evaluate(c, g, theta[None])[0]
    return {'graph': 'real' if seed is None else f'shuffle{seed}', 'loss': float(loss(d[None])[0]),
            **{k: round(float(x), 4) for k, x in zip(cols[1:], d[1:])},
            'train_pass': bool(d[1] > 0 and d[2] < 0), 'holdout_pass': bool(d[3] > 0 and d[4] > 0),
            'log10_gs_scale': round(float(np.clip(theta[n], -3, 0)), 3), 'inhibitory_pre': int((theta[:n] > 0).sum())}

t = time.perf_counter(); rng = np.random.default_rng(0); out = []
for seed in (None, 1, 2):
    out.append(fit(seed, rng)); print(json.dumps(out[-1]), f'{time.perf_counter() - t:.0f}s', flush=True)
json.dump(out, open(os.path.join(ROOT, '.work/worm_fit.json'), 'w'), indent=1)
