"""Do Shiu descending neurons carry the SIDE of a visual stimulus, and does real wiring carry it better?

Odour gives DNs no side (max |z| 3.0 over 1,291 DNs). Vision is wired by side, so this asks the same
question for one eye: all R1-6 photoreceptors of one side ('eye'), or the looming projection
neurons LC4 + LPLC2 of one side ('loom'), at 150 Hz. All 11 arms (real, 5 shuffled, 5 block) side
by side (agent/experiment.py); K repeats per (stimulus, side). Per arm, stimulus and window:
DNs with |z| > 4 for left vs right, max |z|, and leave-one-out nearest-centroid side decode from
the DN count vector (chance 0.5).
"""
import json

import numpy as np

from agent import shiu_g1
from agent.neural_connectome import ROOT
from agent.experiment import Experiment, cells

K, HZ = 8, 150.
STIMS = {'eye': ('R1-6',), 'loom': ('LC4', 'LPLC2')}
WINDOWS = (300, 500, 2000)  # 30 ms, 50 ms, 200 ms


def decode(x, y):
    """Leave-one-out nearest centroid (cosine) on columns of x (dn, m); labels y."""
    u = x / (np.linalg.norm(x, axis=0, keepdims=True) + 1e-9)
    hit = 0
    for k in range(len(y)):
        keep = np.arange(len(y)) != k
        c = [u[:, keep & (y == s)].mean(1) for s in (0, 1)]
        hit += int(np.argmax([c[0] @ u[:, k], c[1] @ u[:, k]]) == y[k])
    return hit / len(y)


def main():
    c = cells()
    side = {(stim, sd): np.flatnonzero(np.isin(c.type, kinds) & (c.side == sd)) for stim, kinds in STIMS.items() for sd in ('left', 'right')}
    dn = np.flatnonzero((c.super == 'descending') & np.isin(c.side, ('left', 'right')))
    types = np.array([f'{t}-{sd[0]}' for t, sd in zip(c.type[dn], c.side[dn])])
    data = shiu_g1.flywire()
    names, graphs = zip(*[(name, g) for name, g, _ in shiu_g1.arms(data, 'flywire')])
    res = Experiment(list(graphs), len(c.klass), {f'{stim}-{sd}': (ids, HZ) for (stim, sd), ids in side.items()},
                     WINDOWS, {'dn': dn}, repeats=K, per_cell=('dn',), seed=41).run()
    got = res['cells']['dn'][:, :, :, 0]  # (window, dn, arm, stimulus, repeat), stimuli eye-l, eye-r, loom-l, loom-r
    window = {'30-50ms': got[1] - got[0], '0-200ms': got[2]}
    out = {'sources': {f'{s}-{sd}': len(v) for (s, sd), v in side.items()}, 'dns': len(dn), 'K': K, 'arms': {}}
    for a, name in enumerate(names):
        out['arms'][name] = {}
        for k, stim in enumerate(STIMS):
            y = np.repeat([0, 1], K)
            for wname, x in window.items():
                v = x[:, a, 2 * k:2 * k + 2].reshape(len(dn), 2 * K)  # left repeats, then right
                L, R = v[:, y == 0], v[:, y == 1]
                z = (L.mean(1) - R.mean(1)) / np.sqrt(L.var(1, ddof=1) / K + R.var(1, ddof=1) / K + 1 / K)  # floor: one spike of noise
                out['arms'][name][f'{stim} {wname}'] = {
                    'dn_spikes': float(v.sum(0).mean()), 'dns_z4': int((np.abs(z) > 4).sum()),
                    'max_abs_z': float(np.abs(z).max()), 'side_decode': decode(v, y),
                    'top': [[str(types[k2]), round(float(z[k2]), 1)] for k2 in np.argsort(-np.abs(z))[:8]]}
    for stim in STIMS:
        for wname in window:
            key = f'{stim} {wname}'
            print(key)
            for name in names:
                r = out['arms'][name][key]
                print(f"  {name:20s} DN {r['dn_spikes']:8.1f}  |z|>4 {r['dns_z4']:4d}  max|z| {r['max_abs_z']:6.1f}  decode {r['side_decode']:.2f}")
    (ROOT / '.work/agent/shiu/visual-side.json').write_text(json.dumps(out, indent=2) + '\n')


if __name__ == '__main__':
    main()
