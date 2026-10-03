"""Do Shiu descending neurons carry the SIDE of a visual stimulus, and does real wiring carry it better?

Odour gives DNs no side (max |z| 3.0 over 1,291 DNs). Vision is wired by side, so this asks the same
question for one eye: all R1-6 photoreceptors of one side ('eye'), or the looming projection
neurons LC4 + LPLC2 of one side ('loom'), at 150 Hz. All 11 arms (real, 5 shuffled, 5 block) side
by side in one resident worker; K columns per (stimulus, side). Per arm, stimulus and window:
DNs with |z| > 4 for left vs right, max |z|, and leave-one-out nearest-centroid side decode from
the DN count vector (chance 0.5).
"""
import csv
import json

import numpy as np

from agent import shiu_g1
from agent.neural_connectome import ANNOTATIONS, ROOT
from agent.shiu import ShiuLive
from agent.shiu_loop import combine
from agent.verify_shiu import _reference_graph

K, HZ = 8, 150.
STIMS = {'eye': ('R1-6',), 'loom': ('LC4', 'LPLC2')}
WINDOWS = (300, 500, 2000)  # 30 ms, 50 ms, 200 ms


def cells():
    index, src, dn, types = _reference_graph()[3], {}, [], {}
    for r in csv.DictReader(open(ANNOTATIONS), delimiter='\t'):
        i = index.get(int(r['root_id']))
        if i is None or r['side'] not in ('left', 'right'):
            continue
        for stim, kinds in STIMS.items():
            if r['cell_type'] in kinds:
                src.setdefault((stim, r['side']), []).append(i)
        if r['super_class'] == 'descending':
            dn.append(i)
            types[i] = f"{r['cell_type']}-{r['side'][0]}"
    return src, dn, types


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
    data = shiu_g1.flywire()
    n = len(data[3])
    src, dn, types = cells()
    names, graphs = zip(*[(name, g) for name, g, _ in shiu_g1.arms(data, 'flywire')])
    arms, cols = len(graphs), len(STIMS) * 2 * K
    layout = [(stim, side) for stim in STIMS for side in ('left', 'right') for _ in range(K)]
    brain = ShiuLive(*combine(list(graphs), n), n=arms * n, flies=cols, seed=41)
    for c, key in enumerate(layout):
        for a in range(arms):
            brain.stimulate([a * n + i for i in src[key]], HZ, fly=c)
    j = np.arange(arms * cols)  # flat = arm * cols + c
    readout = ((j % cols) * arms * n + (j // cols) * n + np.asarray(dn)[:, None]).reshape(-1)
    brain.start(WINDOWS, readout)
    got = brain.sniff(np.full(len(brain._stim_arrays()[1]), HZ * brain.dt, np.float32))
    brain.close()
    got = got.reshape(len(WINDOWS), len(dn), arms, cols).astype(float)
    window = {'30-50ms': got[1] - got[0], '0-200ms': got[2]}
    out = {'sources': {f'{s}-{side}': len(v) for (s, side), v in src.items()}, 'dns': len(dn), 'K': K, 'arms': {}}
    for a, name in enumerate(names):
        out['arms'][name] = {}
        for stim in STIMS:
            cs = [c for c, (s, _) in enumerate(layout) if s == stim]
            y = np.array([layout[c][1] == 'right' for c in cs], int)
            for wname, x in window.items():
                v = x[:, a, cs]
                L, R = v[:, y == 0], v[:, y == 1]
                z = (L.mean(1) - R.mean(1)) / np.sqrt(L.var(1, ddof=1) / K + R.var(1, ddof=1) / K + 1 / K)  # floor: one spike of noise
                out['arms'][name][f'{stim} {wname}'] = {
                    'dn_spikes': float(v.sum(0).mean()), 'dns_z4': int((np.abs(z) > 4).sum()),
                    'max_abs_z': float(np.abs(z).max()), 'side_decode': decode(v, y),
                    'top': [[types[dn[k]], round(float(z[k]), 1)] for k in np.argsort(-np.abs(z))[:8]]}
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
