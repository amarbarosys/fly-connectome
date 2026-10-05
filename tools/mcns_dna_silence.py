"""mcns Q4 (astra-progress): is the DNa01/02 null (no leg-MN lateralization at 150 Hz) the cancellation of their direct
ipsilateral excitation by the 2-hop ipsilateral inhibition (tools/mcns_dna_reach.py)? Intervention: silence the
inhibitory interneurons on that route, i.e. cells with a DNa01/02 input AND a net-inhibitory synapse onto a leg MN of
the same side (all outgoing weights of the silenced cells set to 0), and re-measure.

Conditions (one column each, 2 s, counts after a 200 ms onset): forward DNs 140 Hz, plus DNa01/02 left or right at
150 Hz, or none. Lateralization of a turn = (L - R MN spikes)/(L + R) minus the no-turn value; sign: left DNa -> left
MNs up. Arms: intact, route silenced, control (as many inhibitory cells with a leg-MN output, chosen at random from those
NOT on the route, silenced).
Gate (hypothesis): route-silenced left and right turns lateralize with the ipsilateral sign, |shift| >= 3x intact's
and >= 3x the control's.
usage: gpu-wait run --shared --vram 4 --priority 60 --timeout 1200 --env PYTHONPATH=$PWD -- .venv/bin/python tools/mcns_dna_silence.py
"""
import json
import sys

import numpy as np

from agent import mcns, shiu_g1
from agent.mcns_mapping import descending_name_sets, leg_motor_groups
from agent.shiu import ShiuGPU

data = mcns.graph()
pre, post, w = next(shiu_g1.arms(data, 'mcns'))[1]
n = data[4][0]
dn, _ = descending_name_sets()
mg, _ = leg_motor_groups()
mn = {s: np.concatenate([mg[k] for k in mg if k.endswith('_' + s)]) for s in 'LR'}
route = []
for s, dna in (('L', dn['turn_left']), ('R', dn['turn_right'])):
    first = np.unique(post[np.isin(pre, dna)])
    m = np.isin(pre, first) & np.isin(post, mn[s])
    net = np.bincount(np.searchsorted(first, pre[m]), w[m], minlength=len(first))
    route.append(first[net < 0])
route = np.unique(np.concatenate(route))
inh_to_mn = np.unique(pre[(w < 0) & np.isin(post, np.r_[mn['L'], mn['R']])])
pool = np.setdiff1d(inh_to_mn, np.r_[route, dn['turn_left'], dn['turn_right']])
ctrl = np.random.default_rng(5).choice(pool, min(len(route), len(pool)), replace=False)
COND = [None, 'left', 'right']
win = np.arange(2000, 20001, 18000)
cells = np.r_[mn['L'], mn['R']]
nl = len(mn['L'])


def measure(silenced):
    ww = np.where(np.isin(pre, silenced), 0, w) if len(silenced) else w
    brain = ShiuGPU(pre, post, ww, n=n, flies=len(COND), seed=7)
    for c, t in enumerate(COND):
        brain.stimulate(dn['forward'], 140, fly=c)
        if t:
            brain.stimulate(dn['turn_' + t], 150, fly=c)
    cg = np.full(n, -1, np.int32); cg[cells] = 0
    _, got = brain.run_groups(win, cg, np.concatenate([c * n + cells for c in range(len(COND))]).astype(np.int32))
    x = np.diff(got.astype(np.int64), axis=0).reshape(len(COND), len(cells))
    L, R = x[:, :nl].sum(1), x[:, nl:].sum(1)
    lat = (L - R) / np.maximum(L + R, 1)
    return {'L': L.tolist(), 'R': R.tolist(), 'shift_left': float(lat[1] - lat[0]), 'shift_right': float(lat[2] - lat[0])}


out = {'route_cells': int(len(route)), 'control_cells': int(len(ctrl)),
       'intact': measure([]), 'route_silenced': measure(route), 'control_silenced': measure(ctrl)}
print(json.dumps(out))
r, i, c = out['route_silenced'], out['intact'], out['control_silenced']
big = lambda a, b: abs(a) >= 3 * max(abs(b), 1e-3)
ok = [('ipsilateral sign after route silencing', r['shift_left'] > 0 and r['shift_right'] < 0),
      ('route shift >= 3x intact', big(r['shift_left'], i['shift_left']) and big(r['shift_right'], i['shift_right'])),
      ('route shift >= 3x control', big(r['shift_left'], c['shift_left']) and big(r['shift_right'], c['shift_right']))]
for k, v in ok:
    print(('PASS' if v else 'FAIL'), k)
if not all(v for _, v in ok):
    sys.exit('FAIL mcns DNa cancellation hypothesis')
print('PASS mcns DNa cancellation: silencing the 2-hop inhibitory route uncovers ipsilateral DNa lateralization')
