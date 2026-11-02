"""Which DNs carry the turn bias: T contribution w x rate / sd per DN type, arm 'all' vs 'none'. python tools/turn_bias_dn.py SEED (PYTHONPATH=.)"""
import sys

import numpy as np

from agent.body_engine import fly_link as fl
from agent.mcns_mapping import annotation_table

z = np.load(fl.OUT / f'hybrid-dn-{sys.argv[1]}.npz')
v = np.load('.work/agent/mcns/vision_readout.npz')
a = annotation_table().iloc[v['dn']]
c = v['w'] / v['sd']
m = {k: z['dn'][z['arms'] == k].mean(0) for k in np.unique(z['arms'])}
print({k: round(float(c @ x + v['y0']), 3) for k, x in m.items()})
n = m['none'] if 'none' in m else 0 * c
for k in ('all', 'no-legs'):
    d = c * (m[k] - n)
    print(k, 'dT', round(float(d.sum()), 3), 'neg sum', round(float(d[d < 0].sum()), 3), 'pos sum', round(float(d[d > 0].sum()), 3))
    for i in np.argsort(-np.abs(d))[:10]:
        print('  ', a.type.iloc[i], str(a.instance.iloc[i])[-3:] if 'instance' in a else '', 'dT %.3f rate %.2f none %.2f w %.3f' % (d[i], m[k][i], (n if np.ndim(n) == 0 else n[i]), v['w'][i]))
