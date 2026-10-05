"""Dim light on the current visual path: does the Shiu FlyWire brain's lamina respond to a weak photoreceptor drive?

The old 'dim stripe broken' row (docs/plans/SENSORY_VALIDITY_GATES.md) is the legacy pooled-eye interface, which
injects per-eye brightness into L1-L3 directly and never reaches a first spike at dim contrast (0.139 of threshold 1).
Here the drive enters where light does: all R1-6 photoreceptors of the left eye at Poisson rates from dim to bright.
Readout: lamina L1, L2, L3 spikes of the left side, 0-200 ms, K repeats. Mutant: R1-6 output edges scaled to 0.
Gate: dim (10 Hz) lamina spikes > 0; spikes rise with rate; the mutant is silent at every rate.
usage: gpu-wait run --shared --vram 3 --priority 60 --timeout 600 --env PYTHONPATH=$PWD -- .venv/bin/python tools/shiu_dim_test.py
"""
import json
import sys

import numpy as np

from agent import shiu_g1
from agent.experiment import Experiment, cells

K, RATES, DIM = 4, (2, 5, 10, 20, 40, 150), 10

c = cells()
photo = np.flatnonzero((c.type == 'R1-6') & (c.side == 'left'))
lamina = np.flatnonzero(np.isin(c.type, ('L1', 'L2', 'L3')) & (c.side == 'left'))
_, (pre, post, w), _ = next(shiu_g1.arms(shiu_g1.flywire(), 'flywire'))
res = Experiment([(pre, post, w)], len(c.klass), {f'{hz}Hz': (photo, hz) for hz in RATES}, (2000,), {'lamina': lamina},
                 edges={'photo': lambda p, q, x: np.isin(p, photo)}, variants=[{}, {'photo': 0}], repeats=K, seed=17).run()
s = res['sums']['lamina'][0, 0]  # (variant, rate, repeat)
real, mut = s[0].mean(1), s[1].mean(1)
out = {'photoreceptors': int(len(photo)), 'lamina_cells': int(len(lamina)),
       'lamina_spikes_per_rate': dict(zip(map(str, RATES), real.tolist())), 'mutant': dict(zip(map(str, RATES), mut.tolist()))}
print(json.dumps(out))
ok = [(f'dim {DIM} Hz drives the lamina', real[RATES.index(DIM)] > 0),
      ('lamina spikes rise with rate', bool(np.all(np.diff(real) >= 0) and real[-1] > real[0])),
      ('mutant (R1-6 edges cut) silent', bool(np.all(mut == 0)))]
for k, v in ok:
    print(('PASS ' if v else 'FAIL ') + k)
if not all(v for _, v in ok):
    sys.exit('FAIL shiu dim light')
print('PASS shiu dim light: lamina responds from', RATES[int(np.argmax(real > 0))], 'Hz')
