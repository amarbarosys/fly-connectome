"""Where the odour stops (FULL_FLY U2 S1a follow-up, measured only): forward DNs 80 Hz plus the vinegar ORNs of both
antennae at 150 Hz vs none, 500 ms, no body. Per group (ORNs, their glomeruli's PNs, KCs, MBONs, LH neurons, DNs) the
mean spikes per cell per 100 ms with and without odour, so the hop where the signal dies shows.
  gpu-wait run --shared --vram 8 --preemptible --timeout 1200 --env PYTHONPATH=$PWD -- .venv/bin/python tools/full_fly_smell_trace.py SEED"""
import json
import sys

import numpy as np
import pandas as pd

from agent import mcns, senses as sn, seeds
from agent.body_engine import fly_link as fl
from agent.body_engine.mcns_link import FWD_HZ, brain
from agent.mcns_mapping import descending_name_sets

REPEATS, N = 8, 500
seed = int(sys.argv[1])
seeds.claim('full_fly_smell_trace', [seed])
a = pd.read_feather(mcns.DATA / 'body-annotations-male-cns-v1.0-minconf-0.5.feather', columns=['status', 'superclass', 'type'])
a = a[a['status'].isin(['Traced', 'Anchor'])].reset_index(drop=True)
ty, sup = pd.Series(a['type'].fillna('')), a['superclass'].fillna('')
c = sn.cells()
glom = '|'.join(sn.VINEGAR)
groups = {'ORN': np.r_[c['smell', 'L'], c['smell', 'R']],
          'PN': np.flatnonzero(ty.str.match(rf'^({glom})_.*PN').to_numpy()),
          'KC': np.flatnonzero(ty.str.match(r'^KC').to_numpy()),
          'MBON': np.flatnonzero(ty.str.match(r'^MBON').to_numpy()),
          'LH': np.flatnonzero(ty.str.match(r'^LH').to_numpy()),
          'DN': np.flatnonzero(sup.str.startswith('descending').to_numpy())}
read = np.concatenate(list(groups.values()))
dn, _ = descending_name_sets()
cols = 2 * REPEATS
b, *_ = brain(cols=cols, seed=seed)
for k in range(cols):
    b.stimulate(dn['forward'], FWD_HZ, fly=k, key=k)
    if k % 2:
        b.stimulate(groups['ORN'], sn.RMAX, fly=k, key=k)
b.start([1], np.concatenate([k * b.n + read for k in range(cols)]).astype(np.int32))
x = b.advance(N * fl.K).astype(float).reshape(cols, -1) * (100 / N)
b.close()
out, i = {}, 0
for g, ids in groups.items():
    y = x[:, i:i + len(ids)]
    i += len(ids)
    on, off = y[1::2].mean(0), y[::2].mean(0)
    out[g] = {'cells': len(ids), 'odour': round(float(on.mean()), 3), 'none': round(float(off.mean()), 3),
              'cells_up_2x': int(((on > 2 * off + 0.1)).sum())}
    print(g, out[g])
(fl.OUT / f'full-fly-smell-trace-{seed}.json').write_text(json.dumps(out))
