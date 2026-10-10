"""Where the odour stops (FULL_FLY U2 S1a follow-up, measured only): forward DNs 80 Hz plus the vinegar ORNs of both
antennae at 150 Hz vs none, 500 ms, no body. Per group (ORNs, their glomeruli's PNs, KCs, MBONs, LH neurons, DNs) the
mean spikes per cell per 100 ms with and without odour, so the hop where the signal dies shows.
side: one antenna at 150 Hz, left vs right, and the side-swapped cells; per DN the left - right paired t over 16
repeats, and the correlation of real vs swapped differences across DNs (side-driven: negative).
  gpu-wait run --shared --vram 8 --preemptible --timeout 1200 --env PYTHONPATH=$PWD -- .venv/bin/python tools/full_fly_smell_trace.py [side] SEED"""
import json
import sys

import numpy as np
import pandas as pd

from agent import mcns, senses as sn, seeds
from agent.body_engine import fly_link as fl
from agent.body_engine.mcns_link import FWD_HZ, brain
from agent.mcns_mapping import descending_name_sets

SIDE = sys.argv[1] == 'side'
REPEATS, N = 16 if SIDE else 8, 500
seed = int(sys.argv[1 + SIDE])
seeds.claim('full_fly_smell_side' if SIDE else 'full_fly_smell_trace', [seed])
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
if SIDE:
    cs = sn.cells(swap=True)
    arms = [(sd, sw) for _ in range(REPEATS) for sw in (False, True) for sd in 'LR']
    b, *_ = brain(cols=len(arms), seed=seed)
    for k, (sd, sw) in enumerate(arms):
        b.stimulate(dn['forward'], FWD_HZ, fly=k, key=k)
        b.stimulate((cs if sw else c)['smell', sd], sn.RMAX, fly=k, key=k)
    D = np.concatenate([groups[g] for g in ('PN', 'LH', 'MBON', 'DN')])
    b.start([1], np.concatenate([k * b.n + D for k in range(len(arms))]).astype(np.int32))
    x = b.advance(N * fl.K).astype(float).reshape(REPEATS, 2, 2, -1) * (100 / N)  # (repeat, swap, side, DN)
    b.close()
    d = x[:, :, 0] - x[:, :, 1]  # odour-left minus odour-right, (repeat, swap, DN)
    t = d.mean(0) / (d.std(0, ddof=1) / np.sqrt(REPEATS) + 1e-9)
    act = (x.mean((0, 1, 2)) > 0.05)
    r = float(np.corrcoef(d[:, 0].mean(0)[act], d[:, 1].mean(0)[act])[0, 1])
    side = pd.read_feather(mcns.DATA / 'body-annotations-male-cns-v1.0-minconf-0.5.feather', columns=['status', 'somaSide'])
    side = side[side['status'].isin(['Traced', 'Anchor'])].somaSide.fillna('').to_numpy()
    top = np.argsort(-np.abs(t[0]))[:12]
    by, i0 = {}, 0
    for g in ('PN', 'LH', 'MBON', 'DN'):
        sl, i0 = slice(i0, i0 + len(groups[g])), i0 + len(groups[g])
        a_ = act[sl]
        by[g] = {'active': int(a_.sum()), 't3_real': int((np.abs(t[0, sl]) >= 3).sum()), 't3_swap': int((np.abs(t[1, sl]) >= 3).sum()),
                 'corr': round(float(np.corrcoef(d[:, 0].mean(0)[sl][a_], d[:, 1].mean(0)[sl][a_])[0, 1]), 3) if a_.sum() > 2 else None}
    print(json.dumps(by))
    out = {'groups': by, 'active_dns': int(act.sum()), 'abs_t_ge_3_real': int((np.abs(t[0]) >= 3).sum()), 'abs_t_ge_3_swap': int((np.abs(t[1]) >= 3).sum()),
           'real_vs_swap_corr': r, 'top': [(ty[D[i]], side[D[i]], round(float(d[:, 0, i].mean()), 3), round(float(t[0, i]), 1), round(float(t[1, i]), 1)) for i in top]}
    print(json.dumps(out))
    (fl.OUT / f'full-fly-smell-side-{seed}.json').write_text(json.dumps(out))
    sys.exit(0)
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
