"""Where the odour stops (FULL_FLY U2 S1a follow-up, measured only): forward DNs 80 Hz plus the vinegar ORNs of both
antennae at 150 Hz vs none, 500 ms, no body. Per group (ORNs, their glomeruli's PNs, KCs, MBONs, LH neurons, DNs) the
mean spikes per cell per 100 ms with and without odour, so the hop where the signal dies shows.
drive / drive0 (no forward DN drive): left antenna 150 Hz vs none (8 repeats), every cell's rate; per DN the net synaptic event rate sum_j r_j w_ij
(events/s) against the ~V_TH / (W_SYN TAU) = 5,091 /s a cell needs to reach threshold; DNa02 / DNa01 / DNp29 inputs by type.
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

SIDE, DRIVE = sys.argv[1] in ('side', 'side0'), sys.argv[1] in ('drive', 'drive0')
FWD = 0.0 if sys.argv[1] in ('drive0', 'side0') else FWD_HZ  # drive0: no forward DN drive
REPEATS, N = 16 if SIDE else 8, 500
seed = int(sys.argv[1 + (SIDE or DRIVE)])
seeds.claim(f'full_fly_smell_{sys.argv[1]}' if SIDE else f'full_fly_smell_{sys.argv[1]}' if DRIVE else 'full_fly_smell_trace', [seed])
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
if DRIVE:
    from agent.shiu import TAU, V_TH, W_SYN
    b, *_ = brain(cols=2 * REPEATS, seed=seed)
    for k in range(2 * REPEATS):
        b.stimulate(dn['forward'], FWD, fly=k, key=k)  # brain() already drives them at FWD_HZ: 0 Hz overrides it
        if k % 2:
            b.stimulate(c['smell', 'L'], sn.RMAX, fly=k, key=k)
    b.start([1], np.arange(2 * REPEATS * b.n, dtype=np.int32))
    x = b.advance(N * fl.K).astype(float).reshape(REPEATS, 2, b.n) * (1000 / N)  # Hz
    b.close()
    r_none, r_odr = x[:, 0].mean(0), x[:, 1].mean(0)
    np.save(fl.OUT / f'full-fly-smell-{sys.argv[1]}-{seed}-rates.npy', np.stack([r_none, r_odr]).astype(np.float32))
    pre, post, w, *_ = mcns.graph()
    need = V_TH / (W_SYN * TAU)
    D = groups['DN']
    net = lambda r: np.bincount(post, r[pre] * w, minlength=len(r))
    exc = lambda r: np.bincount(post, r[pre] * np.maximum(w, 0), minlength=len(r))
    n0, n1, e1 = net(r_none), net(r_odr), exc(r_odr)
    side = pd.read_feather(mcns.DATA / 'body-annotations-male-cns-v1.0-minconf-0.5.feather', columns=['status', 'somaSide'])
    side = side[side['status'].isin(['Traced', 'Anchor'])].somaSide.fillna('').to_numpy()
    out = {'need_events_per_s': need, 'dn_net_median_none': float(np.median(n0[D])), 'dn_net_median_odour': float(np.median(n1[D])),
           'dn_frac_net_above_need_odour': float((n1[D] > need).mean()),
           'dn_odour_change_median': float(np.median(n1[D] - n0[D])), 'dn_odour_change_p99': float(np.percentile(np.abs(n1[D] - n0[D]), 99))}
    for t in ('PS059', 'AOTU019', 'VES104', 'DNge046'):
        out[f'rate_{t}'] = round(float(r_none[ty.to_numpy() == t].mean()), 2)
    for t in ('DNa02', 'DNa01', 'DNp29'):
        for s_ in 'LR':
            i = int(np.flatnonzero((ty.to_numpy() == t) & (side == s_))[0])
            e = post == i
            by = pd.DataFrame({'t': ty.to_numpy()[pre[e]], 'd': (r_odr - r_none)[pre[e]] * w[e], 'n': r_odr[pre[e]] * w[e]}).groupby('t').sum()
            out[f'{t}_{s_}'] = {'rate_none': round(float(r_none[i]), 2), 'rate_odour': round(float(r_odr[i]), 2), 'net_none': round(float(n0[i])), 'net_odour': round(float(n1[i])),
                                'exc_odour': round(float(e1[i])), 'top_change': [(k, round(float(v))) for k, v in by.d.sort_values(key=abs, ascending=False).head(5).items()],
                                'top_net': [(k, round(float(v))) for k, v in by.n.sort_values(key=abs, ascending=False).head(5).items()]}
    print(json.dumps(out))
    (fl.OUT / f'full-fly-smell-{sys.argv[1]}-{seed}.json').write_text(json.dumps(out))
    sys.exit(0)
if SIDE:
    cs = sn.cells(swap=True)
    arms = [(sd, sw) for _ in range(REPEATS) for sw in (False, True) for sd in 'LR']
    b, *_ = brain(cols=len(arms), seed=seed)
    for k, (sd, sw) in enumerate(arms):
        b.stimulate(dn['forward'], FWD, fly=k, key=k)
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
    for t_ in ('DNa02', 'DNa01', 'DNp29'):
        for s_ in 'LR':
            j = int(np.flatnonzero((ty.to_numpy()[D] == t_) & (side[D] == s_))[0])
            by[f'{t_}_{s_}'] = {'odourL_Hz': round(float(x[:, 0, 0, j].mean()) * 10, 2), 'odourR_Hz': round(float(x[:, 0, 1, j].mean()) * 10, 2),
                                't_real': round(float(t[0, j]), 1), 't_swap': round(float(t[1, j]), 1)}
    out = {'groups': by, 'active_dns': int(act.sum()), 'abs_t_ge_3_real': int((np.abs(t[0]) >= 3).sum()), 'abs_t_ge_3_swap': int((np.abs(t[1]) >= 3).sum()),
           'real_vs_swap_corr': r, 'top': [(ty[D[i]], side[D[i]], round(float(d[:, 0, i].mean()), 3), round(float(t[0, i]), 1), round(float(t[1, i]), 1)) for i in top]}
    print(json.dumps(out))
    (fl.OUT / f'full-fly-smell-{sys.argv[1]}-{seed}.json').write_text(json.dumps(out))
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
