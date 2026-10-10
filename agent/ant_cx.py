"""Model ant navigation centre: the male CNS central-complex subgraph read as a wiring hypothesis for the ant CX
(docs/plans/ANT.md A3-A5). Shiu LIF on the GPU (agent/shiu.py ShiuGPU), one column per trial.

Compass: a stand-in for the sun / polarization sensor, cosine-tuned Poisson input on EPG cells by their protocerebral
bridge (PB) glomerulus phase (instance 'EPG(PB08)_L4' -> L4 -> 3 x 45 deg). Heading is read from cells the cue does not
touch (PEN_a, PEN_b, PEG, PFL3), population vector per type, one circular offset per type fitted on 4 cue azimuths and
tested on the other 4.
  gpu-wait run --shared --vram 4 --preemptible --timeout 1200 --env PYTHONPATH=$PWD -- .venv/bin/python -m agent.ant_cx heading
"""
import re
import sys

import numpy as np
import pandas as pd

from agent import mcns

TYPES = r'^(EPG|PEG|PEN_a|PEN_b|Delta7|hDelta|PFNa|PFNd|PFNv|PFL2|PFL3|ER4d|TN)'
READ = ('PEN_a', 'PEN_b', 'PEG', 'PFL3')
CUE_HZ, ON, OFF, REPEATS = 150.0, 3000, 6000, 2  # Hz at the preferred phase; steps (dt 0.1 ms): cue 0-300 ms, read 300-600 ms after
FIT, TEST = (0, 90, 180, 270), (45, 135, 225, 315)
ODO = ('LNO1', 'LNO2', 'LNOa', 'LCNOp', 'IbSpsP')  # noduli inputs onto PFN (LCNOp also matches LCNOpm): A4's odometer
ODO_HZ, LEG, HOME_READ_STEPS, HIT = 150.0, 2000, 3000, 45.0  # Hz at speed 1; steps per outbound leg, readout after; hit bar deg
HOME_READ = (('PFNa', 'PB'), ('PFNa', 'FB'), ('PFNd', 'PB'), ('PFNd', 'FB'), ('PFNv', 'PB'), ('PFNv', 'FB'),
             ('hDelta', 'FB'), ('PFL3', 'PB'), ('PFL3', 'FB'))


def subgraph(shuffle=None, types=TYPES):
    """CX neurons (graph() indices), their types, PB phase (deg, NaN unknown), PB side, the induced edges reindexed to
    0..n-1 (int64 pre, post, w) and FB column phase (deg, (k-1) x 360 / the type's largest k, NaN unknown).
    shuffle: seed for a degree-preserving mutant (post permuted among the subgraph's edges)."""
    a = pd.read_feather(mcns.DATA / 'body-annotations-male-cns-v1.0-minconf-0.5.feather', columns=['status', 'type', 'instance'])
    a = a[a['status'].isin(['Traced', 'Anchor'])].reset_index(drop=True)
    keep = np.flatnonzero(a['type'].fillna('').str.match(types).to_numpy())
    pre, post, w, *_ = mcns.graph()
    idx = np.full(len(a), -1, np.int64)
    idx[keep] = np.arange(len(keep))
    e = (idx[pre] >= 0) & (idx[post] >= 0)
    p, q = idx[pre[e]], idx[post[e]]
    if shuffle is not None:
        q = np.random.default_rng(shuffle).permutation(q)
    inst = a['instance'].fillna('').to_numpy()[keep]
    phase = np.array([45.0 * (int(m.group(1)) - 1) if (m := re.search(r'\)_[LR](\d)', s)) else np.nan for s in inst])
    side = np.array([m.group(1) if (m := re.search(r'\)_([LR])\d', s)) else '' for s in inst])  # PB half
    t = a['type'].fillna('').to_numpy()[keep]
    k = pd.Series(inst).str.extract(r'_C(\d+)')[0].astype(float)
    fb = (360.0 * (k - 1) / k.groupby(t).transform('max')).to_numpy()
    return keep, t, phase, side, p, q, w[e], fb


def decode(counts, phase, offset=0.0):
    """population-vector heading (deg) from per-cell counts and phases."""
    ok = ~np.isnan(phase)
    z = (counts[..., ok] * np.exp(1j * np.radians(phase[ok]))).sum(-1)
    return (np.degrees(np.angle(z)) - offset) % 360


def err(a, b):
    return np.abs((np.asarray(a) - np.asarray(b) + 180) % 360 - 180)


def heading_trials(shuffle=None, seed=71):
    from agent.shiu import ShiuGPU
    keep, types, phase, side, p, q, w, _ = subgraph(shuffle)
    az = np.array([*FIT, *TEST] * REPEATS, float)
    epg = np.flatnonzero(pd.Series(types).str.startswith('EPG').to_numpy() & ~np.isnan(phase))
    b = ShiuGPU(p, q, w, n=len(keep), flies=len(az), seed=seed)
    for c, h in enumerate(az):
        rate = CUE_HZ * np.maximum(0, np.cos(np.radians(phase[epg] - h)))
        for r in np.unique(rate[rate > 0]):
            b.stimulate(epg[rate == r], float(r), fly=c, window=(0, ON))
    n = len(keep)
    cells = np.flatnonzero(pd.Series(types).str.match('^(EPG|' + '|'.join(READ) + ')').to_numpy())  # the worker holds N x C counts: read these only
    _, got = b.run_groups(np.array([0, ON, OFF]), np.zeros(n, np.int32), readout=(np.arange(len(az))[:, None] * n + cells).ravel())
    x = np.zeros((2, len(az), n), np.int64)
    x[:, :, cells] = np.diff(np.asarray(got, np.int64), axis=0).reshape(2, len(az), len(cells))  # (windows, columns, cells)
    return az, types, phase, side, x[0], x[1], epg


def heading_score(az, types, phase, side, on, off, epg):
    """per read type: offset fitted per PB side on the FIT azimuths (the halves are offset in the fly PB; one map for both
    smeared seed 71), held-out error = median over both sides' decodes of the TEST azimuths."""
    fit = np.isin(az, FIT)
    out = {}
    for t in READ:
        e = []
        for sd in 'LR':
            m = pd.Series(types).str.startswith(t).to_numpy() & (side == sd)
            raw = decode(on[:, m], phase[m])
            o = float(np.degrees(np.angle(np.exp(1j * np.radians(raw[fit] - az[fit])).mean())))
            e += list(err(decode(on[~fit][:, m], phase[m], o), az[~fit]))
        out[t] = {'spikes': int(on[:, pd.Series(types).str.startswith(t).to_numpy()].sum()), 'test_err': float(np.median(e))}
    return out, float(np.median(err(decode(off[:, epg], phase[epg]), az)))


def heading(seed=72):
    from agent import seeds
    seeds.claim('ant_cx_heading', [seed])
    res = {}
    for arm, sh in (('real', None), ('shuffled', 5)):
        r, pers = heading_score(*heading_trials(sh, seed))
        res[arm] = float(np.median([v['test_err'] for v in r.values()]))
        print(arm, {k: {kk: round(vv, 1) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in r.items()}, f'| EPG after cue off: median err {pers:.1f} deg')
    ok = [(f"real: held-out heading error from undriven cells {res['real']:.1f} deg < 30", res['real'] < 30),
          (f"mutant shuffled CX: {res['shuffled']:.1f} deg > 60", res['shuffled'] > 60)]
    for k, v in ok:
        print(('PASS ' if v else 'FAIL ') + k)
    if not all(v for _, v in ok):
        sys.exit('FAIL ant CX heading')
    print('PASS ant CX heading: the fly central complex carries the compass heading into undriven cells')


def paths(rng, k):
    """k scripted outbounds of 3 legs: heading (deg), speed (0.25-1, distance = speed x LEG) and home direction (deg)."""
    h, u = rng.uniform(0, 360, (k, 3)), rng.uniform(0.25, 1, (k, 3))
    return h, u, np.degrees(np.angle(-(u * np.exp(1j * np.radians(h))).sum(1))) % 360


def home_trials(cols, shuffle=None, seed=73, slow=None):
    """cols: (heading, speed, odometer on, key column or None) per column. Compass cue on EPG at the leg heading and
    odometer at ODO_HZ x speed during each leg, then both off: counts (columns, n) over HOME_READ_STEPS. CPU oracle."""
    from agent.shiu import ShiuBrain
    keep, types, phase, side, p, q, w, fb = subgraph(shuffle, TYPES[:-1] + '|' + '|'.join(ODO) + ')')
    n, s = len(keep), pd.Series(types)
    epg = np.flatnonzero(s.str.startswith('EPG').to_numpy() & ~np.isnan(phase))
    odo = np.flatnonzero(s.str.match('^(' + '|'.join(ODO) + ')').to_numpy())
    b = ShiuBrain(p, q, w, n=n, flies=len(cols), seed=seed)
    if slow is not None:  # A4b: synapses onto the memory cells feed a slow integrating current (plane 0) instead of g
        mem = np.flatnonzero(s.str.match(r'^(PFNa|PFNd|PFNv|hDelta)').to_numpy())
        b.edge_groups(np.isin(b.col, mem).astype(np.uint8), np.ones((len(cols), 2), np.float32))
        b.modulate([-1, 0], (slow[0],), (0.0,), (slow[1],))
    for leg in range(3):
        win = (leg * LEG, (leg + 1) * LEG)
        for c, (h, u, on, key) in enumerate(cols):
            rate = CUE_HZ * np.maximum(0, np.cos(np.radians(phase[epg] - h[leg])))
            for r in np.unique(rate[rate > 0]):
                b.stimulate(epg[rate == r], float(r), fly=c, key=key, window=win)
            if on:
                b.stimulate(odo, ODO_HZ * float(u[leg]), fly=c, key=key, window=win)
        b.run([win[1]])
    before = b.spikes.copy()
    x = (b.run([3 * LEG + HOME_READ_STEPS])[0] - before).T
    return x, types, phase, fb, odo


def home_decode(x, types, phase, fb, cand, s=1, o=0.0):
    t, mp = cand
    m = pd.Series(types).str.startswith(t).to_numpy()
    return (s * decode(x[:, m], (phase if mp == 'PB' else fb)[m]) - o) % 360


def home(seed=73, k=32, slow=None, gate=True):
    """A4 (docs/plans/ANT.md): home direction from the CX after scripted outbounds; seed claimed in docs/seeds_used.json."""
    rng = np.random.default_rng(seed)
    fit, test = paths(rng, k), paths(rng, k)
    cols = [(h, u, True, None) for h, u in zip(*fit[:2])] + [(h, u, True, None) for h, u in zip(*test[:2])]
    cols += [(h, u, False, k + i) for i, (h, u) in enumerate(zip(*test[:2]))]  # no odometer, cue draws of the real column
    x, types, phase, fb, odo = home_trials(cols, seed=seed, slow=slow)
    xs = home_trials([(h, u, True, None) for h, u in zip(*test[:2])], shuffle=5, seed=seed, slow=slow)[0]
    best = None
    for cand in HOME_READ:
        for s in (1, -1):
            raw = home_decode(x[:k], types, phase, fb, cand, s)
            o = float(np.degrees(np.angle(np.exp(1j * np.radians(raw - fit[2])).mean())))
            e = float(np.median(err((raw - o) % 360, fit[2])))
            print(f'FIT {cand[0]:6s} {cand[1]} s {s:+d}: median err {e:.1f} deg, spikes {int(x[:k, pd.Series(types).str.startswith(cand[0]).to_numpy()].sum())}')
            if best is None or e < best[0]:
                best = (e, cand, s, o)
    e, cand, s, o = best
    print(f'frozen decoder: {cand} s {s:+d} offset {o:.1f} (FIT median {e:.1f} deg)')
    hits = {arm: int((err(home_decode(xx, types, phase, fb, cand, s, o), test[2]) < HIT).sum())
            for arm, xx in (('real', x[k:2 * k]), ('no odometer', x[2 * k:]), ('shuffled CX', xs))}
    for arm, xx in (('real', x[k:2 * k]), ('no odometer', x[2 * k:]), ('shuffled CX', xs)):
        d = home_decode(xx, types, phase, fb, cand, s, o)
        print(f'{arm}: hits {hits[arm]}/{k}, median err {np.median(err(d, test[2])):.1f} deg, vs last heading + 180 {np.median(err(d, test[0][:, 2] + 180)):.1f}, odometer spikes {int(xx[:, odo].sum())}')
    base = int((err(test[0][:, 2] + 180, test[2]) < HIT).sum())
    print(f'heading-only baseline (last leg + 180): hits {base}/{k}')
    if not gate:
        return hits, base
    if slow is not None:  # A4b gate (ANT.md)
        ok = [(f"real: {hits['real']}/{k} >= 26 and >= baseline {base} + 8", hits['real'] >= 26 and hits['real'] >= base + 8),
              (f"mutant shuffled CX: {hits['shuffled CX']}/{k} <= 12", hits['shuffled CX'] <= 12),
              (f"mutant no odometer: {hits['no odometer']}/{k} <= baseline {base} + 3", hits['no odometer'] <= base + 3)]
        for msg, v in ok:
            print(('PASS ' if v else 'FAIL ') + msg)
        if not all(v for _, v in ok):
            sys.exit('FAIL ant CX home vector A4b')
        print('PASS ant CX home vector A4b: integrating memory cells hold the way home')
        return hits, base
    ok = [(f"real: home direction within {HIT:.0f} deg in {hits['real']}/{k} >= 26", hits['real'] >= 26),
          (f"mutant shuffled CX: {hits['shuffled CX']}/{k} <= 12 (chance)", hits['shuffled CX'] <= 12),
          (f"mutant no odometer: {hits['no odometer']}/{k} <= 12 (chance)", hits['no odometer'] <= 12)]
    for msg, v in ok:
        print(('PASS ' if v else 'FAIL ') + msg)
    if not all(v for _, v in ok):
        sys.exit('FAIL ant CX home vector')
    print('PASS ant CX home vector: the fly central complex holds the way home after a 3-leg outbound')


def home_fit(seed=74):
    """A4b fit: T x b grid on seed 74, best real TEST hits (ANT.md); prints the grid, returns (T, b)."""
    from agent import seeds
    seeds.claim('ant_cx_home_fit', [seed])
    res = {}
    for T in (0.5, 1.0, 2.0, 4.0):
        for bb in (0.02, 0.05, 0.1, 0.2):
            hits, base = home(seed, slow=(T, bb), gate=False)
            res[T, bb] = hits['real']
            print(f'GRID T {T} b {bb}: real {hits["real"]}, no odometer {hits["no odometer"]}, shuffled {hits["shuffled CX"]}, baseline {base}', flush=True)
    best = max(res, key=res.get)
    print(f'FROZEN T {best[0]} b {best[1]} (real {res[best]}/32 on the fit seed)')
    return best


def home_b(seed=75, T=None, b=None):
    """A4b gate on seed 75 with the frozen T, b; also the b = 0 arm, reported."""
    from agent import seeds
    seed = int(seed)
    seeds.claim("ant_cx_home_b", [seed])
    print('--- reported: b = 0 (no integration)')
    home(seed, slow=(float(T), 0.0), gate=False)
    print(f'--- gate: T {T} b {b}')
    home(seed, slow=(float(T), float(b)))


if __name__ == '__main__':
    {'heading': heading, 'home': home, 'home_fit': home_fit, 'home_b': home_b}[sys.argv[1]](*map(float, sys.argv[2:])) if sys.argv[1] == 'home_b' else \
        {'heading': heading, 'home': home, 'home_fit': home_fit}[sys.argv[1]](*map(int, sys.argv[2:]))
