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


def subgraph(shuffle=None):
    """CX neurons (graph() indices), their types, PB phase (deg, NaN unknown) and the induced edges reindexed to 0..n-1
    (int64 pre, post, w). shuffle: seed for a degree-preserving mutant (post permuted among the subgraph's edges)."""
    a = pd.read_feather(mcns.DATA / 'body-annotations-male-cns-v1.0-minconf-0.5.feather', columns=['status', 'type', 'instance'])
    a = a[a['status'].isin(['Traced', 'Anchor'])].reset_index(drop=True)
    keep = np.flatnonzero(a['type'].fillna('').str.match(TYPES).to_numpy())
    pre, post, w, *_ = mcns.graph()
    idx = np.full(len(a), -1, np.int64)
    idx[keep] = np.arange(len(keep))
    e = (idx[pre] >= 0) & (idx[post] >= 0)
    p, q = idx[pre[e]], idx[post[e]]
    if shuffle is not None:
        q = np.random.default_rng(shuffle).permutation(q)
    inst = a['instance'].fillna('').to_numpy()[keep]
    phase = np.array([45.0 * (int(m.group(1)) - 1) if (m := re.search(r'\)_[LR](\d)', s)) else np.nan for s in inst])
    return keep, a['type'].fillna('').to_numpy()[keep], phase, p, q, w[e]


def decode(counts, phase, offset=0.0):
    """population-vector heading (deg) from per-cell counts and phases."""
    ok = ~np.isnan(phase)
    z = (counts[..., ok] * np.exp(1j * np.radians(phase[ok]))).sum(-1)
    return (np.degrees(np.angle(z)) - offset) % 360


def err(a, b):
    return np.abs((np.asarray(a) - np.asarray(b) + 180) % 360 - 180)


def heading_trials(shuffle=None, seed=71):
    from agent.shiu import ShiuGPU
    keep, types, phase, p, q, w = subgraph(shuffle)
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
    return az, types, phase, x[0], x[1], epg


def heading_score(az, types, phase, on, off, epg):
    fit = np.isin(az, FIT)
    out = {}
    for t in READ:
        m = pd.Series(types).str.startswith(t).to_numpy()
        raw = decode(on[:, m], phase[m])
        o = float(np.degrees(np.angle(np.exp(1j * np.radians(raw[fit] - az[fit])).mean())))  # circular mean offset, fit azimuths only
        out[t] = {'spikes': int(on[:, m].sum()), 'test_err': float(np.median(err(decode(on[~fit][:, m], phase[m], o), az[~fit]))), 'offset': o}
    return out, float(np.median(err(decode(off[:, epg], phase[epg]), az)))


def heading(seed=71):
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


if __name__ == '__main__':
    {'heading': heading}[sys.argv[1]]()
