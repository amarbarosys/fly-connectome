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
CLASSES = ('EPG', 'PEG', 'PEN_a', 'PEN_b', 'Delta7', 'hDelta', 'PFNa', 'PFNd', 'PFNv', 'PFL2', 'PFL3', 'ER4d', 'LNO1', 'LNO2', 'LNOa', 'LCNOp', 'IbSpsP', 'TN')  # A5 gain classes
A5_SLOW = (1.0, 0.1)
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


def home_trials(cols, shuffle=None, seed=73, slow=None, arm=None, gain=None):
    """cols: (heading, speed, odometer on, key column or None) per column. Compass cue on EPG at the leg heading and
    odometer at ODO_HZ x speed during each leg, then both off: counts (columns, n) over HOME_READ_STEPS. CPU oracle."""
    from agent.shiu import ShiuBrain
    keep, types, phase, side, p, q, w, fb = subgraph(shuffle, TYPES[:-1] + '|' + '|'.join(ODO) + ')')
    n, s = len(keep), pd.Series(types)
    epg = np.flatnonzero(s.str.startswith('EPG').to_numpy() & ~np.isnan(phase))
    odo = np.flatnonzero(s.str.match('^(' + '|'.join(ODO) + ')').to_numpy())
    if arm is not None:  # A4c: E = IbSpsP excitation at ODO_HZ x speed; D = LNO disinhibition at ODO_HZ x (1 - speed)
        odo = np.flatnonzero(s.str.match(r'^IbSpsP' if arm == 'E' else r'^(LNO1|LNO2|LNOa)$').to_numpy())
    b = ShiuBrain(p, q, w, n=n, flies=len(cols), seed=seed)
    if slow is not None:  # A4b: synapses onto the memory cells feed a slow integrating current (plane 0) instead of g
        mem = np.flatnonzero(s.str.match(r'^(PFNa|PFNd|PFNv|hDelta)').to_numpy())
        onto = np.isin(b.col, mem).astype(np.uint8)
        if gain is None:
            b.edge_groups(onto, np.ones((len(cols), 2), np.float32))
            b.modulate([-1, 0], (slow[0],), (0.0,), (slow[1],))
        else:  # A5: group = 2 x pre type class + onto a memory cell; gain (columns or 1, 34) multiplies each group
            cls = s.str.extract('^(' + '|'.join(CLASSES) + ')')[0].map({c: i for i, c in enumerate(CLASSES)}).to_numpy().astype(np.uint8)
            b.edge_groups(2 * cls[np.repeat(np.arange(n), np.diff(b.row))] + onto, np.broadcast_to(np.asarray(gain, np.float32), (len(cols), 2 * len(CLASSES))))
            b.modulate([-1, 0] * len(CLASSES), (slow[0],), (0.0,), (slow[1],))
    for leg in range(3):
        win = (leg * LEG, (leg + 1) * LEG)
        for c, (h, u, on, key) in enumerate(cols):
            rate = CUE_HZ * np.maximum(0, np.cos(np.radians(phase[epg] - h[leg])))
            for r in np.unique(rate[rate > 0]):
                b.stimulate(epg[rate == r], float(r), fly=c, key=key, window=win)
            if arm == 'D':
                b.stimulate(odo, ODO_HZ * (1 - float(u[leg])) if on else ODO_HZ / 2, fly=c, key=key, window=win)
            elif on:
                b.stimulate(odo, ODO_HZ * float(u[leg]), fly=c, key=key, window=win)
        b.run([win[1]])
    before = b.spikes.copy()
    x = (b.run([3 * LEG + HOME_READ_STEPS])[0] - before).T
    return x, types, phase, fb, odo


def home_decode(x, types, phase, fb, cand, s=1, o=0.0):
    t, mp = cand
    m = pd.Series(types).str.startswith(t).to_numpy()
    return (s * decode(x[:, m], (phase if mp == 'PB' else fb)[m]) - o) % 360


def home(seed=73, k=32, slow=None, gate=True, arm=None, gain=None):
    """A4 (docs/plans/ANT.md): home direction from the CX after scripted outbounds; seed claimed in docs/seeds_used.json."""
    rng = np.random.default_rng(seed)
    fit, test = paths(rng, k), paths(rng, k)
    cols = [(h, u, True, None) for h, u in zip(*fit[:2])] + [(h, u, True, None) for h, u in zip(*test[:2])]
    cols += [(h, u, False, k + i) for i, (h, u) in enumerate(zip(*test[:2]))]  # no odometer, cue draws of the real column
    x, types, phase, fb, odo = home_trials(cols, seed=seed, slow=slow, arm=arm, gain=gain)
    xs = home_trials([(h, u, True, None) for h, u in zip(*test[:2])], shuffle=5, seed=seed, slow=slow, arm=arm, gain=gain)[0]
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


def home_fit(seed=74, arm=None):
    """A4b fit: T x b grid on seed 74, best real TEST hits (ANT.md); prints the grid, returns (T, b)."""
    from agent import seeds
    seed = int(seed)
    seeds.claim('ant_cx_home_fit' + (f'_{arm}' if arm else ''), [seed])
    res = {}
    for T in (0.5, 1.0, 2.0, 4.0):
        for bb in (0.02, 0.05, 0.1, 0.2):
            hits, base = home(seed, slow=(T, bb), gate=False, arm=arm)
            res[T, bb] = hits['real']
            print(f'GRID T {T} b {bb}: real {hits["real"]}, no odometer {hits["no odometer"]}, shuffled {hits["shuffled CX"]}, baseline {base}', flush=True)
    best = max(res, key=res.get)
    print(f'FROZEN T {best[0]} b {best[1]} (real {res[best]}/32 on the fit seed)')
    return best


def home_b(seed=75, T=None, b=None, arm=None):
    """A4b gate on seed 75 with the frozen T, b; also the b = 0 arm, reported."""
    from agent import seeds
    seed = int(seed)
    seeds.claim('ant_cx_home_b' + (f'_{arm}' if arm else ''), [seed])
    print('--- reported: b = 0 (no integration)')
    home(seed, slow=(float(T), 0.0), gate=False, arm=arm)
    print(f'--- gate: T {T} b {b}')
    home(seed, slow=(float(T), float(b)), arm=arm)


def fitness(x, types, phase, fb, home_dir):
    """A5 brain fitness: best over HOME_READ x sign of mean cos(home error), offset fitted on the same paths."""
    best = -1.0
    for cand in HOME_READ:
        for sg in (1, -1):
            raw = home_decode(x, types, phase, fb, cand, sg)
            z = np.exp(1j * np.radians(raw - home_dir))
            best = max(best, float(np.abs(z.mean())) if np.isfinite(z).all() else -1.0)  # |mean| = mean cos at the fitted offset
    return best


def search(seed=79, shuffle=None, gens=30, pop=16, k=32, sigma=0.5, step=0.5):
    """A5 search (ANT.md): diagonal ES on log2 gains; fresh paths per generation; returns the final mean g."""
    import time
    rng, g = np.random.default_rng(seed), np.zeros(2 * len(CLASSES))
    for gen in range(gens):
        t0 = time.perf_counter()
        h, u, d = paths(rng, k)
        eps = rng.standard_normal((pop // 2, len(g)))
        eps = np.vstack([eps, -eps])
        G = np.clip(g + sigma * eps, -3, 3)
        cols = [(hh, uu, True, None) for _ in range(pop) for hh, uu in zip(h, u)]
        x, types, phase, fb, _ = home_trials(cols, shuffle, seed=seed + gen, slow=A5_SLOW, gain=np.repeat(2.0 ** G, k, 0))
        f = np.array([fitness(x[i * k:(i + 1) * k], types, phase, fb, d) for i in range(pop)])
        r = (np.argsort(np.argsort(f)) / (pop - 1)) - 0.5
        g = np.clip(g + step / (pop * sigma) * (r @ eps), -3, 3)
        print(f'GEN {gen} {"shuffled" if shuffle is not None else "real"}: fitness mean {f.mean():.3f} max {f.max():.3f} ({time.perf_counter() - t0:.0f} s)', flush=True)
    print('GAINS', ' '.join(f'{c}{"/mem" if j else ""}={v:+.2f}' for (c, j), v in zip([(c, j) for c in CLASSES for j in (0, 1)], g)), flush=True)
    return g


def a5(seed=79, gate_seed=80, gens=30):
    """A5: real and shuffled-wiring searches on `seed`, gate on `gate_seed` (ANT.md)."""
    from agent import seeds
    seeds.claim('ant_cx_a5_search', [int(seed)])
    seeds.claim('ant_cx_a5_gate', [int(gate_seed)])
    g, gs = search(seed, None, gens), search(seed, 5, gens)
    np.savez(mcns.ROOT / '.work/agent/ant/a5-gains.npz', real=g, shuffled=gs)
    print('--- gate: real searched gains')
    hits, base = home(gate_seed, slow=A5_SLOW, gate=False, gain=2.0 ** g)
    print('--- shuffled-search gains (on its own shuffled wiring is the search; here scored on the real CX procedure with shuffle 5)')
    hs = shuffled_hits(gate_seed, 2.0 ** gs)
    ok = [(f"real: {hits['real']}/32 >= 26 and >= baseline {base} + 8", hits['real'] >= 26 and hits['real'] >= base + 8),
          (f"control shuffled search: {hs}/32 <= baseline {base} + 3", hs <= base + 3),
          (f"mutant no odometer: {hits['no odometer']}/32 <= baseline {base} + 3", hits['no odometer'] <= base + 3),
          (f"mutant shuffled CX with real gains: {hits['shuffled CX']}/32 <= 12", hits['shuffled CX'] <= 12)]
    for msg, v in ok:
        print(('PASS ' if v else 'FAIL ') + msg)
    if not all(v for _, v in ok):
        sys.exit('FAIL ant CX A5 gain search')
    print('PASS ant CX A5: searched gains on the fixed MCNS wiring hold the way home')


def shuffled_hits(seed, gain, k=32):
    """the shuffled-search brain under the A4 procedure: decoder on FIT, hits on TEST, all on shuffled wiring 5."""
    rng = np.random.default_rng(seed)
    fit, test = paths(rng, k), paths(rng, k)
    cols = [(h, u, True, None) for h, u in zip(*fit[:2])] + [(h, u, True, None) for h, u in zip(*test[:2])]
    x, types, phase, fb, _ = home_trials(cols, 5, seed=seed, slow=A5_SLOW, gain=gain)
    best = None
    for cand in HOME_READ:
        for sg in (1, -1):
            raw = home_decode(x[:k], types, phase, fb, cand, sg)
            o = float(np.degrees(np.angle(np.exp(1j * np.radians(raw - fit[2])).mean())))
            e = float(np.median(err((raw - o) % 360, fit[2])))
            if best is None or e < best[0]:
                best = (e, cand, sg, o)
    _, cand, sg, o = best
    h = int((err(home_decode(x[k:], types, phase, fb, cand, sg, o), test[2]) < HIT).sum())
    print(f'shuffled search brain: {cand} s {sg:+d}, hits {h}/{k}')
    return h


MEM6, M_PRE = r'^(PFNd|PFNv)', r'^(EPG|PEG|PEN_a|PEN_b|Delta7|PFNa|PFNd|PFNv)'  # A6: memory cells, CX pres feeding M
def cx_types():
    """A7a: regex for every MCNS type of class CX (2,950 traced cells), exact names."""
    a = pd.read_feather(mcns.DATA / 'body-annotations-male-cns-v1.0-minconf-0.5.feather', columns=['status', 'type', 'class'])
    t = a.loc[a['status'].isin(['Traced', 'Anchor']) & (a['class'] == 'CX'), 'type'].dropna().unique()
    return '^(' + '|'.join(re.escape(x) for x in sorted(t)) + ')$'


SUB = TYPES  # m_trials' subgraph; A7a sets cx_types()
M_EXC = False  # A6b sets True: M takes positive edges only
M_T, M_B, SEG, K6 = 4.0, (0.02, 0.05, 0.1, 0.2), 2000, 128  # level decay s, b grid (P1), steps per probe segment, paths


def audit():
    """A6 P0: every MCNS type synapsing onto PFNd / PFNv / hDelta / PFL3 (signed synapses), and whether M's pres are CX."""
    a = pd.read_feather(mcns.DATA / 'body-annotations-male-cns-v1.0-minconf-0.5.feather', columns=['status', 'type', 'superclass'])
    a = a[a['status'].isin(['Traced', 'Anchor'])].reset_index(drop=True)
    t, sc = a['type'].fillna('').to_numpy(), a['superclass'].fillna('').to_numpy()
    pre, post, w, *_ = mcns.graph()
    tgt = pd.Series(t).str.extract(r'^(PFNd|PFNv|hDelta|PFL3)')[0].to_numpy()
    e = pd.notna(tgt[post])
    df = pd.DataFrame({'pre': t[pre[e]], 'sc': sc[pre[e]], 'onto': tgt[post[e]], 'w': w[e]})
    tab = df.pivot_table(index=['pre', 'sc'], columns='onto', values='w', aggfunc='sum', fill_value=0)
    tab['cells'] = [int((t == p).sum()) for p, _ in tab.index]
    tab = tab.loc[tab.drop(columns='cells').abs().sum(1).sort_values(ascending=False).index]
    print(tab.head(40).to_string())
    m = df[df['onto'].isin(['PFNd', 'PFNv']) & df['pre'].str.match(M_PRE)]
    exc = m.groupby('pre')['w'].sum()
    print('M pres onto PFNd / PFNv:', exc.round(0).to_dict())
    ok = [(f"M pres are all CX-internal (subgraph types): {sorted(set(m['pre']))}", bool(m['pre'].str.match(TYPES).all())),
          (f"memory cells get excitatory M input: {int(exc[exc > 0].sum())} synapses", bool((exc > 0).any()))]
    return ok


def m_trials(cols, shuffle=None, seed=81, b=0.1, read=3000, exc=None, bias=0.0, gain=None):
    """A6 brain with M (ANT.md): cols = (headings, speed multipliers, key) per column, one SEG-step segment per entry;
    cue on EPG at each heading; M edges (M_PRE -> PFNd / PFNv) feed slow plane 0 scaled by the segment's speed, then
    scale 0 and cue off for `read` steps. Returns counts in the last 1000 cue steps, counts in the read window, the
    final levels m (columns, n) and the subgraph."""
    from agent.shiu import ShiuBrain
    exc = M_EXC if exc is None else exc
    keep, types, phase, side, p, q, w, fb = subgraph(shuffle, SUB)
    n, s = len(keep), pd.Series(types)
    epg = np.flatnonzero(s.str.startswith('EPG').to_numpy() & ~np.isnan(phase))
    br = ShiuBrain(p, q, w, n=n, flies=len(cols), seed=seed)
    pre = np.repeat(np.arange(n), np.diff(br.row))
    grp = (s.str.match(M_PRE).to_numpy()[pre] & s.str.match(MEM6).to_numpy()[br.col] & ((br.w > 0) | (not exc))).astype(np.uint8)  # A6b exc: positive edges only
    gain = np.ones((len(cols), 1), np.float32) if gain is None else np.broadcast_to(np.asarray(gain, np.float32), (len(cols), np.shape(gain)[-1]))
    if gain.shape[1] > 1:  # A7d: M edges grouped by target column (column_groups), one gain each
        grp = np.where(grp > 0, column_groups(types, phase, side)[br.col], 0).astype(np.uint8)
    G = gain.shape[1]
    br.edge_groups(grp, np.ones((len(cols), G + 1), np.float32))
    br.modulate([-1] + [0] * G, (M_T,), (0.0,), (b,))
    segs = len(cols[0][0])
    for j in range(segs):
        win = (j * SEG, (j + 1) * SEG)
        for c, (h, u, key) in enumerate(cols):
            br.scale[c, 1:] = u[j] * gain[c]
            rate = CUE_HZ * np.maximum(0, np.cos(np.radians(phase[epg] - h[j])))
            for r in np.unique(rate[rate > 0]):
                br.stimulate(epg[rate == r], float(r), fly=c, key=key, window=win)
        if j == segs - 1:
            br.run([win[1] - 1000])
            c0 = br.spikes.copy()
        br.run([win[1]])
    c1 = br.spikes.copy()
    br.scale[:, 1:] = 0.0
    if bias > 0:  # A6c: constant, heading-free current (mV / ms) on the memory cells during the read window
        br.inject(np.flatnonzero(s.str.match(MEM6).to_numpy()), float(bias), window=(segs * SEG, segs * SEG + read))
    br.run([segs * SEG + read])
    return (c1 - c0).T, (br.spikes - c1).T, br.m[0].reshape(len(cols), n), types, phase, fb


def vec(x, phase):
    ok = ~np.isnan(phase)
    return (x[..., ok] * np.exp(1j * np.radians(phase[ok]))).sum(-1)


def fit_off(z, target):
    """one circular offset (deg) fitted on the same trials, and the decoded angles after it."""
    raw = np.degrees(np.angle(z))
    o = float(np.degrees(np.angle(np.exp(1j * np.radians(raw - target)).mean())))
    return (raw - o) % 360


def dec6(x, types, phase, fb, cand, s=1, o=0.0):
    """home_decode, NaN (a miss) for a column whose read cells never spiked: a zero vector has no direction."""
    d = home_decode(x, types, phase, fb, cand, s, o)
    return np.where(x[:, pd.Series(types).str.startswith(cand[0]).to_numpy()].sum(1) > 0, d, np.nan)


def e6(a, b):
    return np.nan_to_num(err(a, b), nan=180.0)


def persistence(seed=81, shuffle=None):
    """A6 P1: 8 azimuths, cue 200 ms at speed 1; per b: hits (decode within 45 deg 300 ms after cue off) and norm ratio."""
    az = np.arange(0, 360, 45.0)
    out = {}
    for b in M_B:
        on, off, _, types, phase, _ = m_trials([([h], [1.0], None) for h in az], shuffle, seed, b)
        mem = pd.Series(types).str.match(MEM6).to_numpy()
        zo, zf = vec(on[:, mem], phase[mem]), vec(off[:, mem], phase[mem])
        hits = int((err(fit_off(zf, az), az) < HIT).sum())
        ratio = float(np.median(np.abs(zf) / np.maximum(np.abs(zo) / 3, 1e-9)))  # on window 100 ms, read window 300 ms
        out[b] = (hits, ratio)
        print(f'P1 {"shuffled" if shuffle is not None else "real"} b {b}: {hits}/8 within 45 deg 300 ms after cue off, norm ratio {ratio:.2f}, memory spikes after {int(off[:, mem].sum())}', flush=True)
    return out


def conjunction(seed=81, b=0.1):
    """A6 P2: headings 0/90/180/270 x speeds 0.2-1.0, one segment each; level sum vs speed and level phase vs heading."""
    hs, us = (0.0, 90.0, 180.0, 270.0), (0.2, 0.4, 0.6, 0.8, 1.0)
    cols = [([h], [u], None) for h in hs for u in us]
    _, _, m, types, phase, _ = m_trials(cols, None, seed, b, read=0)
    mem = pd.Series(types).str.match(MEM6).to_numpy()
    tot = m[:, mem].sum(1).reshape(len(hs), len(us))
    rho = [float(pd.Series(r).corr(pd.Series(us), method='spearman')) for r in tot]
    ph = fit_off(vec(m[:, mem], phase[mem]), np.repeat(hs, len(us))).reshape(len(hs), len(us))[:, -1]
    good = int((err(ph, hs) < HIT).sum())
    print(f'P2 level sum per heading x speed:\n{np.round(tot, 1)}\nSpearman per heading {np.round(rho, 2)}; phase error at u 1 {np.round(err(ph, hs), 1)}')
    return [(f'P2 level rises with speed: Spearman {min(rho):.2f} >= 0.9 at every heading', round(min(rho), 9) >= 0.9),
            (f'P2 level phase tracks heading: {good}/4 within 45 deg >= 3', good >= 3)]


def paths6(rng, k):
    """3-leg outbounds as paths(), rejection-sampled so last leg + 180 misses home by > 90 deg (heading baseline 0)."""
    h, u, d = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    while len(d) < k:
        hh, uu, dd = paths(rng, 4 * k)
        ok = err(hh[:, 2] + 180, dd) > 90
        h, u, d = np.vstack([h, hh[ok]]), np.vstack([u, uu[ok]]), np.r_[d, dd[ok]]
    return h[:k], u[:k], d[:k]


def g6(seed=82, b=0.1, k=K6, pf=None, gain=None, sh=None):
    """A6 G6: decoder on k FIT paths, hits on k TEST paths; arms real, no odometer (u 0.6), M off (b 0), shuffled with M."""
    rng, pf = np.random.default_rng(seed), pf or paths6
    fit, test = pf(rng, k), pf(rng, k)
    cols = [(h, u, None) for h, u in zip(*fit[:2])] + [(h, u, None) for h, u in zip(*test[:2])]
    cols += [(h, np.full(3, 0.6), k + i) for i, h in enumerate(test[0])]
    _, x, _, types, phase, fb = m_trials(cols, sh, seed, b, gain=gain)
    arms = {'real': x[k:2 * k], 'no odometer': x[2 * k:]}
    arms['M off'] = m_trials([(h, u, None) for h, u in zip(*test[:2])], sh, seed, 0.0, gain=gain)[1]
    arms['shuffled CX'] = m_trials([(h, u, None) for h, u in zip(*test[:2])], 5, seed, b, gain=gain)[1]
    best = None
    for cand in HOME_READ:
        for sg in (1, -1):
            raw = dec6(x[:k], types, phase, fb, cand, sg)
            o = float(np.degrees(np.angle(np.nanmean(np.exp(1j * np.radians(raw - fit[2]))))))
            e = float(np.median(e6((raw - o) % 360, fit[2])))
            if np.isfinite(o) and (best is None or e < best[0]):
                best = (e, cand, sg, o)
    e, cand, sg, o = best
    print(f'G6 frozen decoder: {cand} s {sg:+d} offset {o:.1f} (FIT median {e:.1f} deg)')
    hits = {}
    for arm, xx in arms.items():
        d = dec6(xx, types, phase, fb, cand, sg, o)
        hits[arm] = int((e6(d, test[2]) < HIT).sum())
        print(f'G6 {arm}: hits {hits[arm]}/{k}, median err {np.median(e6(d, test[2])):.1f} deg')
    eq = np.degrees(np.angle(-np.exp(1j * np.radians(test[0])).sum(1))) % 360
    print(f'G6 heading-only baseline: {int((err(test[0][:, 2] + 180, test[2]) < HIT).sum())}/{k}; speed-blind: {int((err(eq, test[2]) < HIT).sum())}/{k}', flush=True)
    global LAST_HITS
    LAST_HITS = hits
    bound = 41 * k // 128
    return [(f"G6 real: {hits['real']}/{k} >= {3 * k // 4}", hits['real'] >= 3 * k // 4)] + \
           [(f'G6 {a}: {hits[a]}/{k} <= {bound}', hits[a] <= bound) for a in ('no odometer', 'M off', 'shuffled CX')]


def report(ok, name):
    for msg, v in ok:
        print(('PASS ' if v else 'FAIL ') + msg, flush=True)
    if not all(v for _, v in ok):
        sys.exit(f'FAIL ant CX {name}')


def a6(probe=81, gate=82, tag='a6'):
    """A6 (ANT.md): P0 audit, P1 persistence (freezes b), P2 conjunction, G6 home vector; stops at the first FAIL."""
    from agent import seeds
    report(audit(), 'P0 audit')
    seeds.claim(f'ant_cx_{tag}_probe', [int(probe)])
    p1 = persistence(probe)
    persistence(probe, 5)  # reported
    b = next((bb for bb in M_B if p1[bb][0] >= 7 and p1[bb][1] >= 0.5), None)
    report([(f'P1 some b in {M_B} holds the heading 300 ms (>= 7/8, norm >= 0.5): frozen b {b}', b is not None)], 'P1 persistence')
    report(conjunction(probe, b), 'P2 conjunction')
    seeds.claim(f'ant_cx_{tag}_gate', [int(gate)])
    report(g6(gate, b), 'G6 home vector')
    print(f'PASS ant CX {tag}: MCNS CX wiring + the named integrator M holds the way home; shuffled wiring with M does not')


def a6b(probe=83, gate=84):
    """A6b (ANT.md): A6 with M on excitatory edges only."""
    global M_EXC
    M_EXC = True
    a6(probe, gate, 'a6b')


A6C_R = (0.2, 0.4, 0.8, 1.6)  # mV / ms constant current


def calibrate(seed, b, bias, shuffle=None):
    """A6c: preferred heading per memory cell from 16 single legs (u 1, bias); NaN for silent cells; and the held-out
    style check: single-leg decode error with those phases (median, deg)."""
    az = np.arange(0, 360, 22.5)
    _, off, _, types, _, _ = m_trials([([h], [1.0], None) for h in az], shuffle, seed, b, bias=bias)
    mem = pd.Series(types).str.match(MEM6).to_numpy()
    z = (off * np.exp(1j * np.radians(az))[:, None]).sum(0)
    pref = np.where(mem & (np.abs(z) > 0), np.degrees(np.angle(z)), np.nan)
    e = float(np.median(err(fit_off(vec(off, pref), az), az)))
    return pref, e


def bias_probe(seed, b):
    """A6c P3: smallest R whose post-leg memory spikes are > 0 at u 0.2 and rise with u at every heading."""
    hs, us = (0.0, 90.0, 180.0, 270.0), (0.2, 0.4, 0.6, 0.8, 1.0)
    for R in A6C_R:
        _, off, _, types, _, _ = m_trials([([h], [u], None) for h in hs for u in us], None, seed, b, bias=R)
        S = off[:, pd.Series(types).str.match(MEM6).to_numpy()].sum(1).reshape(len(hs), len(us))
        rho = [round(float(pd.Series(r).corr(pd.Series(us), method='spearman')), 9) for r in S]
        print(f'P3 R {R}: spikes per heading x speed {S.tolist()}, Spearman {rho}', flush=True)
        if (S[:, 0] > 0).all() and min(rho) >= 0.9:
            return R
    return None


def g6c(seed, b, R, pref, prefs, k=K6):
    """A6c G6: calibrated-phase memory decoder; prefs = the shuffled wiring's own calibration."""
    rng = np.random.default_rng(seed)
    fit, test = paths6(rng, k), paths6(rng, k)
    cols = [(h, u, None) for h, u in zip(*fit[:2])] + [(h, u, None) for h, u in zip(*test[:2])]
    cols += [(h, np.full(3, 0.6), k + i) for i, h in enumerate(test[0])]
    tc = [(h, u, None) for h, u in zip(*test[:2])]
    x = m_trials(cols, None, seed, b, bias=R)[1]
    arms = {'real': (x[k:2 * k], pref), 'no odometer': (x[2 * k:], pref),
            'M off': (m_trials(tc, None, seed, 0.0, bias=R)[1], pref), 'shuffled CX': (m_trials(tc, 5, seed, b, bias=R)[1], prefs)}
    zf = vec(x[:k], pref)
    best = None
    for sg in (1, -1):
        raw = sg * np.degrees(np.angle(zf))
        o = float(np.degrees(np.angle(np.exp(1j * np.radians(raw - fit[2])).mean())))
        e = float(np.median(err((raw - o) % 360, fit[2])))
        if best is None or e < best[0]:
            best = (e, sg, o)
    e, sg, o = best
    print(f'G6c decoder: memory cells, calibrated phases, s {sg:+d} offset {o:.1f} (FIT median {e:.1f} deg)')
    hits = {}
    for arm, (xx, pp) in arms.items():
        z = vec(xx, pp)
        d = np.where(np.abs(z) > 0, (sg * np.degrees(np.angle(z)) - o) % 360, np.nan)
        hits[arm] = int((e6(d, test[2]) < HIT).sum())
        print(f'G6c {arm}: hits {hits[arm]}/{k}, median err {np.median(e6(d, test[2])):.1f} deg', flush=True)
    bound = 41 * k // 128
    return [(f"G6c real: {hits['real']}/{k} >= {3 * k // 4}", hits['real'] >= 3 * k // 4)] + \
           [(f'G6c {a}: {hits[a]}/{k} <= {bound}', hits[a] <= bound) for a in ('no odometer', 'M off', 'shuffled CX')]


def a6c(probe=85, gate=86):
    """A6c (ANT.md): A6b + readout bias R (P3) + calibrated memory-cell phases; stops at the first FAIL."""
    from agent import seeds
    global M_EXC
    M_EXC = True
    report(audit(), 'P0 audit')
    seeds.claim('ant_cx_a6c_probe', [int(probe)])
    p1 = persistence(probe)
    b = next((bb for bb in M_B if p1[bb][0] >= 7 and p1[bb][1] >= 0.5), None)
    report([(f'P1 frozen b {b}', b is not None)], 'P1 persistence')
    report(conjunction(probe, b), 'P2 conjunction')
    R = bias_probe(probe, b)
    report([(f'P3 bias R in {A6C_R}: frozen {R}', R is not None)], 'P3 bias')
    pref, e = calibrate(probe, b, R)
    prefs, es = calibrate(probe, b, R, 5)
    print(f'P3 calibration: {int(np.isfinite(pref).sum())} memory cells tuned; single-leg error {e:.1f} deg (shuffled {es:.1f})')
    report([(f'P3 single-leg decode with calibrated phases {e:.1f} < 20 deg', e < 20)], 'P3 calibration')
    seeds.claim('ant_cx_a6c_gate', [int(gate)])
    report(g6c(gate, b, R, pref, prefs), 'G6c home vector')
    print('PASS ant CX A6c: MCNS CX wiring + M with a calibrated readout holds the way home')


def a6d(probe=87, gate=88):
    """A6d (ANT.md): A6b with M fed by the EPG compass cells only."""
    global M_EXC, M_PRE
    M_EXC, M_PRE = True, r'^EPG'
    a6(probe, gate, 'a6d')


def paths7(rng, k):
    """A7: 3-leg outbounds whose true home differs by > 90 deg from the speed-blind home and from last leg + 180."""
    h, u, d = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    while len(d) < k:
        hh, uu, dd = paths(rng, 200 * k)
        eq = np.degrees(np.angle(-np.exp(1j * np.radians(hh)).sum(1))) % 360
        ok = (err(eq, dd) > 90) & (err(hh[:, 2] + 180, dd) > 90)
        h, u, d = np.vstack([h, hh[ok]]), np.vstack([u, uu[ok]]), np.r_[d, dd[ok]]
    return h[:k], u[:k], d[:k]


def a7(probe=89, gate=90, tag='a7', pre=None, grid=None):
    """A7 (ANT.md): full CX + excitatory M; b fitted on `probe` by real TEST hits, gate on `gate`, A7 paths."""
    from agent import seeds
    global M_EXC, SUB, M_PRE
    M_EXC, SUB, M_PRE = True, cx_types(), pre or M_PRE
    seeds.claim(f'ant_cx_{tag}_fit', [int(probe)])
    res, grid = {}, grid or M_B
    for b in grid:
        g6(probe, b, pf=paths7)
        res[b] = LAST_HITS['real']
        print(f'A7 FIT b {b}: real {res[b]}/{K6}', flush=True)
    b = max(grid, key=lambda x: (res[x], -x))
    print(f'A7 frozen b {b}', flush=True)
    seeds.claim(f'ant_cx_{tag}_gate', [int(gate)])
    report(g6(gate, b, pf=paths7), 'A7 distance')
    print('PASS ant CX A7: the full fly CX + M integrates distance-weighted legs into the way home')


def a7c(probe=91, gate=92):
    """A7c (ANT.md): A7 with M fed by non-PFN CX pres only."""
    a7(probe, gate, 'a7c', r'^(EPG|PEG|PEN_a|PEN_b|Delta7)')


def a7e(probe=93, gate=94):
    """A7e (ANT.md): A7c with b grid {0.2, 0.3, 0.5, 0.8}."""
    a7(probe, gate, 'a7e', r'^(EPG|PEG|PEN_a|PEN_b|Delta7)', (0.2, 0.3, 0.5, 0.8))


def column_groups(types, phase, side):
    """A7d: group id per cell, 1.. for each memory (type, PB side, glomerulus), 0 otherwise."""
    t = pd.Series(types).str.extract('^(PFNd|PFNv)')[0]
    key = t + side + pd.Series(phase).astype(str)
    ok = t.notna().to_numpy() & ~np.isnan(phase)
    ids = {kk: i + 1 for i, kk in enumerate(sorted(set(key[ok])))}
    return np.array([ids.get(kk, 0) if o else 0 for kk, o in zip(key, ok)], np.int64)


def n_column_groups():
    keep, types, phase, side, *_ = subgraph(None, SUB)
    return int(column_groups(types, phase, side).max())


A7D = dict(b=0.3, pre=r'^(EPG|PEG|PEN_a|PEN_b|Delta7)', sigma=0.3, pop=16, k=32, step=0.5)


def search7d(seed, shuffle=None, gens=30):
    """A7d: diagonal ES on log2 gains per memory column (in [-2, 2]); fresh paths7 per generation."""
    import time
    o = A7D
    rng, g = np.random.default_rng(seed), np.zeros(n_column_groups())
    for gen in range(gens):
        t0 = time.perf_counter()
        h, u, d = paths7(rng, o['k'])
        eps = rng.standard_normal((o['pop'] // 2, len(g)))
        eps = np.vstack([eps, -eps])
        G = np.clip(g + o['sigma'] * eps, -2, 2)
        cols = [(hh, uu, None) for _ in range(o['pop']) for hh, uu in zip(h, u)]
        _, x, _, types, phase, fb = m_trials(cols, shuffle, seed + gen, o['b'], gain=np.repeat(2.0 ** G, o['k'], 0))
        f = np.array([fitness(x[i * o['k']:(i + 1) * o['k']], types, phase, fb, d) for i in range(o['pop'])])
        r = (np.argsort(np.argsort(f)) / (o['pop'] - 1)) - 0.5
        g = np.clip(g + o['step'] / (o['pop'] * o['sigma']) * (r @ eps), -2, 2)
        print(f'GEN {gen} {"shuffled" if shuffle is not None else "real"}: fitness mean {f.mean():.3f} max {f.max():.3f} ({time.perf_counter() - t0:.0f} s)', flush=True)
    return g


def a7d(fit=95, gate=96, gens=30):
    """A7d (ANT.md): calibration search on M column gains, real and shuffled wiring; gate on `gate`."""
    from agent import seeds
    global M_EXC, SUB, M_PRE
    M_EXC, SUB, M_PRE = True, cx_types(), A7D['pre']
    seeds.claim('ant_cx_a7d_fit', [int(fit)])
    g, gs = search7d(fit, None, gens), search7d(fit, 5, gens)
    np.savez(mcns.ROOT / '.work/agent/ant/a7d-gains.npz', real=g, shuffled=gs)
    print('GAINS real', np.round(g, 2).tolist(), '\nGAINS shuffled', np.round(gs, 2).tolist(), flush=True)
    seeds.claim('ant_cx_a7d_gate', [int(gate)])
    print('--- shuffled-search brain on its own wiring')
    g6(gate, A7D['b'], pf=paths7, gain=2.0 ** gs, sh=5)
    hs = LAST_HITS['real']
    print('--- gate: real searched gains')
    ok = g6(gate, A7D['b'], pf=paths7, gain=2.0 ** g)
    report(ok + [(f'A7d shuffled-search brain: {hs}/{K6} <= 41', hs <= 41)], 'A7d calibration')
    print('PASS ant CX A7d: calibrated column gains on the full fly CX bring the distance-weighted home vector to the bar')


if __name__ == '__main__':
    num = lambda v: int(v) if v.lstrip('-').isdigit() else float(v) if v.replace('.', '', 1).lstrip('-').isdigit() else v
    {'heading': heading, 'home': home, 'home_fit': home_fit, 'home_b': home_b, 'search': search, 'a5': a5,
     'audit': lambda: report(audit(), 'P0 audit'), 'a6': a6, 'a6b': a6b, 'a6c': a6c, 'a6d': a6d, 'a7': a7, 'a7c': a7c, 'a7e': a7e, 'a7d': a7d}[sys.argv[1]](*map(num, sys.argv[2:]))
