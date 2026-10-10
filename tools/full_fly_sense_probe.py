"""Brain-only probe for FULL_FLY U2 S3 (gravity and body touch), as tools/full_fly_eye_probe.py: forward DNs at 80 Hz plus
one sense on one side, no body; per column the DN turn readout T (mcns_link steer). Gravity: roll 20 deg = the downhill
side's Johnston's organ wind_gravity cells at RMAX x 20/30 Hz (agent/senses.py); touch: the left or right notum bristles at
RMAX. Gate (pre-registered, FULL_FLY.md U2): T(left) vs T(right) paired over REPEATS columns, |t| >= 2.5; the side-swap
mutant (cells of the other side) reverses the sign of the left - right difference.
Smell (U2 S1a): vinegar ORNs at 150 Hz on one antenna and 150 (1 - a) on the other; DNa02 / DNa01 left - right, no fitting.
  gpu-wait run --shared --vram 8 --preemptible --timeout 1200 --env PYTHONPATH=$PWD -- .venv/bin/python tools/full_fly_sense_probe.py [smell] [SEED]"""
import json
import sys

import numpy as np

from agent import senses as sn
from agent.body_engine import fly_link as fl
from agent.body_engine.mcns_link import FWD_HZ, brain
from agent.mcns_mapping import descending_name_sets

REPEATS, N = 16, 500
SMELL = sys.argv[1:2] == ['smell']
seed = int(sys.argv[1 + SMELL]) if len(sys.argv) > 1 + SMELL else 301
from agent import seeds
seeds.claim('full_fly_smell_probe' if SMELL else 'full_fly_sense_probe', [seed])
c, cs = sn.cells(), sn.cells(swap=True)
if SMELL:
    from agent.mcns_mapping import annotation_table
    a = annotation_table()
    ty, sd_ = a.type.fillna('').to_numpy(), a.somaSide.fillna('').str[:1].str.upper().to_numpy()
    steer = {(t, s_): int(np.flatnonzero((ty == t) & (sd_ == s_))[0]) for t in ('DNa02', 'DNa01') for s_ in 'LR'}
    arms = [(g, sd, sw, j) for j in range(REPEATS) for g in (1.0, 0.2, 0.05) for sd in 'LR' for sw in ((False, True) if g == 1.0 else (False,))]
    v = np.load('.work/agent/mcns/vision_readout.npz')
    dn, _ = descending_name_sets()
    b, f, nl, _ = brain(cols=len(arms), seed=seed)
    for k, (g, sd, sw, j) in enumerate(arms):
        cc = cs if sw else c
        b.stimulate(dn['forward'], FWD_HZ, fly=k, key=k)
        b.stimulate(cc['smell', sd], sn.RMAX, fly=k, key=k)
        if g < 1:
            b.stimulate(cc['smell', 'R' if sd == 'L' else 'L'], sn.RMAX * (1 - g), fly=k, key=k)
    read = np.r_[v['dn'], list(steer.values())]
    b.start([1], np.concatenate([k * b.n + read for k in range(len(arms))]).astype(np.int32))
    x = b.advance(N * fl.K).astype(float).reshape(len(arms), -1) * (100 / N)
    b.close()
    T = ((x[:, :len(v['dn'])] - v['mu']) / v['sd']) @ v['w'] + v['y0']
    col = {k: len(v['dn']) + i for i, k in enumerate(steer)}
    res, ok = {}, []
    for g, sw in ((1.0, False), (1.0, True), (0.2, False), (0.05, False)):
        L = [arms.index((g, 'L', sw, j)) for j in range(REPEATS)]
        R = [arms.index((g, 'R', sw, j)) for j in range(REPEATS)]
        for name, val in [(t, x[:, col[t, 'L']] - x[:, col[t, 'R']]) for t in ('DNa02', 'DNa01')] + [('T', T)]:
            d = val[L] - val[R]  # odour-left minus odour-right
            res[f'{name} a={g} swap={sw}'] = (float(d.mean()), float(d.mean() / (d.std(ddof=1) / np.sqrt(REPEATS) + 1e-12)))
    for k, (m_, t_) in res.items():
        print(f'{k}: odour-left minus odour-right {m_:+.3f} (t {t_:+.1f})')
    print('spikes per 100 ms, DNa02 L/R at a=1 left odour:', x[[arms.index((1.0, 'L', False, j)) for j in range(REPEATS)]][:, [col['DNa02', 'L'], col['DNa02', 'R']]].mean(0).round(2).tolist())
    m1, t1 = res['DNa02 a=1.0 swap=False']
    mw, _ = res['DNa02 a=1.0 swap=True']
    ok = [(f'G1 DNa02 left - right follows the odour side: {m1:+.3f} (t {t1:+.1f}) > 0, t >= 2.5', m1 > 0 and t1 >= 2.5),
          (f'G2 side-swap mutant reverses it: {mw:+.3f}', mw * m1 < 0)]
    (fl.OUT / f'full-fly-smell-probe-{seed}.json').write_text(json.dumps(res))
    for k, v_ in ok:
        print(('PASS ' if v_ else 'FAIL ') + k)
    sys.exit(0 if all(v_ for _, v_ in ok) else 'FAIL full fly S1a')
HZ = {'gravity': sn.RMAX * 20 / 30, 'body': sn.RMAX}
arms = [(g, sd, sw, j) for j in range(REPEATS) for g in HZ for sd in 'LR' for sw in (False, True)]
v = np.load('.work/agent/mcns/vision_readout.npz')
dn, _ = descending_name_sets()
b, f, nl, _ = brain(cols=len(arms), seed=seed)
for k, (g, sd, sw, j) in enumerate(arms):
    b.stimulate(dn['forward'], FWD_HZ, fly=k, key=k)
    b.stimulate((cs if sw else c)[g, sd], HZ[g], fly=k, key=k)
b.start([1], np.concatenate([k * b.n + v['dn'] for k in range(len(arms))]).astype(np.int32))
x = b.advance(N * fl.K).astype(float).reshape(len(arms), -1) * (100 / N)
b.close()
T = ((x - v['mu']) / v['sd']) @ v['w'] + v['y0']
res, ok = {}, []
for g in HZ:
    for sw in (False, True):
        d = np.array([T[arms.index((g, 'L', sw, j))] - T[arms.index((g, 'R', sw, j))] for j in range(REPEATS)])
        res[g, sw] = (float(d.mean()), float(d.mean() / (d.std(ddof=1) / np.sqrt(REPEATS))))
    (m, t), (mw, _) = res[g, False], res[g, True]
    print(f'{g}: T(left) - T(right) {m:+.3f} (t {t:+.1f}); side-swapped {mw:+.3f}')
    ok += [(f'{g}: the side sets the turn readout, |t| {abs(t):.1f} >= 2.5', abs(t) >= 2.5), (f'{g}: side-swap mutant reverses it ({mw:+.3f} vs {m:+.3f})', mw * m < 0)]
(fl.OUT / f'full-fly-sense-probe-{seed}.json').write_text(json.dumps({f'{g}-{sw}': v_ for (g, sw), v_ in res.items()}))
for k, v_ in ok:
    print(('PASS ' if v_ else 'FAIL ') + k)
if not all(v_ for _, v_ in ok):
    sys.exit('FAIL full fly S3')
print('PASS full fly S3: gravity and body touch reach the turn readout by side')
