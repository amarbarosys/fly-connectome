"""Brain-only probe for FULL_FLY U1: the rendered frames at full_fly's two start poses (and their mirrored-map versions,
and blank) held on the MCNS lamina cells with forward DNs at 80 Hz, no body; per column the DN turn readout T
(mcns_link steer, T > 0 = away from a left disc) and the spikes of LC4+LPLC2 per side. Says whether side information
survives eyes -> brain -> readout before any body is involved.
  gpu-wait run --shared --vram 8 --preemptible --timeout 1200 --env PYTHONPATH=$PWD -- .venv/bin/python tools/full_fly_eye_probe.py"""
import json
import mujoco
import numpy as np
from agent import fly_vision as fv
from agent.body_engine import fly_link as fl, live
from agent.body_engine.full_fly import OPEN, START
from agent.body_engine.mcns_link import FWD_HZ, brain
from agent.eye_render import Eyes
from agent.mcns_mapping import descending_name_sets
from agent.eye_map import annotations

params = live.WORLD[:5] + (1,) + live.WORLD[6:]
d, m, _ = fl.prepare(fl.OUT.parent / 'full-fly-world1', mjcf=live.terrain(fl.MJCF, params)[0], flat=fl.MJCF)
q0 = np.fromfile(d / 'qpos.bin', np.float32).astype(float)

def pose(x, y, yw):
    q = q0.copy()
    q[0], q[1] = x, y
    mujoco.mju_mulQuat(q[3:7], np.array([np.cos(np.radians(yw) / 2), 0, 0, np.sin(np.radians(yw) / 2)]), q0[3:7].copy())
    return q

ids, az, el = fv.receptors(fv.LAMINA)
e = Eyes(m, ids, az, el)
em = Eyes(m, ids, az, el, mirror=True, renderer=e.r)
for x in (e, em):
    x.set_base(pose(*OPEN))
arms = {'blank': np.zeros(len(ids)), 'wallL': e.rates(pose(*START['L'])), 'wallR': e.rates(pose(*START['R'])),
        'wallL-mirror': em.rates(pose(*START['L']))}
v = np.load('.work/agent/mcns/vision_readout.npz')
t = annotations()['type'].fillna('').to_numpy()
side = annotations()['somaSide'].fillna('').str[:1].str.upper().to_numpy()
lc = {s: np.flatnonzero(np.isin(t, ['LC4', 'LPLC2']) & (side == s)) for s in 'LR'}
dn, _ = descending_name_sets()
b, f, nl, _ = brain(cols=len(arms), seed=7)
for c, hz in enumerate(arms.values()):
    b.stimulate(dn['forward'], FWD_HZ, fly=c, key=c)
    for r in np.unique(hz[hz > 0]):
        b.stimulate(ids[hz == r], float(r), fly=c, key=c)
cells = np.r_[v['dn'], lc['L'], lc['R']]
b.start([1], np.concatenate([c * b.n + cells for c in range(len(arms))]).astype(np.int32))
N = 500  # 0.5 s, the readout's fit window scale
x = b.advance(N * fl.K).astype(float).reshape(len(arms), len(cells)) * (100 / N)  # counts per 100 ms
b.close()
nd = len(v['dn'])
out = {}
for c, k in enumerate(arms):
    T = float(((x[c, :nd] - v['mu']) / v['sd']) @ v['w'] + v['y0'])
    out[k] = {'T': T, 'LC_L': float(x[c, nd:nd + len(lc['L'])].sum()), 'LC_R': float(x[c, nd + len(lc['L']):].sum()),
              'eye_Hz_L': float(arms[k][az > 0].mean()), 'eye_Hz_R': float(arms[k][az <= 0].mean())}
    print(k, {a: round(b_, 2) for a, b_ in out[k].items()})
(fl.OUT / 'full-fly-eye-probe.json').write_text(json.dumps(out))
