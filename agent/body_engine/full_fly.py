"""FULL_FLY U1: one male CNS brain per fly, with what the fly sees and feels as its input, in the walled world.

Per control step (K physics steps, 1 ms): every EYE_EVERY steps the engine pose is rendered from the two eye cams into
the 6,291 R1-6/L1-L3 cells (agent/eye_render.py); every step qpos -> 2,330 leg sensory neurons (agent/fly_legsense.py);
the resident MCNS Shiu brain advances K ticks; leg-MN ridge readout + mirrored DN turn readout (mcns_link steer)
-> FlyGym gait -> engine. The only fixed stimulus is the forward DNs at 80 Hz (a stand-in for the urge to walk,
as in every mcns_link mode); nothing is scripted after it.
Columns per repeat: wall ahead-left and ahead-right, each seeing and blind (no eye input), plus wall-left through the
mirrored eye map (mutant). Start poses (world 1, yaw composed onto the stance quaternion, mj_ray from c_thorax, rays at
-75..+75 deg): wall-left 5.0 mm at +45 deg, 8.0 at +75, rays at 0 and right > 10 mm (-9, -4, yaw 120); wall-right its
mirror within 8.6 mm summed over 7 rays (3.9 at -45, 7.8 at -75; -7, -7, yaw 260).
Gates (1 s, pre-registered, fresh seed): yaw of seeing minus blind twin away from the wall, wall-left < 0 < wall-right,
each |mean| >= 10 deg and |t| >= 2.5; mutant mirrored map turns the other way from seeing wall-left; seeing walks > 1
body length, tilt < 20 deg.
  gpu-wait run --shared --vram 8 --priority 60 --timeout 3000 --env PYTHONPATH=$PWD -- .venv/bin/python -m agent.body_engine.full_fly REPEATS SEED
"""
import json
import sys

import mujoco  # before play: its MUJOCO_GL default is egl, which fails off the GPU lease
import numpy as np

from agent import fly_legsense as ls, fly_vision as fv
from agent.body_engine import fly_link as fl, live
from agent.body_engine.mcns_link import FWD_HZ, WINDOW, brain, drive
from agent.eye_render import Eyes

EYE_EVERY = 10  # control steps per eye frame: 100 Hz
START = {'L': (-9.0, -4.0, 120), 'R': (-7.0, -7.0, 260)}
ARMS = (('L', 'see'), ('L', 'blind'), ('R', 'see'), ('R', 'blind'), ('L', 'mirror'))


def main(repeats=1, seed=7):
    from agent.mcns_mapping import descending_name_sets
    if repeats > 1:
        from agent import seeds
        seeds.claim('full_fly', [seed])
    params = live.WORLD[:5] + (1,) + live.WORLD[6:]
    d, m, _ = fl.prepare(fl.OUT.parent / 'full-fly-world1', mjcf=live.terrain(fl.MJCF, params)[0], flat=fl.MJCF)
    q0 = np.fromfile(d / 'qpos.bin', np.float32).astype(float)
    cols = [(w, k, j) for j in range(repeats) for w, k in ARMS]

    def start(w):
        q = q0.copy()
        x, y, yaw = START[w]
        q[0], q[1] = x, y
        mujoco.mju_mulQuat(q[3:7], np.array([np.cos(np.radians(yaw) / 2), 0, 0, np.sin(np.radians(yaw) / 2)]), q0[3:7].copy())  # yaw on top of the stance (pitched thorax)
        return q
    qs = {c: start(c[0]) for c in cols}
    eng = fl.Engine(d, fl.K, f'fullfly{seed}', fl.STEPS, F=len(cols), q0=[qs[c] for c in cols])
    gait = {c: fl.Gait(m, d, fl.K) for c in cols}
    feel = {c: ls.Sense(m, qs[c]) for c in cols}
    ids, az, el = fv.receptors(fv.LAMINA)
    eyes = Eyes(m, ids, az, el)
    eyes_m = Eyes(m, ids, az, el, mirror=True, renderer=eyes.r)
    legs = ls.cells()
    v = np.load('.work/agent/mcns/vision_readout.npz')
    b, f, nl, _ = brain(cols=len(cols), seed=seed)
    dn, _ = descending_name_sets()
    for c in range(len(cols)):
        b.stimulate(dn['forward'], FWD_HZ, fly=c, key=c)
        b.stimulate(ids, 0.0, fly=c, key=c)
        for cells in legs.values():
            b.stimulate(cells, 0.0, fly=c, key=c)
    nm = len(f['mnL']) + len(f['mnR'])
    nc = nm + len(v['dn'])
    b.start([1], np.concatenate([c * b.n + np.r_[f['mnL'], f['mnR'], v['dn']] for c in range(len(cols))]).astype(np.int32))
    pos = {e: i for i, e in enumerate(b.stim)}
    eye_slot = {c: np.array([pos[c * b.n + x] for x in ids], np.int64) for c in range(len(cols))}
    leg_slot = {(c, g): np.array([pos[c * b.n + x] for x in legs[g]], np.int64) for c in range(len(cols)) for g in legs}
    p = np.fromiter(b.stim.values(), np.float32, len(b.stim))

    def act(xc):
        dr = drive(f, xc[:nm], nl)
        t = float(np.clip(((xc[nm:] - v['mu']) / v['sd']) @ v['w'] + v['y0'], -1, 1))  # T > 0: away from a left disc = left drive up
        return np.clip(dr + 0.5 * np.array([max(t, 0), max(-t, 0)]), 0, 1)
    cum, tr = [np.zeros(len(cols) * nc)], {c: [qs[c]] for c in cols}
    for i in range(fl.STEPS):
        for ci, c in enumerate(cols):
            if i % EYE_EVERY == 0 and c[1] != 'blind':
                p[eye_slot[ci]] = (eyes_m if c[1] == 'mirror' else eyes).rates(qs[c]) * b.dt
            for g, hz in feel[c](qs[c]).items():
                p[leg_slot[ci, g]] = hz * b.dt
        p.tofile(b.work / 'stim_p.bin')
        cum.append(b.advance(fl.K).astype(float))
        x = (cum[-1] - cum[-1 - min(i + 1, WINDOW)]).reshape(len(cols), nc) * (WINDOW / min(i + 1, WINDOW))
        eng.send([gait[c].ctrl(act(x[ci]) if i + 1 >= WINDOW else np.zeros(2)) for ci, c in enumerate(cols)])
        q = eng.recv().reshape(len(cols), -1)
        for ci, c in enumerate(cols):
            qs[c] = q[ci]
            tr[c].append(q[ci])
    b.close()
    eng.close()
    r = {c: fl.metrics(m, None, np.array(tr[c])) for c in cols}
    yaw = {c: float(r[c]['yaw_deg'][-1]) for c in cols}
    rel = {(w, k): np.array([yaw[w, k, j] - yaw[w, 'blind', j] for j in range(repeats)]) for w, k in (('L', 'see'), ('R', 'see'), ('L', 'mirror'))}
    tt = {a: float(x.mean() / (x.std(ddof=1) / np.sqrt(repeats))) if repeats > 1 else np.inf for a, x in rel.items()}
    dl, dr, dm = (float(rel[a].mean()) for a in (('L', 'see'), ('R', 'see'), ('L', 'mirror')))
    fw = float(np.mean([r[w, 'see', j]['forward'][-1] for w in 'LR' for j in range(repeats)]))
    tilt = max(r[w, 'see', j]['tilt_max'] for w in 'LR' for j in range(repeats))
    out = {'yaw_deg': {f'{w}-{k}-{j}': y for (w, k, j), y in yaw.items()}, 'rel_mean': [dl, dr, dm], 'forward_mm': fw, 'tilt': tilt}
    print(json.dumps(out))
    ok = [(f"seeing walks {fw:.2f} mm > {fl.BODY_LENGTH}, tilt {tilt:.1f} < 20", fw > fl.BODY_LENGTH and tilt < 20),
          (f"away from the wall vs blind twin: wall-left {dl:+.1f} (t {tt['L', 'see']:+.1f}) < 0 < wall-right {dr:+.1f} (t {tt['R', 'see']:+.1f}), each >= 10, |t| >= 2.5",
           dl < 0 < dr and min(-dl, dr) >= 10 and min(-tt['L', 'see'], tt['R', 'see']) >= 2.5),
          (f"mutant mirrored eyes: wall-left {dm:+.1f}, opposite to seeing {dl:+.1f}", dm * dl < 0)]
    for k, ok_ in ok:
        print(('PASS ' if ok_ else 'FAIL ') + k)
    (fl.OUT / f'full-fly-{repeats}-{seed}.json').write_text(json.dumps(out))
    if not all(ok_ for _, ok_ in ok):
        sys.exit('FAIL full fly U1')
    print('PASS full fly U1: what the fly sees and feels turns it away from walls')


if __name__ == '__main__':
    main(*map(int, sys.argv[1:]))
