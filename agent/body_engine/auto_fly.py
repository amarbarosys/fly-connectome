"""Everything wired, no gait controller: one MCNS brain per fly, every sense in (agent/senses.py: vision, legs, smell,
taste, body touch, gravity), every named leg motor neuron out to its own servo (agent/body_engine/leg_mn.py map and
per-leg signs; 42 leg servos). Kept engineered: the forward DNs at 80 Hz (the urge to walk) and foot adhesion on while
that foot touches the ground. Optional spike adaptation per column (leg_mn.adapt_brain). Measured only, no gate yet:
distance, tilt, NaN, MN flexor / extensor alternation and the legs' tibia correlations, per arm.
  gpu-wait run --shared --vram 16 --preemptible --timeout 5400 --env PYTHONPATH=$PWD --env MUJOCO_GL=egl -- .venv/bin/python -m agent.body_engine.auto_fly SEED [STEPS] [DROP_MM]
Arms: all (every sense, adaptation 0), all-adapt (every sense, adaptation 1), none (no senses, adaptation 0).
"""
import json
import sys
import time

import mujoco  # before play (MUJOCO_GL)
import numpy as np
import pandas as pd

from agent import fly_vision as fv, mcns, seeds, senses as sn
from agent.body_engine import fly_link as fl, leg_mn as L, live
from agent.body_engine.full_fly import ALL, EYE_EVERY, OPEN
from agent.eye_render import Eyes

ARMS = {'all': (True, 0.0), 'all-adapt': (True, 1.0), 'none': (False, 0.0)}  # senses on, adaptation strength
REPEATS = 2


def main(seed, steps=2000, drop=0.0, ring=0.0):
    """drop: start this many mm above the standing height (the push-start test: does a landing jolt start a rhythm?)."""
    seed, steps, drop, ring = int(seed), int(steps), float(drop), float(ring)
    seeds.claim('auto_fly' if not drop else 'auto_fly_drop', [seed])
    cols = [(a, j) for j in range(REPEATS) for a in ARMS]
    params = live.WORLD[:5] + (1,) + live.WORLD[6:]
    d, m, _ = fl.prepare(fl.OUT.parent / 'full-fly-world1', mjcf=live.terrain(fl.MJCF, params)[0], flat=fl.MJCF)
    q0 = np.fromfile(d / 'qpos.bin', np.float32).astype(float)
    held = np.fromfile(d / 'ctrl.bin', np.float32).astype(np.float64)
    q = q0.copy()
    q[0], q[1] = OPEN[:2]
    q[2] += drop
    mujoco.mju_mulQuat(q[3:7], np.array([np.cos(np.radians(OPEN[2]) / 2), 0, 0, np.sin(np.radians(OPEN[2]) / 2)]), q0[3:7].copy())
    qs = {c: q.copy() for c in cols}
    for k, c in enumerate(cols):  # ring: start each fly on a circle of this radius (mm) so one scene can show them all
        qs[c][:2] += (ring > 0) * np.array([10.0, 0.0]) + ring * np.array([np.cos(2 * np.pi * k / len(cols)), np.sin(2 * np.pi * k / len(cols))])  # ring centre 10 mm +x: open ground, clear of world 1's stone
    feel = {c: sn.Senses(m, qs[c], odor=qs[c][:2] + [8.0, 0.0], food=np.r_[qs[c][:2], 3.0]) for c in cols}  # odour 8 mm ahead, food under the start (as full_fly all)
    ids, az, el = fv.receptors(fv.LAMINA)
    eyes = Eyes(m, ids, az, el)
    eyes.set_base(q)
    cells = sn.cells()
    # outputs: named MNs of all six legs, per-leg signs from the stance pose
    a = pd.read_feather(mcns.DATA / 'body-annotations-male-cns-v1.0-minconf-0.5.feather', columns=['status', 'superclass', 'subclass', 'somaSide', 'type'])
    a = a[a['status'].isin(['Traced', 'Anchor'])].reset_index(drop=True)
    sub, sd, ty = a.subclass.fillna('').to_numpy(), a.somaSide.fillna('').str[:1].to_numpy(), a.type.fillna('').to_numpy()
    leg_mn = {leg: np.flatnonzero((a.superclass == 'vnc_motor').to_numpy() & (sub == f'{leg[1]}l') & (sd == leg[0].upper()) & np.isin(ty, list(L.MAP)))
              for leg in L.LEGS6}
    flip = L.leg_signs(m, q0)
    acts = {leg: [L.act_id(m, leg, j) for j in L.JOINTS] for leg in L.LEGS6}
    adh = {leg: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f'fly/{leg}_tarsus5-adhesion') for leg in L.LEGS6}
    jsg = {leg: L.assign(ty[leg_mn[leg]]) for leg in L.LEGS6}
    out = np.concatenate(list(leg_mn.values()))
    span = np.cumsum([0] + [len(leg_mn[leg]) for leg in L.LEGS6])
    b = L.adapt_brain(seed, [ARMS[c[0]][1] for c in cols])
    for ci, c in enumerate(cols):
        if ARMS[c[0]][0]:
            b.stimulate(ids, 0.0, fly=ci, key=ci)
            for k, x in cells.items():
                if k[0] in ALL:
                    b.stimulate(x, 0.0, fly=ci, key=ci)
    b.start([1], np.concatenate([ci * b.n + out for ci in range(len(cols))]).astype(np.int32))
    pos = {e: i for i, e in enumerate(b.stim)}
    eye_slot = {ci: np.array([pos[ci * b.n + x] for x in ids], np.int64) for ci, c in enumerate(cols) if ARMS[c[0]][0]}
    slot = {ci: {k: np.array([pos[ci * b.n + x] for x in xs], np.int64) for k, xs in cells.items() if k[0] in ALL}
            for ci, c in enumerate(cols) if ARMS[c[0]][0]}
    p = np.fromiter(b.stim.values(), np.float32, len(b.stim))
    eng = fl.Engine(d, fl.K, f'autofly{seed}', steps, F=len(cols), q0=[qs[c] for c in cols])
    cum, tr, mnc, t0, host = [np.zeros(len(cols) * len(out))], {c: [] for c in cols}, [], time.perf_counter(), 0.0
    for i in range(steps):
        h = time.perf_counter()
        touch = {}
        for ci, c in enumerate(cols):
            r = feel[c].rates(qs[c])
            touch[c] = {leg: r.get(('legs', (leg, 'touch')), 0.0) > 0 for leg in L.LEGS6}
            if ci in slot:
                if i % EYE_EVERY == 0:
                    p[eye_slot[ci]] = eyes.rates(qs[c]) * b.dt
                for k, hz in r.items():
                    if k in slot[ci]:
                        p[slot[ci][k]] = hz * b.dt
        host += time.perf_counter() - h
        p.tofile(b.work / 'stim_p.bin')
        cum.append(b.advance(fl.K).astype(float))
        mnc.append((cum[-1] - cum[-2]).reshape(len(cols), len(out)).astype(np.uint8))
        w = min(i + 1, L.WIN)
        rate = ((cum[-1] - cum[-1 - w]) / (w * 1e-3)).reshape(len(cols), len(out))
        u = []
        for ci, c in enumerate(cols):
            x = held.copy()
            for li, leg in enumerate(L.LEGS6):
                x[acts[leg]] = held[acts[leg]] + L.targets(rate[ci, span[li]:span[li + 1]], *jsg[leg]) * flip[leg]
                x[adh[leg]] = 1.0 if touch[c][leg] else 0.0
            u.append(x)
        eng.send(u)
        qq = eng.recv().reshape(len(cols), -1)
        for ci, c in enumerate(cols):
            qs[c] = qq[ci]
            tr[c].append(qq[ci])
    b.close()
    eng.close()
    sec = time.perf_counter() - t0
    mnc = np.array(mnc)  # (steps, cols, MNs)
    res = {}
    for arm in ARMS:
        cs = [c for c in cols if c[0] == arm]
        mt = [fl.metrics(m, None, np.array(tr[c])) for c in cs]
        alt, cor = [], []
        for c in cs:
            ci = cols.index(c)
            ang = np.degrees(np.array(tr[c]))
            tib = np.array([ang[:, m.jnt_qposadr[m.actuator_trnid[acts[leg][5], 0]]] for leg in L.LEGS6])
            cor.append([round(float(np.corrcoef(tib[0], tib[k])[0, 1]), 2) for k in range(6)])
            for li, leg in enumerate(L.LEGS6):
                tl = ty[leg_mn[leg]]
                bins = lambda msk: mnc[:steps // 10 * 10, ci, span[li]:span[li + 1]][:, msk].sum(1).reshape(-1, 10).sum(1).astype(float)
                F, E = bins(np.isin(tl, ['Ti flexor MN', 'Acc. ti flexor MN'])), bins(tl == 'Ti extensor MN')
                if F.std() and E.std():
                    alt.append(float(np.corrcoef(F, E)[0, 1]))
        res[arm] = {'forward_mm': [round(float(x['forward'][-1]), 2) for x in mt], 'tilt_max': [round(float(x['tilt_max']), 1) for x in mt],
                    'nan': any(not np.isfinite(np.array(tr[c])).all() for c in cs), 'mn_firing': int((mnc[:, [cols.index(c) for c in cs]].sum(0) > 0).sum() / len(cs)),
                    'tibia_flex_ext_corr': round(float(np.mean(alt)), 2) if alt else None, 'tibia_corr_vs_lf': cor}
        print(arm, json.dumps(res[arm]))
    print(f'{sec:.0f} s for {steps} steps x {len(cols)} flies; host senses {1e3 * host / steps / len(cols):.2f} ms per fly-step')
    tag = f'auto-fly-{seed}{"-drop" if drop else ""}'
    (fl.OUT / f'{tag}.json').write_text(json.dumps(res))
    np.savez_compressed(fl.OUT / f'{tag}-traj.npz', arms=np.array([f'{c[0]} #{c[1]}' for c in cols]), traj=np.array([tr[c] for c in cols], np.float32))  # tools/auto_fly_video.py


if __name__ == '__main__':
    main(*sys.argv[1:])
