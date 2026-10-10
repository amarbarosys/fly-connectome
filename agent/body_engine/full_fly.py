"""FULL_FLY: one male CNS brain per fly, fed by every sense the world supplies (agent/senses.py, eye_render.py), in the world.

Per control step (K physics steps, 1 ms): every EYE_EVERY steps the pose is rendered from the eye cams into the 6,291
lamina cells; every step the non-visual senses (legs, smell, taste, body touch, gravity) read the engine qpos; the
resident MCNS Shiu brain advances K ticks; leg-MN ridge readout + mirrored DN turn readout (mcns_link steer) -> FlyGym
gait -> engine. The only fixed stimulus is the forward DNs at 80 Hz (the urge to walk, as in every mcns_link mode).
Scenarios (gates pre-registered in docs/plans/FULL_FLY.md):
  u1     wall ahead-left / ahead-right, seeing vs blind twin, mirrored-eye mutant (U1)
  smell  odor blob 12 mm ahead, 45 deg left or right; smelling vs anosmic twin, antennae-swapped mutant (U2 S1)
  taste  food patch under the start; tasting vs tasteless twin (U2 S2)
  all    U1 poses with every sense on vs vision + legs only (U2 S4)
  gpu-wait run --shared --vram 14 --preemptible --timeout 6000 --env PYTHONPATH=$PWD -- .venv/bin/python -m agent.body_engine.full_fly SCENARIO REPEATS SEED
FULL_FLY_SYM=1: the completed left visual path (agent/mcns_sym.py). FULL_FLY_DUMP=path: stim_p of the first 50 steps (S0 check).
"""
import json
import os
import sys
import time

import mujoco  # before play: its MUJOCO_GL default is egl, which fails off the GPU lease
import numpy as np

from agent import fly_vision as fv, senses as sn
from agent.body_engine import fly_link as fl, live
from agent.body_engine.mcns_link import FWD_HZ, WINDOW, brain, drive
from agent.eye_render import Eyes

SYM = os.environ.get('FULL_FLY_SYM') == '1'
EYE_EVERY = 10  # control steps per eye frame: 100 Hz
START = {'L': (-9.0, -4.0, 120), 'R': (-7.0, -7.0, 260)}  # world 1: wall ahead-left / ahead-right (mj_ray, U1)
OPEN = (-7.0, 0.0, 0)  # eye reference pose: open ground (eye_render.set_base)
BASE = ('legs',)  # non-visual senses of the U1 arms
ALL = ('legs', 'smell', 'taste', 'body', 'gravity')


def arm(start, eyes=None, senses=BASE, swap=False, odor=None, food=None):
    """one condition: start pose (x, y, yaw deg), eyes None / 'see' / 'mirror', non-visual senses, side-swapped cells,
    odor (dx, dy) mm and food (dx, dy, r) mm relative to the start position."""
    return dict(start=start, eyes=eyes, senses=senses, swap=swap, odor=odor, food=food)


def scenario(name):
    o = 12 / np.sqrt(2)
    if name == 'u1':
        return {f'{w}-{k}': arm(START[w], eyes) for w in 'LR' for k, eyes in (('see', 'see'), ('blind', None))} | {'L-mirror': arm(START['L'], 'mirror')}
    if name == 'smell':  # open ground; heading 0 = +x, so ahead-left is (+o, +o)
        return {f'{s}-{k}': arm(OPEN, None, ('legs', 'smell') if k != 'anosmic' else BASE, k == 'swap', odor=(o, o if s == 'L' else -o))
                for s in 'LR' for k in ('smell', 'anosmic', 'swap')}
    if name == 'taste':
        return {k: arm(OPEN, None, ('legs', 'taste') if k == 'taste' else BASE, food=(0, 0, 8.0)) for k in ('taste', 'tasteless')}
    if name == 'all':
        return {f'{w}-{k}': arm(START[w], 'see', ALL if k == 'all' else BASE, odor=(8, 0), food=(0, 0, 3.0)) for w in 'LR' for k in ('all', 'base')}
    raise SystemExit(f'FAIL full_fly: unknown scenario {name}')


def trial(arms, repeats, seed):
    """runs every arm x repeats as one batched engine + one brain; returns {(arm, j): metrics} plus timing."""
    from agent.mcns_mapping import descending_name_sets
    params = live.WORLD[:5] + (1,) + live.WORLD[6:]
    d, m, _ = fl.prepare(fl.OUT.parent / 'full-fly-world1', mjcf=live.terrain(fl.MJCF, params)[0], flat=fl.MJCF)
    q0 = np.fromfile(d / 'qpos.bin', np.float32).astype(float)
    cols = [(a, j) for j in range(repeats) for a in arms]

    def pose(x, y, yaw):
        q = q0.copy()
        q[0], q[1] = x, y
        mujoco.mju_mulQuat(q[3:7], np.array([np.cos(np.radians(yaw) / 2), 0, 0, np.sin(np.radians(yaw) / 2)]), q0[3:7].copy())  # yaw on the stance (pitched thorax)
        return q
    qs = {c: pose(*arms[c[0]]['start']) for c in cols}
    eng = fl.Engine(d, fl.K, f'fullfly{seed}', fl.STEPS, F=len(cols), q0=[qs[c] for c in cols])
    gait = {c: fl.Gait(m, d, fl.K) for c in cols}
    rel = lambda c, k: None if arms[c[0]][k] is None else np.r_[qs[c][:2] + arms[c[0]][k][:2], arms[c[0]][k][2:]]
    feel = {c: sn.Senses(m, qs[c], odor=rel(c, 'odor'), food=rel(c, 'food')) for c in cols}
    ids, az, el = fv.receptors(fv.LAMINA)
    eyes = {'see': Eyes(m, ids, az, el)}
    eyes['mirror'] = Eyes(m, ids, az, el, mirror=True, renderer=eyes['see'].r)
    for e in eyes.values():
        e.set_base(pose(*OPEN))
    cells = {False: sn.cells(), True: sn.cells(swap=True)}
    v = np.load('.work/agent/mcns/vision_readout.npz')
    b, f, nl, _ = brain(cols=len(cols), seed=seed, sym=SYM)
    dn, _ = descending_name_sets()
    for ci, c in enumerate(cols):
        b.stimulate(dn['forward'], FWD_HZ, fly=ci, key=ci)
        b.stimulate(ids, 0.0, fly=ci, key=ci)
        for k, x in cells[arms[c[0]]['swap']].items():
            if k[0].split('_')[0] in arms[c[0]]['senses']:
                b.stimulate(x, 0.0, fly=ci, key=ci)
    nm = len(f['mnL']) + len(f['mnR'])
    nc = nm + len(v['dn'])
    b.start([1], np.concatenate([ci * b.n + np.r_[f['mnL'], f['mnR'], v['dn']] for ci in range(len(cols))]).astype(np.int32))
    pos = {e: i for i, e in enumerate(b.stim)}
    eye_slot = {ci: np.array([pos[ci * b.n + x] for x in ids], np.int64) for ci in range(len(cols))}
    slot = {ci: {k: np.array([pos[ci * b.n + x] for x in xs], np.int64) for k, xs in cells[arms[c[0]]['swap']].items() if k[0].split('_')[0] in arms[c[0]]['senses']}
            for ci, c in enumerate(cols)}
    p = np.fromiter(b.stim.values(), np.float32, len(b.stim))

    def act(xc):
        dr = drive(f, xc[:nm], nl)
        t = float(np.clip(((xc[nm:] - v['mu']) / v['sd']) @ v['w'] + v['y0'], -1, 1))  # T > 0: away from a left disc = left drive up
        return np.clip(dr + 0.25 * t * np.array([1.0, -1.0]), 0, 1)  # differential: a left MN drive at ~0.98 clipped a one-sided add (U1)
    dump = os.environ.get('FULL_FLY_DUMP')
    cum, tr, host = [np.zeros(len(cols) * nc)], {c: [qs[c]] for c in cols}, 0.0
    for i in range(fl.STEPS):
        t0 = time.perf_counter()
        for ci, c in enumerate(cols):
            a = arms[c[0]]
            if i % EYE_EVERY == 0 and a['eyes']:
                p[eye_slot[ci]] = eyes[a['eyes']].rates(qs[c]) * b.dt
            for k, hz in feel[c].rates(qs[c]).items():
                if k in slot[ci]:
                    p[slot[ci][k]] = hz * b.dt
        host += time.perf_counter() - t0
        if dump and i < 50:
            with open(dump, 'ab') as fh:
                fh.write(p.tobytes())
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
    out = {c: fl.metrics(m, None, np.array(tr[c])) | {'xy': np.array(tr[c])[-1, :2], 'xy0': np.array(tr[c])[0, :2], 'nan': not np.isfinite(np.array(tr[c])).all()} for c in cols}
    return out, {'host_ms_per_fly_step': 1e3 * host / fl.STEPS / len(cols)}


def paired(a, b):
    d = np.asarray(a) - np.asarray(b)
    return float(d.mean()), float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))) if len(d) > 1 else np.inf


def main(name='u1', repeats=1, seed=7):
    repeats, seed = int(repeats), int(seed)
    if repeats > 1:
        from agent import seeds
        seeds.claim(f'full_fly_{name}' + ('_sym' if SYM else '') if name != 'u1' else ('full_fly_sym' if SYM else 'full_fly'), [seed])
    arms = scenario(name)
    r, timing = trial(arms, repeats, seed)
    J = range(repeats)
    ok = []
    if name == 'u1':
        yaw = lambda a, j: float(r[a, j]['yaw_deg'][-1])
        dl, tl = paired([yaw('L-see', j) for j in J], [yaw('L-blind', j) for j in J])
        dr, tr_ = paired([yaw('R-see', j) for j in J], [yaw('R-blind', j) for j in J])
        dm, _ = paired([yaw('L-mirror', j) for j in J], [yaw('L-blind', j) for j in J])
        fw = float(np.mean([r[a, j]['forward'][-1] for a in ('L-see', 'R-see') for j in J]))
        tilt = max(r[a, j]['tilt_max'] for a in ('L-see', 'R-see') for j in J)
        ok = [(f"seeing walks {fw:.2f} mm > {fl.BODY_LENGTH}, tilt {tilt:.1f} < 20", fw > fl.BODY_LENGTH and tilt < 20),
              (f"away from the wall vs blind twin: wall-left {dl:+.1f} (t {tl:+.1f}) < 0 < wall-right {dr:+.1f} (t {tr_:+.1f}), each >= 10, |t| >= 2.5",
               dl < 0 < dr and min(-dl, dr) >= 10 and min(-tl, tr_) >= 2.5),
              (f"mutant mirrored eyes: wall-left {dm:+.1f}, opposite to seeing {dl:+.1f}", dm * dl < 0)]
    elif name == 'smell':
        dist = lambda a, j: float(np.linalg.norm(r[a, j]['xy'] - (r[a, j]['xy0'] + np.array(arms[a]['odor']))))
        pool = lambda k: [dist(f'{s}-{k}', j) for s in 'LR' for j in J]
        ds, ts = paired(pool('smell'), pool('anosmic'))
        dw, _ = paired(pool('swap'), pool('anosmic'))
        ok = [(f"smelling ends nearer the odor than the anosmic twin: {ds:+.2f} mm (t {ts:+.1f}), mean <= -1, t <= -2.5", ds <= -1 and ts <= -2.5),
              (f"mutant antennae swapped: {dw:+.2f} mm vs anosmic, no nearer (>= {ds:+.2f})", dw >= ds)]
    elif name == 'taste':
        fwd = lambda a: [float(r[a, j]['forward'][-1]) for j in J]
        dt_, tt = paired(fwd('taste'), fwd('tasteless'))
        ok = [(f"tasting fly walks less on food than the tasteless twin: {dt_:+.2f} mm (t {tt:+.1f}), t <= -2.5", tt <= -2.5)]
    elif name == 'all':
        fw = float(np.mean([r[a, j]['forward'][-1] for a in ('L-all', 'R-all') for j in J]))
        tilt = max(r[a, j]['tilt_max'] for a in ('L-all', 'R-all') for j in J)
        nan = any(r[c]['nan'] for c in r)
        ok = [(f"all senses on: walks {fw:.2f} mm > {fl.BODY_LENGTH}, tilt {tilt:.1f} < 20, NaN {nan}", fw > fl.BODY_LENGTH and tilt < 20 and not nan)]
    print(json.dumps({'scenario': name, 'repeats': repeats, 'seed': seed} | timing))
    for k, v in ok:
        print(('PASS ' if v else 'FAIL ') + k)
    if not all(v for _, v in ok):
        sys.exit(f'FAIL full fly {name}')
    print(f'PASS full fly {name}')


if __name__ == '__main__':
    main(*sys.argv[1:])
