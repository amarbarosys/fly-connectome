"""FULL_FLY U6-hybrid: an engineered stepping clock, the brain sets everything else (docs/plans/FULL_FLY.md U6).

Clock: fly_link.Gait (FlyGym HybridTurningController, tripod phases, 42 leg servos + 6 adhesion). Per control step (1 ms)
the MCNS brain (forward DNs 80 Hz, every sense wired as in auto_fly: eyes, legs, smell, taste, body touch, gravity) sets:
  speed  s    = clip(F / F0, 0, 1), F = forward DN population (minus DNg103) spikes per cell per 100 ms
  stop   z    = clip((g - g0) / GS, 0, 1), g = DNg103 (2 cells) spikes per cell per 100 ms (sugar, U2 S2a)
  turn   T    = the DN turn readout (.work/agent/mcns/vision_readout.npz, as full_fly act()), in [-1, 1]
  drive       = s (1 - z) ((1 - KT) + KT T (+1, -1)) -> Gait (left, right)
  posture     = Gait joint targets + GP x leg_mn.targets(named MN rates, 20 ms) x leg_mn.leg_signs flip
Mutant arms read the stop from a count-matched random DN set (fixed draw) with its own g0.
  fit  SEED          4 arms x 4 repeats, stop off, GP 0: prints F0, g0, g0_mut, GS
  sweep SEED         taste / tasteless x GP in GPS x 3 repeats, stop on: posture gain pick
  gate SEED [R]      H1-H3, R repeats (32)
  gpu-wait run --shared --vram 16 --preemptible --timeout 5400 --env PYTHONPATH=$PWD --env MUJOCO_GL=egl -- .venv/bin/python -m agent.body_engine.hybrid_fly MODE SEED
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
from agent.body_engine.mcns_link import WINDOW, brain
from agent.eye_render import Eyes

KT, FOOD_R = 0.25, 8.0  # turn differential (full_fly act); sugar patch radius under the start, mm (U2 S2)
K0 = {'F0': 9.449, 'g0': 11.545, 'g0_mut': 8.239, 'GS': 6.220, 'GP': 0.3, 'T0': -0.373}  # fit seed 430 (GP: sweep seed 431), FULL_FLY.md U6-hybrid mapping
ARMS = {'taste': (True, False), 'tasteless': (False, False), 'mut-taste': (True, True), 'mut-tasteless': (False, True)}  # taste on, mutant stop set
GPS = (0.0, 0.1, 0.3, 1.0)


def stop_sets():
    """DNg103 and a count-matched random draw from the other descending neurons (rng 0)."""
    from agent.mcns_mapping import annotation_table
    a = annotation_table()
    t = a.type.fillna('').to_numpy()
    real = np.flatnonzero(t == 'DNg103')
    pool = np.flatnonzero((a.superclass == 'descending_neuron').to_numpy() & (t != 'DNg103'))
    return real, np.sort(np.random.default_rng(0).choice(pool, len(real), replace=False))


def command(F, g, T, k, g0):
    """readouts -> (left, right) gait drive, speed, stop."""
    s = float(np.clip(F / k['F0'], 0, 1))
    z = float(np.clip((g - g0) / k['GS'], 0, 1)) if k['GS'] else 0.0
    return s * (1 - z) * ((1 - KT) + KT * T * np.array([1.0, -1.0])), s, z


def run(seed, cols, cfg, k, steps=fl.STEPS, keep=()):
    """cols: [(arm, j)]; cfg[arm] = (taste on, mutant stop set, GP[, senses off: 'eyes', 'smell', ...]). Returns per-column metrics and readout means."""
    params = live.WORLD[:5] + (1,) + live.WORLD[6:]
    d, m, _ = fl.prepare(fl.OUT.parent / 'full-fly-world1', mjcf=live.terrain(fl.MJCF, params)[0], flat=fl.MJCF)
    q0 = np.fromfile(d / 'qpos.bin', np.float32).astype(float)
    q = q0.copy()
    q[0], q[1] = OPEN[:2]
    mujoco.mju_mulQuat(q[3:7], np.array([np.cos(np.radians(OPEN[2]) / 2), 0, 0, np.sin(np.radians(OPEN[2]) / 2)]), q0[3:7].copy())
    qs = {c: q.copy() for c in cols}
    feel = {c: sn.Senses(m, q, odor=q[:2] + [8.0, 0.0], food=np.r_[q[:2], FOOD_R]) for c in cols}  # odour 8 mm ahead (as auto_fly), sugar patch under the start
    ids, az, el = fv.receptors(fv.LAMINA)
    eyes = Eyes(m, ids, az, el)
    eyes.set_base(q)
    cells = sn.cells()
    a = pd.read_feather(mcns.DATA / 'body-annotations-male-cns-v1.0-minconf-0.5.feather', columns=['status', 'superclass', 'subclass', 'somaSide', 'type'])
    a = a[a['status'].isin(['Traced', 'Anchor'])].reset_index(drop=True)
    sub, sd, ty = a.subclass.fillna('').to_numpy(), a.somaSide.fillna('').str[:1].to_numpy(), a.type.fillna('').to_numpy()
    leg_mn = {leg: np.flatnonzero((a.superclass == 'vnc_motor').to_numpy() & (sub == f'{leg[1]}l') & (sd == leg[0].upper()) & np.isin(ty, list(L.MAP)))
              for leg in L.LEGS6}
    flip, jsg = L.leg_signs(m, q0), {leg: L.assign(ty[leg_mn[leg]]) for leg in L.LEGS6}
    acts = {leg: [L.act_id(m, leg, j) for j in L.JOINTS] for leg in L.LEGS6}
    v = np.load('.work/agent/mcns/vision_readout.npz')
    real, mut = stop_sets()
    b, _, _, _ = brain(cols=len(cols), seed=seed)  # forward DNs 80 Hz in every column
    from agent.mcns_mapping import descending_name_sets
    fwd = np.setdiff1d(descending_name_sets()[0]['forward'], real)
    read = np.unique(np.r_[np.concatenate(list(leg_mn.values())), v['dn'], fwd, real, mut])
    ix = lambda x: np.searchsorted(read, x)
    i_mn, i_dn, i_fw, i_stop = {leg: ix(leg_mn[leg]) for leg in L.LEGS6}, ix(v['dn']), ix(fwd), {False: ix(real), True: ix(mut)}
    off = {c: cfg[c[0]][3] if len(cfg[c[0]]) > 3 else () for c in cols}
    on = {c: [kk for kk in cells if kk[0].split('_')[0] in ALL and kk[0].split('_')[0] not in off[c] and (cfg[c[0]][0] or not kk[0].startswith('taste'))] for c in cols}
    for ci, c in enumerate(cols):
        if 'eyes' not in off[c]:
            b.stimulate(ids, 0.0, fly=ci, key=ci)
        for kk in on[c]:
            b.stimulate(cells[kk], 0.0, fly=ci, key=ci)
    nr = len(read)
    b.start([1], np.concatenate([ci * b.n + read for ci in range(len(cols))]).astype(np.int32))
    pos = {e: i for i, e in enumerate(b.stim)}
    eye_slot = {ci: np.array([pos[ci * b.n + x] for x in ids], np.int64) for ci, c in enumerate(cols) if 'eyes' not in off[c]}
    slot = {ci: {kk: np.array([pos[ci * b.n + x] for x in cells[kk]], np.int64) for kk in on[c]} for ci, c in enumerate(cols)}
    p = np.fromiter(b.stim.values(), np.float32, len(b.stim))
    eng = fl.Engine(d, fl.K, f'hybrid{seed}', steps, F=len(cols), q0=[qs[c] for c in cols])
    gait = {c: fl.Gait(m, d, fl.K) for c in cols}
    hist = np.zeros((WINDOW + 1, len(cols), nr), np.float32)  # cumulative counts, ring of the last WINDOW + 1 steps (not every step)
    rec = np.zeros((steps, len(cols), 5), np.float32)  # F, g, T, s, z
    tr = np.zeros((steps + 1, len(cols), len(q)), np.float32)
    tr[0] = q
    host, t0 = 0.0, time.perf_counter()
    for i in range(steps):
        h = time.perf_counter()
        for ci, c in enumerate(cols):
            if i % EYE_EVERY == 0 and ci in eye_slot:
                p[eye_slot[ci]] = eyes.rates(qs[c]) * b.dt
            r = feel[c].rates(qs[c])
            for kk, ss in slot[ci].items():
                p[ss] = r.get(kk, 0.0) * b.dt
        host += time.perf_counter() - h
        p.tofile(b.work / 'stim_p.bin')
        cur = b.advance(fl.K).astype(np.float32).reshape(len(cols), nr)
        hist[(i + 1) % (WINDOW + 1)] = cur
        w = min(i + 1, WINDOW)
        x = (cur - hist[(i + 1 - w) % (WINDOW + 1)]) * (WINDOW / w)  # counts per 100 ms
        w2 = min(i + 1, L.WIN)
        mr = (cur - hist[(i + 1 - w2) % (WINDOW + 1)]) / (w2 * 1e-3)  # Hz over 20 ms
        u = []
        for ci, c in enumerate(cols):
            taste, mutant, gp = cfg[c[0]][:3]
            F, g = float(x[ci, i_fw].mean()), float(x[ci, i_stop[mutant]].mean())
            T = float(np.clip(((x[ci, i_dn] - v['mu']) / v['sd']) @ v['w'] + v['y0'] - k.get('T0', 0.0), -1, 1))
            dr, s, z = command(F, g, T, k, k['g0_mut'] if mutant else k['g0'])
            rec[i, ci] = F, g, T, s, z
            ctl = gait[c].ctrl(dr if i + 1 >= WINDOW else np.zeros(2))
            if gp:
                for leg in L.LEGS6:
                    ctl[acts[leg]] += gp * L.targets(mr[ci, i_mn[leg]], *jsg[leg]) * flip[leg]
            u.append(ctl)
        eng.send(u)
        qq = eng.recv().reshape(len(cols), -1)
        for ci, c in enumerate(cols):
            qs[c] = qq[ci]
        tr[i + 1] = qq
    b.close()
    cost = eng.close()
    sec = time.perf_counter() - t0
    out = {}
    for ci, c in enumerate(cols):
        mt = fl.metrics(m, None, tr[:, ci].astype(float))
        rr = rec[WINDOW:, ci]
        out[c] = {'forward': float(mt['forward'][-1]), 'yaw': float(mt['yaw_deg'][-1]), 'tilt': mt['tilt_max'], 'nan': not np.isfinite(tr[:, ci]).all(),
                  'F': float(rr[:, 0].mean()), 'g': float(rr[:, 1].mean()), 'T': float(rr[:, 2].mean()), 'T_sd': float(rr[:, 2].std()),
                  's': float(rr[:, 3].mean()), 'z': float(rr[:, 4].mean())}
    print(f'{sec:.0f} s for {steps} steps x {len(cols)} flies; host senses {1e3 * host / steps / len(cols):.2f} ms per fly-step; engine {cost}')
    if keep:
        tag = keep[0]
        sel = [ci for ci, c in enumerate(cols) if c[1] < keep[1]]
        np.savez_compressed(fl.OUT / f'{tag}-traj.npz', arms=np.array([f'{cols[ci][0]} #{cols[ci][1]}' for ci in sel]), traj=tr[:, sel].transpose(1, 0, 2))  # tools/auto_fly_video.py
    return out


def summary(out, arms):
    for arm in arms:
        rs = [r for c, r in out.items() if c[0] == arm]
        print(arm, json.dumps({kk: round(float(np.mean([r[kk] for r in rs])), 3) for kk in rs[0] if kk != 'nan'} | {'tilt_max': round(max(r['tilt'] for r in rs), 1), 'nan': any(r['nan'] for r in rs)}))


def paired(out, a, b, n):
    dd = np.array([out[a, j]['forward'] - out[b, j]['forward'] for j in range(n)])
    return float(dd.mean()), float(dd.mean() / (dd.std(ddof=1) / np.sqrt(n)))


def fit(seed, reps=4):
    seed = int(seed)
    seeds.claim('u6_hybrid_fit', [seed])
    cols = [(arm, j) for j in range(reps) for arm in ARMS]
    k = {'F0': 1.0, 'g0': 0.0, 'g0_mut': 0.0, 'GS': None}  # stop off, speed unnormalised (clipped at 1)
    out = run(seed, cols, {arm: (t, mu, 0.0) for arm, (t, mu) in ARMS.items()}, k)
    summary(out, ARMS)
    mean = lambda arm, kk: float(np.mean([out[arm, j][kk] for j in range(reps)]))
    fk = {'F0': mean('tasteless', 'F'), 'g0': mean('tasteless', 'g'), 'g0_mut': mean('mut-tasteless', 'g'), 'GS': (mean('taste', 'g') - mean('tasteless', 'g')) / 2}
    print('FIT', json.dumps(fk), 'mut-taste g', round(mean('mut-taste', 'g'), 3))
    (fl.OUT / f'u6-hybrid-fit-{seed}.json').write_text(json.dumps({'k': fk, 'cols': {f'{a} {j}': r for (a, j), r in out.items()}}))


def sweep(seed, reps=3):
    seed = int(seed)
    seeds.claim('u6_hybrid_fit', [seed])
    cfg = {f'{arm} gp{gp}': (arm == 'taste', False, gp) for gp in GPS for arm in ('taste', 'tasteless')}
    out = run(seed, [(a, j) for j in range(reps) for a in cfg], cfg, K0)
    summary(out, cfg)
    (fl.OUT / f'u6-hybrid-sweep-{seed}.json').write_text(json.dumps({f'{a} {j}': r for (a, j), r in out.items()}))


def gate(seed, reps=32):
    seed, reps = int(seed), int(reps)
    seeds.claim('u6_hybrid_gate', [seed])
    if None in K0.values():
        sys.exit('FAIL hybrid gate: constants not fitted (K0)')
    cols = [(arm, j) for j in range(reps) for arm in ARMS]
    out = run(seed, cols, {arm: (t, mu, K0['GP']) for arm, (t, mu) in ARMS.items()}, K0, keep=(f'u6-hybrid-{seed}', 2))
    summary(out, ARMS)
    tl = [out['tasteless', j] for j in range(reps)]
    fw, tilt, nan = float(np.mean([r['forward'] for r in tl])), max(r['tilt'] for r in tl), any(r['nan'] for r in out.values())
    h2, h3 = paired(out, 'taste', 'tasteless', reps), paired(out, 'mut-taste', 'mut-tasteless', reps)
    ok = [(f'H1 tasteless walks {fw:.2f} mm > {fl.BODY_LENGTH} in 1 s, tilt max {tilt:.1f} < 20, NaN {nan}', fw > fl.BODY_LENGTH and tilt < 20 and not nan),
          (f'H2 taste - tasteless forward {h2[0]:+.2f} mm (t {h2[1]:+.1f}), t <= -2.5', h2[1] <= -2.5),
          (f'H3 mutant stop set: taste - tasteless {h3[0]:+.2f} mm (t {h3[1]:+.1f}), |t| < 2.5', abs(h3[1]) < 2.5)]
    (fl.OUT / f'u6-hybrid-gate-{seed}.json').write_text(json.dumps({'k': K0, 'gates': [[s_, v_] for s_, v_ in ok], 'cols': {f'{a} {j}': r for (a, j), r in out.items()}}))
    for s_, v_ in ok:
        print(('PASS ' if v_ else 'FAIL ') + s_)
    if not all(v_ for _, v_ in ok):
        sys.exit('FAIL U6-hybrid')
    print('PASS U6-hybrid')


def smoke():
    """150 steps, one column per arm, placeholder constants, seed 0 (no claim): the loop runs end to end."""
    out = run(0, [(arm, 0) for arm in ARMS], {arm: (t, mu, 0.1) for arm, (t, mu) in ARMS.items()}, {'F0': 8.0, 'g0': 1.0, 'g0_mut': 1.0, 'GS': 5.0}, steps=150)
    summary(out, ARMS)


def probe(seed, reps=4):
    """measured only: which sense sets the turn readout's bias. Tasteless, one sense group off per arm."""
    seed = int(seed)
    seeds.claim('u6_hybrid_probe', [seed])
    cfg = {name: (False, False, K0['GP'], o) for name, o in (('all', ()), ('blind', ('eyes',)), ('no-gravity-body', ('gravity', 'body')),
                                                             ('anosmic', ('smell',)), ('no-legs', ('legs',)), ('none', ('eyes', 'smell', 'gravity', 'body', 'legs')))}
    out = run(seed, [(a, j) for j in range(reps) for a in cfg], cfg, K0)
    summary(out, cfg)
    (fl.OUT / f'u6-hybrid-probe-{seed}.json').write_text(json.dumps({f'{a} {j}': r for (a, j), r in out.items()}))


def check():
    """CPU preflight: stop sets, the command mapping, taste wiring keys."""
    from agent.mcns_mapping import annotation_table
    a = annotation_table()
    real, mut = stop_sets()
    k = {'F0': 8.0, 'GS': 5.0}
    ok = [(f'DNg103 cells {real.tolist()}', len(real) == 2 and set(a.type.to_numpy()[real]) == {'DNg103'}),
          (f'mutant set {mut.tolist()}: descending, not DNg103', len(mut) == 2 and (a.superclass.to_numpy()[mut] == 'descending_neuron').all() and not np.isin(mut, real).any()),
          ('baseline: drive (1 - KT) both sides', np.allclose(command(8.0, 1.0, 0.0, k, 1.0)[0], 1 - KT)),
          ('full stop: drive 0', np.allclose(command(8.0, 6.0, 0.5, k, 1.0)[0], 0)),
          ('T > 0: left drive up', np.subtract(*command(8.0, 1.0, 0.5, k, 1.0)[0]) > 0),
          ('half speed halves the drive', np.allclose(command(4.0, 1.0, 0.0, k, 1.0)[0], (1 - KT) / 2)),
          ('taste cells wired', sum(len(x) for kk, x in sn.cells().items() if kk[0].startswith('taste') and kk[0].split('_')[0] in ALL) == 42)]
    for s_, v_ in ok:
        print(('PASS ' if v_ else 'FAIL ') + s_)
    if not all(v_ for _, v_ in ok):
        sys.exit('FAIL hybrid_fly check')


if __name__ == '__main__':
    {'fit': fit, 'sweep': sweep, 'gate': gate, 'check': check, 'smoke': smoke, 'probe': probe}[sys.argv[1]](*sys.argv[2:])
