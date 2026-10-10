"""FULL_FLY U4a (attempt main): the left front leg driven by its own MCNS motor neurons, map from MN type names only.

Brain: the MCNS Shiu brain (mcns_link.brain), forward DNs off; per column one MN group (a joint's agonist or antagonist
types, docs/plans/FULL_FLY.md U4a table) at 150 Hz. Body: the standing fly in the engine (flat floor), the other legs and
the head at their standing targets, lf adhesion off; per control step (1 ms) the lf servo targets = standing angle +
GAIN x (mean agonist rate - mean antagonist rate) over the last 20 ms. Mutant: the type -> joint assignment permuted.
G1: every testable joint moves the predicted way under its agonist and the opposite way under its antagonist, each
|change| >= 5 deg; G2: the permuted map passes on at most 2 of the 5. Measured only: forward DNs 80 Hz, no MN stimulus,
lf joint power at 5-15 Hz.
  gpu-wait run --shared --vram 14 --preemptible --timeout 3000 --env PYTHONPATH=$PWD -- .venv/bin/python -m agent.body_engine.leg_mn SEED
"""
import json
import sys
import time

import mujoco  # before play (MUJOCO_GL)
import numpy as np
import pandas as pd

from agent import mcns, seeds
from agent.body_engine import fly_link as fl
from agent.body_engine.mcns_link import FWD_HZ, brain
from agent.mcns_mapping import descending_name_sets

JOINTS = ('coxa-yaw', 'coxa-pitch', 'coxa-roll', 'trochanterfemur-pitch', 'trochanterfemur-roll', 'tibia-pitch', 'tarsus1-pitch')
MAP = {  # MN type -> (joint, sign); signs from the MJCF stance pose (FULL_FLY.md U4a)
    'Pleural remotor/abductor MN': ('coxa-pitch', 1), 'Tergopleural/Pleural promotor MN': ('coxa-pitch', -1),
    'Sternal posterior rotator MN': ('coxa-roll', 1), 'Sternal anterior rotator MN': ('coxa-roll', -1),
    'Sternal adductor MN': ('coxa-yaw', -1),
    'Tr extensor MN': ('trochanterfemur-pitch', 1), 'Sternotrochanter MN': ('trochanterfemur-pitch', 1), 'Tergotr. MN': ('trochanterfemur-pitch', 1),
    'Tr flexor MN': ('trochanterfemur-pitch', -1), 'Acc. tr flexor MN': ('trochanterfemur-pitch', -1),
    'Fe reductor MN': ('trochanterfemur-roll', 1),
    'Ti flexor MN': ('tibia-pitch', 1), 'Acc. ti flexor MN': ('tibia-pitch', 1), 'Ti extensor MN': ('tibia-pitch', -1),
    'Ta depressor MN': ('tarsus1-pitch', 1), 'ltm MN': ('tarsus1-pitch', 1), 'ltm1-tibia MN': ('tarsus1-pitch', 1), 'ltm2-femur MN': ('tarsus1-pitch', 1),
    'Ta levator MN': ('tarsus1-pitch', -1)}
TESTED = ('coxa-pitch', 'coxa-roll', 'trochanterfemur-pitch', 'tibia-pitch', 'tarsus1-pitch')
GAIN, HZ, WIN, STEPS, MOVE = 0.005, 150.0, 20, 200, 5.0  # rad/Hz; stimulus Hz; rate window and run length (control steps of 1 ms); deg


def mns():
    a = pd.read_feather(mcns.DATA / 'body-annotations-male-cns-v1.0-minconf-0.5.feather', columns=['status', 'superclass', 'subclass', 'somaSide', 'type'])
    a = a[a['status'].isin(['Traced', 'Anchor'])].reset_index(drop=True)
    m = (a.superclass == 'vnc_motor') & (a.subclass == 'fl') & (a.somaSide.fillna('').str[:1] == 'L') & a.type.isin(list(MAP))
    return np.flatnonzero(m.to_numpy()), a.type.to_numpy()[m.to_numpy()]


def assign(types, shuffle=False):
    """per MN: (joint index, sign); shuffle: the type -> joint assignment permuted (seed 0), signs kept."""
    names = sorted(MAP)
    joint = {t: MAP[t][0] for t in names}
    if shuffle:
        joint = dict(zip(names, np.random.default_rng(0).permutation([joint[t] for t in names])))
    return np.array([JOINTS.index(joint[t]) for t in types]), np.array([MAP[t][1] for t in types], float)


def targets(rate, jn, sg):
    """rate (cells,) Hz -> (7,) offsets in rad: GAIN x (mean agonist rate - mean antagonist rate) per joint."""
    off = np.zeros(len(JOINTS))
    for j in range(len(JOINTS)):
        for s in (1, -1):
            k = (jn == j) & (sg == s)
            if k.any():
                off[j] += s * GAIN * rate[k].mean()
    return np.clip(off, -1, 1)


def main(seed):
    seed = int(seed)
    seeds.claim('leg_mn_u4a', [seed])
    ids, types = mns()
    real, mut = assign(types), assign(types, True)
    conds = [(j, s) for j in TESTED for s in (1, -1)] + [('walk', 0)]  # brain columns: a joint's agonist / antagonist group, then forward DNs only
    d, m, _ = fl.prepare(fl.OUT.parent / 'full-fly-flat', mjcf=fl.MJCF)
    q0 = np.fromfile(d / 'qpos.bin', np.float32).astype(float)
    held = np.fromfile(d / 'ctrl.bin', np.float32).astype(np.float64)
    act = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f'fly/{p}-lf_{j.split("-")[0]}-{j.split("-")[1]}-position')
           for p, j in zip(('c_thorax', 'c_thorax', 'c_thorax', 'lf_coxa', 'lf_coxa', 'lf_trochanterfemur', 'lf_tibia'), JOINTS)]
    adh = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, 'fly/lf_tarsus5-adhesion')
    if min(act + [adh]) < 0:
        sys.exit('FAIL leg_mn: an lf actuator name did not resolve')
    qadr = m.jnt_qposadr[m.actuator_trnid[act, 0]]
    rest = held[act]
    dn, _ = descending_name_sets()
    b, *_ = brain(cols=len(conds), seed=seed)
    for c, (j, s) in enumerate(conds):
        b.stimulate(dn['forward'], FWD_HZ if j == 'walk' else 0.0, fly=c, key=c)
        if j != 'walk':
            g = (real[0] == JOINTS.index(j)) & (real[1] == s)
            b.stimulate(ids[g], HZ, fly=c, key=c)
    b.start([1], np.concatenate([c * b.n + ids for c in range(len(conds))]).astype(np.int32))
    cols = [(c, mp) for mp in ('real', 'mutant') for c in range(len(conds))]  # engine columns: each brain column through both maps
    eng = fl.Engine(d, fl.K, f'legmn{seed}', STEPS, F=len(cols), q0=[q0] * len(cols))
    cum, tr, t0 = [np.zeros(len(conds) * len(ids))], [], time.perf_counter()
    for i in range(STEPS):
        cum.append(b.advance(fl.K).astype(float))
        w = min(i + 1, WIN)
        rate = ((cum[-1] - cum[-1 - w]) / (w * 1e-3)).reshape(len(conds), len(ids))
        u = []
        for c, mp in cols:
            x = held.copy()
            x[act] = rest + targets(rate[c], *(real if mp == 'real' else mut))
            x[adh] = 0.0
            u.append(x)
        eng.send(u)
        tr.append(eng.recv()[:, qadr])
    b.close()
    eng.close()
    sec = time.perf_counter() - t0
    tr = np.degrees(np.array(tr))  # (steps, engine columns, 7)
    ch = tr[-1] - tr[0]
    res, ok = {}, []
    for mp in ('real', 'mutant'):
        good = 0
        for j in TESTED:
            ji = JOINTS.index(j)
            ag, an = (cols.index((conds.index((j, s)), mp)) for s in (1, -1))
            a_, n_ = float(ch[ag, ji]), float(ch[an, ji])
            p = a_ >= MOVE and n_ <= -MOVE
            good += p
            res[f'{mp} {j}'] = {'agonist_deg': round(a_, 1), 'antagonist_deg': round(n_, 1), 'pass': bool(p)}
        res[f'{mp} passed'] = good
    wc = cols.index((conds.index(('walk', 0)), 'real'))
    f = np.fft.rfftfreq(STEPS, 1e-3)
    pw = {JOINTS[k]: float((np.abs(np.fft.rfft(tr[:, wc, k] - tr[:, wc, k].mean())) ** 2)[(f >= 5) & (f <= 15)].sum()
                           / ((np.abs(np.fft.rfft(tr[:, wc, k] - tr[:, wc, k].mean())) ** 2)[1:].sum() + 1e-12)) for k in range(len(JOINTS))}
    res['walk_5_15Hz_power_fraction'] = {k: round(v, 3) for k, v in pw.items()}
    res['walk_range_deg'] = {JOINTS[k]: round(float(np.ptp(tr[:, wc, k])), 1) for k in range(len(JOINTS))}
    res['mn_cells'] = len(ids)
    res['mean_rate_hz'] = {f'{j} {s:+d}': round(float(rate[conds.index((j, s))].mean()), 1) for j, s in conds if j != 'walk'}
    print(json.dumps(res))
    print(f'{sec:.1f} s for {STEPS} steps x {len(cols)} engine columns')
    (fl.OUT / f'leg-mn-u4a-{seed}.json').write_text(json.dumps(res))
    ok = [(f"G1 real map: {res['real passed']} of 5 joints move agonist >= +{MOVE} and antagonist <= -{MOVE} deg", res['real passed'] == 5),
          (f"G2 mutant map: {res['mutant passed']} of 5 <= 2", res['mutant passed'] <= 2)]
    for k, v in ok:
        print(('PASS ' if v else 'FAIL ') + k)
    if not all(v for _, v in ok):
        sys.exit('FAIL U4a')
    print('PASS U4a: the left front leg moves by its own motor neurons')


if __name__ == '__main__' and sys.argv[1:2] != ['watch']:
    main(*sys.argv[1:])


LEGS6 = ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')
METRIC = {  # U4b sign rule: the change in this quantity for +0.2 rad, sign as in lf
    'coxa-yaw': lambda g, s: s * g['tip'][1], 'coxa-pitch': lambda g, s: -g['tip'][0], 'coxa-roll': lambda g, s: -g['tip'][0],
    'trochanterfemur-pitch': lambda g, s: -g['cf'], 'trochanterfemur-roll': lambda g, s: 1.0,
    'tibia-pitch': lambda g, s: g['ft'], 'tarsus1-pitch': lambda g, s: -g['tip'][2]}


def leg_signs(m, q0):
    """{leg: (7,) +1/-1}: each joint's direction relative to the anatomical + of MAP, from the stance pose (U4b rule)."""
    d = mujoco.MjData(m)
    bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'fly/' + n)
    ang = lambda a, b: np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)))
    out = {}
    for leg in LEGS6:
        cx, fe, ti, ta, tip = (bid(f'{leg}_{n}') for n in ('coxa', 'trochanterfemur', 'tibia', 'tarsus1', 'tarsus5'))
        def geo(q):
            d.qpos[:] = q
            mujoco.mj_kinematics(m, d)
            P, R = d.xpos, d.xmat[bid('c_thorax')].reshape(3, 3)
            return {'tip': R.T @ (P[tip] - P[cx]), 'cf': ang(P[fe] - P[cx], P[ti] - P[fe]), 'ft': ang(P[ti] - P[fe], P[ta] - P[ti])}
        g0, sg = geo(q0), []
        for j in JOINTS:
            a = act_id(m, leg, j)
            q = q0.copy()
            q[m.jnt_qposadr[m.actuator_trnid[a, 0]]] += 0.2
            g = geo(q)
            dg = {k: g[k] - g0[k] for k in g}
            sg.append(1.0 if METRIC[j](dg, 1 if leg[0] == 'l' else -1) > 0 else -1.0)
        out[leg] = np.array(sg)
    return out


def act_id(m, leg, j):
    parent = {'coxa': 'c_thorax', 'trochanterfemur': f'{leg}_coxa', 'tibia': f'{leg}_trochanterfemur', 'tarsus1': f'{leg}_tibia'}[j.split('-')[0]]
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f'fly/{parent}-{leg}_{j.split("-")[0]}-{j.split("-")[1]}-position')


def watch(seed, steps=1000):
    """What fires: forward DNs 80 Hz, no MN stimulus, all six legs' named MNs mapped to their own leg (U4a rule)."""
    seed = int(seed)
    seeds.claim('leg_mn_watch', [seed])
    a = pd.read_feather(mcns.DATA / 'body-annotations-male-cns-v1.0-minconf-0.5.feather', columns=['status', 'superclass', 'subclass', 'somaSide', 'type'])
    a = a[a['status'].isin(['Traced', 'Anchor'])].reset_index(drop=True)
    sub, sd, ty = a.subclass.fillna('').to_numpy(), a.somaSide.fillna('').str[:1].to_numpy(), a.type.fillna('').to_numpy()
    leg_of = {f'{s.lower()}{p[0]}': (p, s) for p in ('fl', 'ml', 'hl') for s in 'LR'}
    cells = {leg: np.flatnonzero((a.superclass == 'vnc_motor').to_numpy() & (sub == leg_of[leg][0]) & (sd == leg_of[leg][1].upper()) & np.isin(ty, list(MAP)))
             for leg in LEGS6}
    allmn = np.flatnonzero((a.superclass == 'vnc_motor').to_numpy() & np.isin(sub, ['fl', 'ml', 'hl']))
    d, m, _ = fl.prepare(fl.OUT.parent / 'full-fly-flat', mjcf=fl.MJCF)
    q0 = np.fromfile(d / 'qpos.bin', np.float32).astype(float)
    held = np.fromfile(d / 'ctrl.bin', np.float32).astype(np.float64)
    flip = leg_signs(m, q0)
    acts = {leg: [act_id(m, leg, j) for j in JOINTS] for leg in LEGS6}
    qadr = {leg: m.jnt_qposadr[m.actuator_trnid[acts[leg], 0]] for leg in LEGS6}
    b, *_ = brain(cols=1, seed=seed)  # brain() drives the forward DNs at 80 Hz
    b.start([1], allmn.astype(np.int32))
    pos = {x: i for i, x in enumerate(allmn)}
    eng = fl.Engine(d, fl.K, f'legwatch{seed}', steps, F=1, q0=[q0])
    cum, tr, counts = [np.zeros(len(allmn))], [], []
    for i in range(steps):
        cum.append(b.advance(fl.K).astype(float))
        counts.append(cum[-1] - cum[-2])
        w = min(i + 1, WIN)
        rate = (cum[-1] - cum[-1 - w]) / (w * 1e-3)
        x = held.copy()
        for leg in LEGS6:
            jn, sg = assign(ty[cells[leg]])
            off = targets(rate[[pos[c] for c in cells[leg]]], jn, sg) * flip[leg]
            x[acts[leg]] = held[acts[leg]] + off
        eng.send(x)
        q = eng.recv()
        tr.append(np.r_[[q[qadr[leg]] for leg in LEGS6]].ravel())
    b.close()
    eng.close()
    counts, tr = np.array(counts), np.degrees(np.array(tr)).reshape(steps, 6, 7)
    hz = counts.sum(0) / (steps * 1e-3)
    df = pd.DataFrame({'leg': sub[allmn] + sd[allmn], 'type': ty[allmn], 'hz': hz})
    fire = df[df.hz > 0.5].groupby(['leg', 'type']).hz.agg(['size', 'mean']).round(1)
    print('MNs firing > 0.5 Hz:', int((hz > 0.5).sum()), 'of', len(allmn))
    print(fire.to_string())
    f = np.fft.rfftfreq(steps, 1e-3)
    for li, leg in enumerate(LEGS6):
        y = tr[:, li] - tr[:, li].mean(0)
        P = np.abs(np.fft.rfft(y, axis=0)) ** 2
        pk = f[1:][P[1:].sum(1).argmax()]
        print(leg, 'range deg', np.ptp(tr[:, li], 0).round(1).tolist(), f'peak {pk:.0f} Hz', 'power 5-15 Hz', round(float(P[(f >= 5) & (f <= 15)].sum() / P[1:].sum()), 2))
    ti = tr[:, :, 5] - tr[:, :, 5].mean(0)
    print('tibia correlation with lf:', {leg: round(float(np.corrcoef(ti[:, 0], ti[:, k])[0, 1]), 2) for k, leg in enumerate(LEGS6)})
    xy = q[:2]
    print('body moved (mm):', np.round(xy - q0[:2], 2).tolist())
    np.save(fl.OUT / f'leg-watch-{seed}.npy', {'counts': counts.astype(np.uint8), 'angles': tr.astype(np.float32), 'cells': allmn}, allow_pickle=True)


if __name__ == '__main__' and sys.argv[1:2] == ['watch']:
    watch(*sys.argv[2:])
