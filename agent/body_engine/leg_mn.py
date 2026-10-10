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


if __name__ == '__main__':
    main(*sys.argv[1:])
