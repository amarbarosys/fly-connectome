"""Every sense the world supplies, as Poisson rates on male CNS sensory neurons (docs/plans/FULL_FLY.md U2).

One registry: each sense is a list of (cell ids, Hz) built once per model and read from the engine's qpos on the CPU.
  vision   agent/eye_render.Eyes: rendered eye cams -> 6,291 R1-6 / L1-L3 cells (every EYE_EVERY steps, by the caller)
  legs     agent/fly_legsense: tarsal touch, load, joint angle -> 2,330 leg sensory cells
  smell    odor at the l/r_odor sites -> ORNs of the stand-in odor's glomeruli on that antenna (VINEGAR), RMAX x concentration
  taste    a leg's taste site over a food patch -> that leg's gustatory leg-bristle neurons (MCNS labels no sweet receptor)
  body     a non-leg fly geom touching the world -> mechanosensory bristles entering by ADMN / PDMN (notum), by side
  gravity  thorax roll -> Johnston's organ wind_gravity neurons of the downhill side, RMAX x |roll| / 30 deg
Sensory neurons have no soma side in MCNS (somata sit outside the CNS), so each takes the side its synapses mostly go to
(one hop, as fly_legsense does for legs). Odor and food are engineered stand-ins at fly scale (the world plume is 100 mm
wide, so two antennae 0.5 mm apart read the same value); hygro- and thermosensory cells get nothing: the world has no
humidity or temperature field.
  python -m agent.senses check     CPU: every sense has cells, rates are finite, and each side mutant flips on a static pose
"""
import sys

import mujoco
import numpy as np
import pandas as pd

from agent import fly_legsense as ls

VINEGAR = ('DM1', 'DM4', 'VM2', 'VA2', 'DP1m')  # stand-in odor: glomeruli reported for vinegar (Semmelhack & Wang 2009), not a fitted odor model
RMAX, ODOR_SIGMA = 150.0, 6.0  # Hz; mm, the odor blob's width
LEG_NERVE = {'f': 'ProLN', 'm': 'MesoLN', 'h': 'MetaLN'}


def sided(ids, pre, post, w, side):
    """'L' / 'R' per cell: the soma side its outgoing synapses mostly reach."""
    ids = np.asarray(ids, np.int64)
    pos = np.full(len(side), -1, np.int64)
    pos[ids] = np.arange(len(ids))
    e = (pos[pre] >= 0) & np.isin(side[post], ['L', 'R'])
    acc = np.zeros((len(ids), 2))
    np.add.at(acc, (pos[pre[e]], (side[post[e]] == 'R').astype(int)), np.abs(w[e]))
    return np.where(acc[:, 1] > acc[:, 0], 'R', 'L')


def cells(swap=False):
    """{(sense, key): MCNS ids}; swap: every left/right label exchanged (the side mutant)."""
    from agent import mcns
    a = pd.read_feather(mcns.DATA / 'body-annotations-male-cns-v1.0-minconf-0.5.feather',
                        columns=['status', 'class', 'subclass', 'type', 'entryNerve', 'somaSide'])
    a = a[a['status'].isin(['Traced', 'Anchor'])].reset_index(drop=True)  # graph() order
    pre, post, w, _, _ = mcns.graph()
    side = a.somaSide.fillna('').str[:1].str.upper().to_numpy()
    cl, sub, ty, nerve = (a[c].fillna('').to_numpy() for c in ('class', 'subclass', 'type', 'entryNerve'))
    out = {('legs', k): v for k, v in ls.cells().items()}
    groups = {'smell': (cl == 'olfactory') & np.isin(pd.Series(ty).str.replace('ORN_', '').to_numpy(), VINEGAR),
              'body': (cl == 'mechanosensory_tactile') & np.isin(nerve, ['ADMN', 'PDMN']),
              'gravity': (cl == 'mechanosensory') & (sub == 'wind_gravity')}
    groups |= {f'taste_{g}': (cl == 'gustatory') & (sub == 'leg bristle') & (nerve == n) for g, n in LEG_NERVE.items()}
    for g, m in groups.items():
        ids = np.flatnonzero(m)
        s = sided(ids, pre, post, w, side)
        for sd in 'LR':
            out[g, sd if not swap else {'L': 'R', 'R': 'L'}[sd]] = ids[s == sd]
    return out


class Senses:
    """rates(q) -> {(sense, key): Hz} for the non-visual senses; world: odor (x, y) mm, food patch (x, y, r) mm."""
    def __init__(self, m, q0, odor=None, food=None):
        self.m, self.d, self.odor, self.food = m, mujoco.MjData(m), odor, food
        self.legs = ls.Sense(m, q0)
        site = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f'fly/{n}')
        self.nose = [site('l_odor'), site('r_odor')]
        self.taste = {k: site(f'{k}_taste') for k in ls.LEGS}
        self.th = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'fly/c_thorax')
        nm = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or '' for i in range(m.ngeom)]
        self.fly = np.array([n.startswith('fly/') for n in nm])
        self.leg = np.array([any(n.startswith(f'fly/{k}_') for k in ls.LEGS) for n in nm])

    def rates(self, q):
        out = {('legs', k): v for k, v in self.legs(q).items()}  # runs mj_forward on its own copy
        d = self.d
        d.qpos[:] = q
        mujoco.mj_forward(self.m, d)
        for sd, s in zip('LR', self.nose):
            c = 0.0 if self.odor is None else float(np.exp(-np.sum((d.site_xpos[s][:2] - self.odor) ** 2) / (2 * ODOR_SIGMA ** 2)))
            out['smell', sd] = RMAX * c
        for k, s in self.taste.items():
            on = self.food is not None and np.hypot(*(d.site_xpos[s][:2] - self.food[:2])) < self.food[2]
            out[f'taste_{k[1]}', k[0].upper()] = max(out.get((f'taste_{k[1]}', k[0].upper()), 0.0), RMAX * on)
        hit = {'L': 0.0, 'R': 0.0}
        for c in range(d.ncon):
            g1, g2 = d.contact[c].geom1, d.contact[c].geom2
            g = g1 if self.fly[g1] and not self.fly[g2] else g2 if self.fly[g2] and not self.fly[g1] else -1
            if g >= 0 and not self.leg[g]:
                hit['L' if (d.geom_xpos[g] - d.xpos[self.th]) @ d.xmat[self.th].reshape(3, 3)[:, 1] > 0 else 'R'] = RMAX
        out['body', 'L'], out['body', 'R'] = hit['L'], hit['R']
        roll = float(np.degrees(np.arcsin(np.clip(d.xmat[self.th].reshape(3, 3)[2, 1], -1, 1))))  # + = left side down
        out['gravity', 'L'], out['gravity', 'R'] = RMAX * min(max(roll, 0) / 30, 1), RMAX * min(max(-roll, 0) / 30, 1)
        return out


def check():
    from agent.body_engine import fly_link as fl, live
    m = live.load(live.terrain(fl.MJCF, live.WORLD[:5] + (1,) + live.WORLD[6:])[0])
    d0, _, _ = fl.prepare(fl.OUT.parent / 'full-fly-world1', mjcf=live.terrain(fl.MJCF, live.WORLD[:5] + (1,) + live.WORLD[6:])[0], flat=fl.MJCF)
    q0 = np.fromfile(d0 / 'qpos.bin', np.float32).astype(float)
    c, cs = cells(), cells(swap=True)
    s = Senses(m, q0, odor=q0[:2] + [0, 3.0], food=np.r_[q0[:2], 8.0])  # odor 3 mm to the left (world +y at the stance heading is tested below), food under the fly
    r = s.rates(q0)
    q = q0.copy()
    mujoco.mju_mulQuat(q[3:7], np.array([np.cos(np.radians(10)), np.sin(np.radians(10)), 0, 0]), q0[3:7].copy())  # roll 20 deg about world x
    rr = s.rates(q)
    print({g: (len(c[g, 'L']), len(c[g, 'R'])) for g in ('smell', 'body', 'gravity', 'taste_f', 'taste_m', 'taste_h')})
    print('smell L/R Hz', round(r['smell', 'L'], 2), round(r['smell', 'R'], 2), '| taste', {k: v for k, v in r.items() if k[0].startswith('taste')},
          '| gravity at roll +20', rr['gravity', 'L'], rr['gravity', 'R'])
    ok = [('every sense has cells on both sides', all(len(c[g, sd]) for g in ('smell', 'body', 'gravity', 'taste_f', 'taste_m', 'taste_h') for sd in 'LR')),
          ('rates finite', all(np.isfinite(v) for v in list(r.values()) + list(rr.values()))),
          ('side mutant exchanges L and R cells', all(np.array_equal(c[g, 'L'], cs[g, 'R']) for g in ('smell', 'gravity'))),
          ('the odor side and the downhill side read more', r['smell', 'L'] != r['smell', 'R'] and rr['gravity', 'L'] != rr['gravity', 'R'])]
    for k, v in ok:
        print(('PASS ' if v else 'FAIL ') + k)
    if not all(v for _, v in ok):
        sys.exit('FAIL senses')
    print('PASS senses')


if __name__ == '__main__':
    check()
