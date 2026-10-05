"""Leg touch and proprioception for the male CNS brain: the body's state each control step -> Poisson rates on the leg
sensory neurons of the leg they belong to (the leg counterpart of agent/fly_vision.py).

The male CNS v1.0 gives leg sensory neurons no soma neuromere or side (their somata sit in the leg), so each takes the leg
(T1-T3, L/R) its synapses mostly go to, among cells with a known neuromere and side (one hop, as agent/eye_map.py does for
photoreceptors): 2,330 of 2,601 placed, median purity 0.63, left/right symmetric. Groups by subclass:
  touch  leg bristle + leg (tarsal)     100 Hz while the leg's tibia/tarsus touches anything outside the fly
  load   campaniform sensilla           150 Hz x the leg's normal contact force / (body weight / 3), clipped at 1
  angle  chordotonal organ + hair plate 150 Hz x the leg joints' mean |q - q_stance| / 0.5 rad, clipped at 1
The state is read with MuJoCo on the CPU from the engine's qpos (same model), so no contact table leaves the engine.
"""
import mujoco
import numpy as np
import pandas as pd

LEGS = {'lf': 'T1L', 'lm': 'T2L', 'lh': 'T3L', 'rf': 'T1R', 'rm': 'T2R', 'rh': 'T3R'}
GROUP = {'leg bristle': 'touch', 'leg': 'touch', 'campaniform sensilla': 'load', 'chordotonal organ': 'angle', 'hair plate': 'angle'}
TOUCH_HZ, RMAX, ANGLE_RAD = 100.0, 150.0, 0.5


def cells(shuffle=None):
    """{(leg, group): MCNS indices}; shuffle = a Generator: leg labels permuted across cells (mutant, same counts)."""
    from agent import mcns
    from agent.mcns_mapping import annotation_table
    a = annotation_table()
    pre, post, w, _, _ = mcns.graph()
    neu, side = a.somaNeuromere.fillna('').to_numpy(), a.somaSide.fillna('').str[:1].str.upper().to_numpy()
    leg = np.where(np.isin(neu, ['T1', 'T2', 'T3']) & np.isin(side, ['L', 'R']), pd.Series(neu).str.cat(side).to_numpy(), '')
    sub = a.subclass.fillna('').to_numpy()
    e = np.isin(sub, list(GROUP))[pre] & (leg[post] != '')
    v = pd.DataFrame({'s': pre[e], 'leg': leg[post[e]], 'w': np.abs(w[e])}).groupby(['s', 'leg']).w.sum().reset_index()
    best = v.sort_values('w').drop_duplicates('s', keep='last')
    s, lg = best.s.to_numpy(), best.leg.to_numpy()
    if shuffle is not None:
        lg = shuffle.permutation(lg)
    g = np.array([GROUP[x] for x in sub[s]])
    return {(k, grp): s[(lg == t) & (g == grp)] for k, t in LEGS.items() for grp in ('touch', 'load', 'angle')}


class Sense:
    """__call__(qpos) -> {(leg, group): Hz} from the body state."""
    def __init__(self, m, q_stance):
        self.m, self.d = m, mujoco.MjData(m)
        name = lambda o, i: mujoco.mj_id2name(m, o, i) or ''
        self.geom_leg = np.array([next((k for k in LEGS if name(mujoco.mjtObj.mjOBJ_GEOM, i).startswith(f'fly/{k}_t')), '') for i in range(m.ngeom)])
        self.fly = np.array([name(mujoco.mjtObj.mjOBJ_GEOM, i).startswith('fly/') for i in range(m.ngeom)])
        self.jq = {k: np.array([m.jnt_qposadr[j] for j in range(m.njnt) if f'-{k}_' in name(mujoco.mjtObj.mjOBJ_JOINT, j)]) for k in LEGS}
        self.q0 = np.asarray(q_stance, float)
        self.share = m.body_subtreemass[1] * abs(m.opt.gravity[2]) / 3
        self.f = np.zeros(6)

    def __call__(self, qpos):
        d = self.d
        d.qpos[:] = qpos
        d.qvel[:] = 0
        mujoco.mj_forward(self.m, d)
        touch, load = dict.fromkeys(LEGS, 0.0), dict.fromkeys(LEGS, 0.0)
        for c in range(d.ncon):
            g1, g2 = d.contact[c].geom1, d.contact[c].geom2
            k = self.geom_leg[g1] if not self.fly[g2] else self.geom_leg[g2] if not self.fly[g1] else ''
            if k:
                mujoco.mj_contactForce(self.m, d, c, self.f)
                touch[k], load[k] = TOUCH_HZ, load[k] + abs(self.f[0])
        out = {}
        for k in LEGS:
            ang = np.abs(qpos[self.jq[k]] - self.q0[self.jq[k]]).mean() / ANGLE_RAD
            out[k, 'touch'], out[k, 'load'], out[k, 'angle'] = touch[k], RMAX * min(load[k] / self.share, 1), RMAX * min(ang, 1)
        return out
