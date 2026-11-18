"""Gate and export for fly_gait.mojo (the FlyGym gait on the card, live.mojo -DBRAIN=3; private note BRAIN_BODY_LOOP step 3).

  python -m agent.body_engine.fly_gait build
  gpu-wait run --shared --vram 1 --priority 60 --timeout 900 --env PYTHONPATH=$PWD -- .venv/bin/python -m agent.body_engine.fly_gait check

The kernel (float64, one thread) and flygym_demo's HybridTurningController (zero observations, the flyloop mapping fly_link.Gait) run the same
drive sequence of 700 control steps (steady 1/1, left weak, right reversed for 300 steps so the slewed drive goes negative, both off, both 0.7) at timestep K x 0.1 ms; the 42 leg servo targets
and 6 adhesion values written into the ctrl buffer must equal the controller's, max abs error <= 1e-5 (float32 ctrl: ~1e-7). The head servos and
every other entry keep their start value. check.sh flygait adds the mutants: the phase bias sign, the adhesion window, the drive sign.
"""
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np

from agent.body_engine import fly_link
from kinestride.model import ENGINE
from kinestride.render_check import MOJO
APP = Path(__file__).parent  # this app's Mojo sources (live, spectate, gait check); the physics core is kinestride's ENGINE

K = fly_link.K
DOFS = (('coxa', 'pitch'), ('coxa', 'roll'), ('coxa', 'yaw'), ('trochanterfemur', 'pitch'), ('trochanterfemur', 'roll'), ('tibia', 'pitch'), ('tarsus1', 'pitch'))  # preprogrammed.py


def export():
    """gait.tab (float64: per leg the periodic cubic spline's 4 x 44 x 7 polynomial coefficients, 7 neutral angles, swing end), gait.idx (int32: the ctrl index of
    every (leg, spline dof), then of the six adhesion controls), gait.state (float64: 6 start phases, 6 start magnitudes, 2 slewed drives at 0), next to the fly's engine files."""
    import mujoco
    from tools.fly_gait_oracle import LEGS, joint_name
    d, m, _ = fly_link.prepare()
    g = fly_link.Gait(m, d, K)
    tab, idx = [], []
    act = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
    for leg in LEGS:
        cs = g.steps._psi_funcs[leg]
        tab.append(np.concatenate([cs.c.ravel(), g.steps.neutral_pos[leg][:, 0], [g.steps.swing_period[leg][1]]]))
        idx += [act(joint_name(leg, seg, axis) + '-position') for seg, axis in DOFS]
    idx += g.adh
    nseg = g.steps._psi_funcs['lf'].c.shape[1]
    np.concatenate(tab).astype(np.float64).tofile(d / 'gait.tab')
    np.array(idx, np.int32).tofile(d / 'gait.idx')
    np.r_[[0, np.pi, 0, np.pi, 0, np.pi], np.zeros(8)].astype(np.float64).tofile(d / 'gait.state')
    return d, m, g, nseg


def build(d, nseg):
    sizes = (d / 'sizes.txt').read_text().split()
    nb = int([f for f in sizes if f.startswith('-DNBODY=')][0][8:])
    deps = [ENGINE / f for f in ('fly_gait_check.mojo', 'fly_gait.mojo', 'smooth.mojo')]
    h = hashlib.sha1(b''.join(f.read_bytes() for f in deps) + str(nseg).encode()).hexdigest()[:8]
    b = d / f'gait_check_{h}'  # content-addressed: mutant-check restores old mtimes
    if not b.is_file():
        subprocess.run([MOJO, 'build', '-I', str(ENGINE), *sizes, '-DFLIES=1', '-DSTAGE=7', f'-DK={K}', '-DWAVE=1', f'-DGSEG={nseg}', f'-DPROWS={7 * (nb - 1)}',
                        str(APP / 'fly_gait_check.mojo'), '-o', str(b)], check=True, capture_output=True)
    return b


def drives(n=700):
    """The slew moves a drive by at most dt / 0.2 s per step, so the reversed segment is 300 steps: the right drive reaches -0.5 (from +1) at step
    500; a shorter one never goes negative and the drive-sign mutant passes unseen (2026-10-07)."""
    seq = np.zeros((n, 2))
    seq[:100] = (1.0, 1.0)
    seq[100:200] = (0.3, 1.0)
    seq[200:500] = (1.0, -1.0)
    seq[500:550] = (0.0, 0.0)
    seq[550:] = (0.7, 0.7)
    return seq


def check():
    d, m, g, nseg = export()
    b = build(d, nseg)
    seq = drives()
    seq.astype(np.float64).tofile(d / 'drive_seq.bin')
    subprocess.run([str(b), str(d) + '/', str(len(seq)), repr(K * 1e-4)], check=True, capture_output=True)
    got = np.fromfile(d / 'gait_trace.bin', np.float32).reshape(len(seq), -1)
    want = np.array([g.ctrl(s) for s in seq])
    hit = np.r_[g.act, g.adh]
    keep = np.setdiff1d(np.arange(got.shape[1]), hit)
    err = float(np.abs(got[:, hit] - want[:, hit]).max())
    held = float(np.abs(got[:, keep] - got[0, keep]).max()) if len(keep) else 0.0
    ok = err <= 1e-5 and held == 0.0
    print(('PASS' if ok else 'FAIL'), f'fly gait kernel vs HybridTurningController, {len(seq)} control steps (dt {K * 1e-4:g} s, drives steady/weak/reversed/off): max abs err {err:.2e} '
          f'(bar 1e-5) over {len(g.act)} servos + {len(g.adh)} adhesion; untouched entries drift {held:.1e}')
    if not ok:
        sys.exit('FAIL fly_gait')
    print('PASS fly_gait')


if __name__ == '__main__':
    if (sys.argv[1] if len(sys.argv) > 1 else 'check') == 'build':
        d, m, g, nseg = export()
        print('built', build(d, nseg))
    else:
        check()
