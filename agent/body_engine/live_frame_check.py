"""live.world_camera and live.lift (one batched world.mojo run each) against a per-azimuth, per-animal reference on
world.py, the terrain oracle; placements are sampled until the landmark cones occlude the first azimuth 12 times.

  .venv/bin/python -m agent.body_engine.live_frame_check     (CPU, ~3 s)
"""
import sys
from pathlib import Path

import mujoco
import numpy as np
from kinestride.world import height_grad

from agent.body_engine import fly_link as fl, live


def camera(q, m, p, fovy):
    """live.world_camera's search, one azimuth and one root at a time: (eye, the first azimuth taken)."""
    look = np.mean([q[a:a + 3] for a in live.roots(m)], 0)
    dist = (max(np.linalg.norm(q[a:a + 3] - look) for a in live.roots(m)) + 5.0) / np.sin(np.radians(float(fovy)) / 2)
    s = np.linspace(0, 1, 200)[:190, None]
    for k, az in enumerate(np.radians(np.arange(-130, 230, 30))):
        eye = look + dist * np.array([0.85 * np.cos(az), 0.85 * np.sin(az), 0.52])
        if all((l := eye + s * (q[a:a + 3] - eye))[:, 2].min() > 0 and (l[:, 2] > height_grad(l[:, 0], l[:, 1], p)[0] + 0.2).all() for a in live.roots(m)):
            break
    return eye, k


def lift(m, q, p):
    """live.lift, one animal at a time."""
    d = mujoco.MjData(m)
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    for a, j in zip(live.roots(m), np.flatnonzero(m.jnt_type == mujoco.mjtJoint.mjJNT_FREE)):
        b = np.flatnonzero(m.body_rootid == m.body_rootid[m.jnt_bodyid[j]])
        q[a + 2] += float(height_grad(d.xpos[b, 0], d.xpos[b, 1], p)[0].max()) + 0.05


def main():
    m = live.load(live.terrain(fl.MJCF, live.WORLD[:5] + (1,) + live.WORLD[6:])[0])
    t = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_NUMERIC, 'terrain')
    p = m.numeric_data[m.numeric_adr[t]:m.numeric_adr[t] + 16]
    out = Path('.work/live-frame-check')
    out.mkdir(parents=True, exist_ok=True)
    rng, a, hid, clear, bad = np.random.default_rng(5), live.roots(m)[0], 0, 0, []
    for _ in range(1000):
        q = m.qpos0.astype(np.float32)
        q[a:a + 2] = rng.uniform(-400, 400, 2)
        eye, k = camera(q.astype(float), m, p, 45)
        if k == 0 and clear >= 4:
            continue
        hid, clear = hid + (k > 0), clear + (k == 0)
        q.tofile(out / 'qpos.bin')
        if np.abs(np.array(live.world_camera(out, m, 45)[:3], float) - eye).max() > 1e-3:
            bad.append(f'camera@{q[a]:.0f},{q[a + 1]:.0f}')
        q0, q1 = q.astype(float), q.astype(float)
        lift(m, q0, p)
        live.lift(m, q1)
        if abs(q0[a + 2] - q1[a + 2]) > 1e-5:
            bad.append(f'lift@{q[a]:.0f},{q[a + 1]:.0f}')
        if hid >= 12:
            break
    ok = hid >= 12 and not bad
    print('PASS' if ok else 'FAIL', f'live frame check: {hid} occluded + {clear} clear placements, {len(bad)} off', *bad)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
