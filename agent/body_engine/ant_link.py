"""Model ant body in the engine (private note ANT unit A2): a tripod gait drives agent/ant_body.py over spider_link's
host transport (live.mojo -DBRAIN=1: ctrl over stdin, sensor packets on stdout).

Gait: not an ant circuit, a stand-in so the brain units (A3-A5) have speed and turn drives that move the body. Per leg a
phase at FREQ Hz; coxa yaw sweeps +-YAW about the stance, coxa pitch lifts the foot on the forward sweep; tripods
l1 r2 l3 and r1 l2 r3 half a cycle apart. speed in [0, 1] scales the sweep; turn in [-1, 1] rotates every coxa yaw by
TURN x turn (+ = left). Shortening one side's sweep (the spider recipe) stalled this body in MuJoCo: +0.8 walked 1.0 mm, -3 deg.
ctrl layout = ant_body servo order: petiole_pitch, petiole_yaw, then l1 l2 l3 r1 r2 r3 x (coxa_yaw, coxa_pitch, femur, tibia, tarsus).

  python -m agent.body_engine.ant_link mujoco      CPU preflight of the same gate in MuJoCo
  gpu-wait run --shared --vram 2 --preemptible --timeout 900 --env PYTHONPATH=$PWD -- .venv/bin/python -m agent.body_engine.ant_link check
Gate (1 s): speed 1 walks > 1 body length (BODY_MM) with tilt < 20; speed 0 moves < 10 % of that; turn +0.8 / -0.8 turn
opposite ways, each beyond 30 deg; mutant (MUTANT=noctrl: zero ctrl written) must FAIL the walk.
"""
import os
import sys

import mujoco
import numpy as np

from agent.ant_body import build as ant_build
from agent.body_engine.spider_link import run, tilt_of, yaw_of

OUT = 'ant-link'
FREQ, YAW, LIFT, TURN = 8.0, 0.35, 0.2, 0.3  # MuJoCo sweep (freq 8/11, yaw .35/.5, lift .2/.3): 40.4 mm in 1 s, tilt 15.9; lift .3 tilts 24-39
NU = 2 + 6 * 5
TRIPOD = np.array([0, 1, 0, 1, 0, 1], np.float64)  # l1 l2 l3 r1 r2 r3: l1 r2 l3 in group 0
SIDE = np.array([1, 1, 1, -1, -1, -1], np.float64)
BODY_MM = 8.0  # head + mesosoma + petiole + gaster, ant_body.py
DRIVES = np.array([[1, 0], [0, 0], [1, 0.8], [1, -0.8]], np.float32)  # (speed, turn) per animal: walk, stand, left, right


def gait(t, speed, turn):
    """Servo targets (F, NU) at time t (s) for drives speed, turn ((F,) each)."""
    t, speed, turn = (np.asarray(x, np.float64) for x in (t, speed, turn))
    ph = (FREQ * t[..., None] + 0.5 * TRIPOD) % 1.0
    u = np.zeros(speed.shape + (NU,))
    leg = u[..., 2:].reshape(speed.shape + (6, 5))
    leg[..., 0] = SIDE * YAW * speed[..., None] * np.cos(2 * np.pi * ph) + TURN * turn[..., None]
    leg[..., 1] = -LIFT * speed[..., None] * np.maximum(0.0, np.sin(2 * np.pi * ph))
    return u.astype(np.float32)


def gate(q0, q):
    """q0, q: (NQ, F) start and end poses for DRIVES; prints the gate, returns ok."""
    disp = np.linalg.norm(q[:2] - q0[:2], axis=0)
    yaw = np.array([yaw_of(q[:, i]) - yaw_of(q0[:, i]) for i in range(q.shape[1])])
    tilt = max(tilt_of(q[:, i]) for i in range(q.shape[1]))
    ok = [(f"walk {disp[0]:.2f} mm > {BODY_MM}, tilt max {tilt:.1f} < 20", disp[0] > BODY_MM and tilt < 20),
          (f"stand {disp[1]:.2f} mm < 10 % of walk", disp[1] < 0.1 * disp[0]),
          (f"turn +0.8 {yaw[2]:+.0f} deg, -0.8 {yaw[3]:+.0f} deg: opposite, each beyond 30", yaw[2] > 30 and yaw[3] < -30)]
    for k, v in ok:
        print(('PASS ' if v else 'FAIL ') + k)
    return all(v for _, v in ok)


def mujoco_preflight(seconds=1.0):
    m = mujoco.MjModel.from_xml_path(str(ant_build()))
    F = len(DRIVES)
    q0, q = [], []
    for s, tn in DRIVES:
        d = mujoco.MjData(m)
        for _ in range(2000):  # settle 0.2 s at the stance
            mujoco.mj_step(m, d)
        q0.append(d.qpos.copy())
        for i in range(round(seconds / m.opt.timestep)):
            if i % 10 == 0:
                d.ctrl[:] = gait(i * m.opt.timestep, np.array([s]), np.array([tn]))[0]
            mujoco.mj_step(m, d)
        q.append(d.qpos.copy())
    return gate(np.array(q0).T, np.array(q).T)


def check():
    from agent.body_engine.spider_link import OUT as SL
    from agent.body_engine.live import prepare
    F, d = len(DRIVES), SL.parent / OUT / f'f{len(DRIVES)}'
    if not (d / 'drive.bin').exists():
        prepare(d, mjcf=str(ant_build()), dist=15.0)
        for n in ('qpos', 'qvel', 'ctrl'):
            np.tile(np.fromfile(d / f'{n}.bin', np.float32).reshape(-1, 1), (1, F)).astype(np.float32).tofile(d / f'{n}.bin')
        np.zeros(2 * F, np.float32).tofile(d / 'drive.bin')
    from agent.body_engine import spider_link
    spider_link.NU = NU  # run() reads the packet's ctrl rows with its module's servo count (spider 58)
    noctrl = os.environ.get('MUTANT') == 'noctrl'
    ctrl = lambda i: np.zeros((F, NU), np.float32) if noctrl else gait(i * 10 * 1e-4, DRIVES[:, 0], DRIVES[:, 1])
    pk, sec = run(d, F, 10, 1, 1000, ctrl=ctrl)
    q0 = np.fromfile(d / 'qpos.bin', np.float32).reshape(-1, F)
    ok = gate(q0, pk[-1][1])
    print(f'{sec * 1e3 / 1000:.2f} ms per control step (K=10, {F} ants)')
    if not ok:
        sys.exit('FAIL ant_link')
    print('PASS ant_link: the model ant walks, stands and turns in the engine')


if __name__ == '__main__':
    if sys.argv[1] == 'mujoco':
        sys.exit(0 if mujoco_preflight() else 'FAIL ant gait in MuJoCo')
    check()
