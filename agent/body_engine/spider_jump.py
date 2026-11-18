"""Jumping for the model spider (private note SPIDER_BRAIN unit 3): a ballistic jump of the spider body in the engine, checked
against MuJoCo on the same rollout and against published jump numbers.

Mechanism (a MODEL, not anatomy): the jump variant of models/spider.py (`build(jump=True)`) gives the first four servos of every leg
JUMP_S x the gain and force limit, a stand-in for the hydraulic leg extension real salticids jump with. The ctrl script holds the stance
for T_PRE control steps, then sets every leg's coxa yaw, coxa pitch, femur and patella to one fixed extension pose, weighted per leg pair
(PAIR_GAIN: legs III push hardest, I, II, IV about half). The constants come from a random search on MuJoCo alone (CPU; objective: take-off
speed 750 mm/s, forward heading, speed 640-950 and angle -10.5..53.5 deg as hard bounds, tilt and spin small); they were not tuned on the engine.

Published numbers (Nabawy et al. 2018 Sci Rep, PMC5940701, Phidippus regius 150 mg, L 15 mm: take-off velocity 0.52 to 0.97 m/s,
take-off angle -10.5 to 53.5 deg, acceleration ~5 g; Parry & Brown via the same paper: Sitticus pubescens ~5 mm, 10 body-length gap
50 mm at 0.67 m/s). The model spider is 23.5 mg, L 5.6 mm. KNOWN MISMATCH, reported not hidden: the model's burst takes ~1.5 ms (about 50 g),
the animals ~15 ms (5 g): the servo has no compliance or stroke limit.

  python -m agent.body_engine.spider_jump check     (under gpu-wait)

gate: (1) engine take-off speed and angle within 3 % and 2 deg of MuJoCo on the same ctrl stream; (2) both inside the published
ranges; (3) after take-off the centre of mass follows a parabola (residual < 0.25 mm over ~80 ms) in both; (4) the engine's landing distance is
within 3 % of the ballistic prediction from its own take-off state (to its touch-down height) and is under 10 body lengths; (5) trajectory difference engine vs
MuJoCo at 20 ms and 60 ms after take-off is reported, bar 1 mm (contact release is sensitive).
"""
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

from agent.body_engine.live import prepare
from kinestride.model import ENGINE, load
from kinestride.models.spider import build as spider_build
from agent.body_engine.spider_link import build, sizes
from agent.body_engine.spider_gait import NU

OUT = Path('.work/agent/body_engine/spider-jump')
G = 9810.0  # mm/s^2
BODY_MM = 5.6
T_PRE = 101  # control steps (0.1 ms each) at the stance before the burst
PAIR_GAIN = (0.574, 0.499, 1.0, 0.448)  # leg pair I..IV
AF, AP, AY, AC = 0.305, -0.909, -1.134, 0.570  # femur, patella, coxa yaw (x side), coxa pitch, rad
BURST_SCALE = 1.0
STEPS = 1400
AIR = 30  # control steps (3 ms) without any contact make a take-off
SPEED_RANGE, ANGLE_RANGE = (520.0, 970.0), (-10.5, 53.5)  # mm/s, deg (Nabawy 2018)


def ctrl(i):
    """Servo targets at control step i (K = 1): the stance, then the burst held."""
    u = np.zeros(NU, np.float32)
    if i >= T_PRE:
        for leg in range(8):
            g = BURST_SCALE * PAIR_GAIN[leg % 4]
            b = 2 + 7 * leg
            u[b] = g * AY * (1 if leg < 4 else -1)
            u[b + 1] = g * AC
            u[b + 2] = g * AF
            u[b + 3] = g * AP
    return u


def files():
    d = OUT / 'f1'
    if not (d / 'drive.bin').exists():
        prepare(d, mjcf=str(spider_build(jump=True)), dist=15.0)
        np.zeros(2, np.float32).tofile(d / 'drive.bin')
    return d


def com_series(m, q, v):
    """Centre of mass position and velocity per step from engine states (q: (steps, nq), v: (steps, nv)), by MuJoCo kinematics."""
    d = mujoco.MjData(m)
    pos, vel = [], []
    for qi, vi in zip(q, v):
        d.qpos[:], d.qvel[:] = qi, vi
        mujoco.mj_forward(m, d)
        mujoco.mj_subtreeVel(m, d)
        pos.append(d.subtree_com[1].copy())
        vel.append(d.subtree_linvel[1].copy())
    return np.array(pos), np.array(vel)


def engine_rollout(d, steps, frames=None, cam=None, w=64, h=64, every=0):
    """steps control steps (K = 1) with the host brain: (qpos, qvel, valid contact slots) per step; frames: file for RGBA frames."""
    nq, nv, ns = sizes(d)
    b = build(d, 1, 1, 1, w, h)
    cam = cam or ['0', '0', '10', '0', '0', '0', '45']
    argv = [str(b), str(d) + '/', str(steps), *cam, str(every), '1']
    if frames:  # fd 4 = the frames file: a shell redirect (subprocess's close_fds would drop an fd dup'ed in preexec_fn)
        argv = ['/usr/bin/bash', '-c', 'exec "$@" 4>' + str(Path(frames).resolve()), 'engine', *argv]
    p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(OUT / 'engine.err', 'wb'))
    q, v, valid = [], [], []
    for i in range(steps):
        try:
            p.stdin.write(ctrl(i).tobytes())
            p.stdin.flush()
        except BrokenPipeError:
            p.wait()
            raise SystemExit(f'FAIL jump: engine exited {p.returncode} at step {i}: {(OUT / "engine.err").read_text()[-300:]}')
        buf = p.stdout.read(4 * (NU + nq + nv + ns))
        a = np.frombuffer(buf, np.float32)
        q.append(a[NU:NU + nq].astype(float))
        v.append(a[NU + nq:NU + nq + nv].astype(float))
        valid.append(a[NU + nq + nv:])
    if p.wait() != 0:
        raise SystemExit(f'FAIL jump: engine exited {p.returncode}')
    return np.array(q), np.array(v), np.array(valid)


def mujoco_rollout(m, q0, v0, steps):
    d = mujoco.MjData(m)
    d.qpos[:], d.qvel[:] = q0, v0
    q, v, nc = [], [], []
    for i in range(steps):
        d.ctrl[:] = ctrl(i)
        mujoco.mj_step(m, d)
        q.append(d.qpos.copy())
        v.append(d.qvel.copy())
        nc.append(sum(c.efc_address >= 0 for c in d.contact[:d.ncon]))
    return np.array(q), np.array(v), np.array(nc)


def takeoff(com, vel, contacts):
    """First step at or after the burst from which no contact is active for AIR steps (the feet flick off the ground for a step or two
    inside the burst and touch again): (step, speed, angle deg, velocity)."""
    k = next((i for i in range(T_PRE, len(contacts) - AIR) if not np.any(contacts[i:i + AIR])), None)
    if k is None:
        return None
    s = vel[k]
    return k, float(np.linalg.norm(s)), float(np.degrees(np.arctan2(s[2], np.hypot(s[0], s[1])))), s


def flight(com, vel, contacts, k):
    """Landing step (first contact after take-off), and the parabola residual of the centre of mass over the flight."""
    land = next((i for i in range(k + 20, len(contacts)) if contacts[i] > 0), len(contacts))
    t = (np.arange(k, land) - k) * 1e-4
    pred = com[k] + vel[k] * t[:, None] + 0.5 * np.array([0, 0, -G]) * t[:, None] ** 2
    return land, float(np.abs(com[k:land] - pred).max())


def check():
    d = files()
    m = load(str(spider_build(jump=True)))
    m.opt.noslip_iterations = 0
    nq, nv, ns = sizes(d)
    q0 = np.fromfile(d / 'qpos.bin', np.float32).astype(float)
    v0 = np.fromfile(d / 'qvel.bin', np.float32).astype(float)
    eq, ev, valid = engine_rollout(d, STEPS)
    mq, mv, mc = mujoco_rollout(m, q0, v0, STEPS)
    ecom, evel = com_series(m, eq, ev)
    mcom, mvel = com_series(m, mq, mv)
    et, mt = takeoff(ecom, evel, (valid > 0).sum(1)), takeoff(mcom, mvel, mc)
    fail = []
    if et is None or mt is None:
        print(f'FAIL no take-off: engine {et is not None}, MuJoCo {mt is not None}')
        raise SystemExit('FAIL spider_jump: no take-off')
    ds, da = abs(et[1] - mt[1]) / mt[1], abs(et[2] - mt[2])
    ok = ds <= 0.03 and da <= 2.0
    print(f'{"PASS" if ok else "FAIL"} (1) take-off speed engine {et[1]:.1f} vs MuJoCo {mt[1]:.1f} mm/s ({100 * ds:.2f} %, bar 3), angle {et[2]:.1f} vs {mt[2]:.1f} deg '
          f'(diff {da:.2f}, bar 2), engine take-off {1e3 * (et[0] - T_PRE) * 1e-4:.1f} ms after the burst, MuJoCo {1e3 * (mt[0] - T_PRE) * 1e-4:.1f}')
    fail += [] if ok else ['take-off']
    ok = all(SPEED_RANGE[0] <= x[1] <= SPEED_RANGE[1] and ANGLE_RANGE[0] <= x[2] <= ANGLE_RANGE[1] for x in (et, mt))
    print(f'{"PASS" if ok else "FAIL"} (2) inside Nabawy 2018 ranges: speed {SPEED_RANGE} mm/s, angle {ANGLE_RANGE} deg; acceleration to take-off '
          f'{et[1] / max((et[0] - T_PRE) * 1e-4, 1e-4) / G:.0f} g (animals ~5 g: known mismatch)')
    fail += [] if ok else ['published ranges']
    (el, er), (ml, mr) = flight(ecom, evel, (valid > 0).sum(1), et[0]), flight(mcom, mvel, mc, mt[0])
    ok = er < 0.25 and mr < 0.25
    print(f'{"PASS" if ok else "FAIL"} (3) centre of mass parabola residual engine {er:.4f} mm, MuJoCo {mr:.4f} mm over {1e3 * (el - et[0]) * 1e-4:.0f} / {1e3 * (ml - mt[0]) * 1e-4:.0f} ms of flight')
    fail += [] if ok else ['ballistic']
    dz = float(ecom[et[0], 2] - ecom[el, 2]) if el < STEPS else np.nan  # height lost between take-off and touch-down
    t_f = (et[3][2] + np.sqrt(et[3][2] ** 2 + 2 * G * dz)) / G
    pred = float(np.hypot(et[3][0], et[3][1]) * t_f)
    dist = float(np.linalg.norm((ecom[el] - ecom[et[0]])[:2])) if el < STEPS else np.nan
    ok = el < STEPS and abs(dist - pred) <= 0.03 * pred and dist < 10 * BODY_MM
    print(f'{"PASS" if ok else "FAIL"} (4) engine jump distance {dist:.1f} mm = {dist / BODY_MM:.1f} body lengths (ballistic from its own take-off state to its touch-down height {pred:.1f} mm, bar 3 %; under 10 L = {10 * BODY_MM:.0f} mm)')
    fail += [] if ok else ['distance']
    for ms in (20, 60):
        n = ms * 10
        ke, km = et[0] + n, mt[0] + n
        if max(ke, km) < min(STEPS, el, ml):
            diff = float(np.linalg.norm(ecom[ke] - mcom[km]))
            print(f'{"PASS" if diff < 1.0 else "FAIL"} (5) CoM engine vs MuJoCo {ms} ms after take-off: {diff:.3f} mm (bar 1)')
            fail += [] if diff < 1.0 else [f'trajectory {ms} ms']
    if fail:
        raise SystemExit(f'FAIL spider_jump: {fail}')
    print('PASS all spider_jump checks')


def strip():
    """Eight frames of the engine's jump (side view) as one picture, .work/spider-jump.png."""
    from PIL import Image
    d = files()
    W, H = 480, 270
    q0 = np.fromfile(d / 'qpos.bin', np.float32)
    look = q0[:3] + np.array([4.0, 0.0, 8.0])
    cam = [str(x) for x in (*(look + np.array([0.0, -30.0, 4.0])), *look, 45)]
    engine_rollout(d, 1000, frames=str(OUT / 'frames.rgba'), cam=cam, w=W, h=H, every=125)
    f = np.fromfile(OUT / 'frames.rgba', np.uint8).reshape(-1, H, W, 4)[..., :3]
    Image.fromarray(np.concatenate([np.concatenate(list(f[r * 4:r * 4 + 4]), 1) for r in range(2)], 0)).save('.work/spider-jump.png')
    print('wrote .work/spider-jump.png', f.shape[0], 'frames')


if __name__ == '__main__':
    {'check': check, 'strip': strip}[sys.argv[1]]()
