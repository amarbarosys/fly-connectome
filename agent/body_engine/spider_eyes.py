"""Spider eyes and prey for unit 2 (private note SPIDER_BRAIN): the eye table eyes.mojo renders from, the environment table
that moves the prey, an independent pinhole reference, and the gate (`python -m agent.body_engine.spider_eyes check`, under gpu-wait).

Eye geometry is a MODEL (jumping-spider layout from Winsor, Morehouse & Jakob 2023 via the existing spider_vision_test.py:
secondary eyes AL 0-70 deg and PL 60-170 deg each side of the heading, 20 deg rear blind zone; principal AM forward, narrow;
PM left out). Eyes sit just outside the prosoma ellipsoid (semi-axes 1.3 x 1.1 x 0.7 mm, models/spider.py), head frame:
+x forward, +y left, +z up. Each retina is a square pinhole image of NPIX x NPIX pixels.

gate (1): prey placed at known angles from each eye's INTENDED axis (INTENDED below, written out by hand, not read from the
exported table): the prey blob's centroid in that eye's image must sit at the analytic pinhole pixel within 0.75 px
(focal = NPIX/2/tan(fov/2), pixel = centre + f * (d.right, d.up) / (d.forward)); (2) each animal sees only its own prey (12
animals share one world position, 11 other prey exist); (3) the prey is moved by the environment only: a prey with t_on after
the run's end stays where it was.
"""
import sys
from pathlib import Path

import os
import threading

import numpy as np

from kinestride.models.spider import PREY_R

NPIX = 64
# name, local position mm, axis yaw deg from heading (+ = left), fov deg
EYES = [('AM_L', (1.25, 0.25, 0.35), 0.0, 14.0), ('AM_R', (1.25, -0.25, 0.35), 0.0, 14.0),
        ('AL_L', (1.10, 0.55, 0.45), 35.0, 70.0), ('AL_R', (1.10, -0.55, 0.45), -35.0, 70.0),
        ('PL_L', (0.20, 0.90, 0.50), 115.0, 110.0), ('PL_R', (0.20, -0.90, 0.50), -115.0, 110.0)]
# the same, written independently for the gate: what the table must mean
INTENDED = {'AM_L': ((1.25, 0.25, 0.35), 0.0, 14.0), 'AM_R': ((1.25, -0.25, 0.35), 0.0, 14.0),
            'AL_L': ((1.10, 0.55, 0.45), 35.0, 70.0), 'AL_R': ((1.10, -0.55, 0.45), -35.0, 70.0),
            'PL_L': ((0.20, 0.90, 0.50), 115.0, 110.0), 'PL_R': ((0.20, -0.90, 0.50), -115.0, 110.0)}
NE = len(EYES)


def eye_table(npix=NPIX):
    """(NE, 8) float32: local position 3, local axis 3, fov rad, focal length px. Axis in the head frame: (cos yaw, sin yaw, 0)."""
    t = np.zeros((NE, 8), np.float32)
    for i, (_, pos, yaw, fov) in enumerate(EYES):
        t[i, :3] = pos
        t[i, 3:6] = (np.cos(np.radians(yaw)), np.sin(np.radians(yaw)), 0)
        t[i, 6] = np.radians(fov)
        t[i, 7] = npix / 2 / np.tan(np.radians(fov) / 2)
    return t


def write_tables(d, env, npix=NPIX):
    """eyes.bin and env.bin in d. env: (F, 6) rows x0 y0 z0 vx vy t_on (mm, mm/s, s), written [field][animal]."""
    for name, a in (('eyes.bin', eye_table(npix)), ('env.bin', np.ascontiguousarray(np.asarray(env, np.float32).T))):
        tmp = Path(d) / f'{name}.{os.getpid()}.{threading.get_ident()}'
        a.tofile(tmp)
        os.replace(tmp, Path(d) / name)  # atomic: parallel arms rewrite the same table while another engine starts


def quat_rot(q, v):
    w, x, y, z = q
    r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    return r @ np.asarray(v, float)


def project(point, head_pos, head_quat, name):
    """Independent reference: pixel (u, v) of a world point in eye `name` of INTENDED, and its depth along the axis (mm)."""
    pos, yaw, fov = INTENDED[name]
    eye = np.asarray(head_pos, float) + quat_rot(head_quat, pos)
    fw = quat_rot(head_quat, (np.cos(np.radians(yaw)), np.sin(np.radians(yaw)), 0))
    right = np.cross(fw, (0, 0, 1))
    right /= np.linalg.norm(right)
    up = np.cross(right, fw)
    d = np.asarray(point, float) - eye
    z = d @ fw
    f = NPIX / 2 / np.tan(np.radians(fov) / 2)
    return NPIX / 2 + f * (d @ right) / z, NPIX / 2 - f * (d @ up) / z, z


def place(head_pos, head_quat, name, az, el, dist):
    """World point at azimuth az, elevation el (deg, from the INTENDED axis of eye `name`, + left / up in the head frame) and
    distance dist mm from the eye."""
    pos, yaw, _ = INTENDED[name]
    a, e = np.radians(yaw + az), np.radians(el)
    return np.asarray(head_pos, float) + quat_rot(head_quat, pos) + dist * quat_rot(head_quat, (np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)))


def blob(img, dark=15):
    """Centroid (u, v) and pixel count of the prey: the pixels darker than the spider's own parts."""
    ys, xs = np.nonzero(img < dark)
    return (xs.mean() + 0.5, ys.mean() + 0.5, len(xs)) if len(xs) else (np.nan, np.nan, 0)


def visibility(q, points):
    """MuJoCo ray casting from the intended eye to the prey sphere: fraction of 7 sample points (centre, 6 axis points of the
    sphere) with no own-body surface in the way. q: standing qpos of the spider (no prey). Independent of the renderer."""
    import mujoco
    from kinestride.model import load
    from kinestride.models.spider import build
    m = load(str(build()))
    d = mujoco.MjData(m)
    d.qpos[:] = q[:m.nq]
    mujoco.mj_kinematics(m, d)
    out = []
    for name, centre in points:
        pos, yaw, _ = INTENDED[name]
        eye = d.xpos[1] + quat_rot(d.xquat[1], pos)
        pts = [centre] + [centre + r * np.eye(3)[k] * sg for k in range(3) for sg in (1, -1) for r in (PREY_R,)]
        free = 0
        for p in pts:
            v = np.asarray(p, float) - eye
            dist = np.linalg.norm(v)
            gid = np.zeros(1, np.int32)
            hit = mujoco.mj_ray(m, d, eye, v / dist, None, 1, -1, gid)
            free += not (0 <= hit < dist - 1e-3)
        out.append(free / len(pts))
    return out


def scenarios(q, head, quat):
    """Per eye: two fully visible targets (the first found and the one farthest from it) and one fully hidden one, all found by ray casting the standing pose."""
    cand = [(az, el, dist) for dist in (5, 7, 12) for az, el in ((0, 0), (8, 4), (-8, -4), (8, -8), (-8, 8), (15, 8), (-15, -8), (15, -10), (-15, 10), (-25, 15), (25, -15), (3, 2), (-3, -2))]
    out = []
    for name, _, _, fov in EYES:
        pts = [(name, place(head, quat, name, az, el, dist)) for az, el, dist in cand]
        vis = visibility(q, pts)
        # angular radius of the prey must fit the field: skip targets whose sphere leaves the image
        ok = [(az, el, dist) for (az, el, dist), v in zip(cand, vis) if v == 1.0 and np.degrees(np.arctan(PREY_R / dist)) + max(abs(az), abs(el)) < fov / 2 - 1]
        hid = [(az, el, dist) for (az, el, dist), v in zip(cand, vis) if v == 0.0 and abs(az) + abs(el) < fov / 2 - 2]
        first = ok[0] if ok else None
        second = max((c for c in ok if c != first), key=lambda c: np.hypot(c[0] - first[0], c[1] - first[1]), default=None) if first else None  # farthest apart
        out += [(name, *c, 'clear') for c in (first, second) if c] + ([(name, *hid[0], 'hidden')] if hid else [])
    return out


def check():
    from kinestride.models.spider import PREY_R
    from agent.body_engine.spider_link import files, run
    d0 = files(1, 'eyes1', prey=True)  # the standing pose, one animal, to design the scenarios on
    q1 = np.fromfile(d0 / 'qpos.bin', np.float32).astype(float)
    head1, quat1 = q1[:3], q1[3:7]
    pts = scenarios(q1, head1, quat1)
    F = len(pts)
    d = files(F, f'eyes{F}', prey=True)
    q0 = np.fromfile(d / 'qpos.bin', np.float32).reshape(-1, F)
    head, quat = q0[:3, 0].astype(float), q0[3:7, 0].astype(float)
    world = [place(head, quat, n, az, el, dist) for n, az, el, dist, _ in pts]
    write_tables(d, [(*w, 0, 0, 1e9) for w in world])
    np.zeros(2 * F, np.float32).tofile(d / 'drive.bin')  # speed 0: the spider stands
    pk, _ = run(d, F, 1, 2, 3, packet=1, eyes=True)
    _, q, _, _, ret = pk[-1]
    tiles = [ret[i, [x[0] for x in EYES].index(pts[i][0])] for i in range(F)]  # what each scenario's eye saw, for a human (retinas.png)
    from PIL import Image
    cols = 6
    rows = [np.concatenate(tiles[r * cols:r * cols + cols] + [np.zeros_like(tiles[0])] * (cols - len(tiles[r * cols:r * cols + cols])), 1) for r in range((F + cols - 1) // cols)]
    Image.fromarray(np.concatenate(rows, 0)).resize((cols * NPIX * 3, len(rows) * NPIX * 3), Image.NEAREST).save('.work/spider-retinas.png')
    bad = []
    for i, (n, az, el, dist, kind) in enumerate(pts):
        e = [x[0] for x in EYES].index(n)
        u, v, cnt = blob(ret[i, e])
        eu, ev, z = project(world[i], q[:3, i], q[3:7, i], n)
        fpx = NPIX / 2 / np.tan(np.radians(INTENDED[n][2]) / 2)
        full = np.pi * (fpx * PREY_R / z) ** 2  # area of the prey's disc if nothing hides it (small angles)
        if kind == 'clear':
            err = float(np.hypot(u - eu, v - ev))
            ys, xs = np.nonzero(ret[i, e] < 15)
            spread = float(np.hypot(xs + 0.5 - u, ys + 0.5 - v).max()) if cnt else np.nan
            ok = cnt > 0 and err <= 0.75 and spread <= 1.3 * fpx * PREY_R / z + 1.5 and 0.6 * full < cnt < 1.5 * full
            print(f'{"PASS" if ok else "FAIL"} (1,2) {n} az {az:+d} el {el:+d} at {dist} mm, ray-clear: centroid ({u:.2f}, {v:.2f}) px vs analytic ({eu:.2f}, {ev:.2f}), '
                  f'err {err:.2f} px, {cnt} px (disc {full:.0f}), spread {spread:.1f} px')
        else:
            ok = cnt < 0.15 * full
            print(f'{"PASS" if ok else "FAIL"} (4) {n} az {az:+d} el {el:+d} at {dist} mm, own body in the way (ray cast): {cnt} px visible of a {full:.0f} px disc')
        bad += [] if ok else [f'{n} {az}/{el} {kind}']
    # (3) the environment moves the prey: still before t_on, constant velocity after; the brain's packet carries no prey state
    env = [(1.0, 2.0, 3.0, 10.0, -5.0, 0.0)] * (F // 2) + [(1.0, 2.0, 3.0, 10.0, -5.0, 1e9)] * (F - F // 2)
    write_tables(d, env)
    steps, k = 300, 10
    pk, _ = run(d, F, k, 2, steps, packet=steps, eyes=True)
    q = pk[-1][1]
    t = ((steps - 1) * k + k) * 1e-4
    moved, still = q[-7:-5, 0], q[-7:-5, F - 1]
    ok = (np.abs(moved - (np.array([1.0, 2.0]) + np.array([10.0, -5.0]) * t)).max() < 5e-3 and np.abs(still - np.array([1.0, 2.0])).max() < 5e-3)
    print(f'{"PASS" if ok else "FAIL"} (3) prey after {t * 1e3:.0f} ms: moving (x, y) {moved} want {np.array([1.0, 2.0]) + np.array([10.0, -5.0]) * t}, still {still}')
    bad += [] if ok else ['prey path']
    if bad:
        raise SystemExit(f'FAIL spider_eyes: {bad}')
    print('PASS all spider_eyes checks')


if __name__ == '__main__':
    {'check': check}[sys.argv[1]]()
