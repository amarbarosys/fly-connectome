"""Flybody flight in the engine (docs/plans/FLIGHT.md steps 4-5): the flight model with its wing beat pattern generator,
open loop (policy action 0), on live.mojo -DBRAIN=1 with K=4 (one frame = one 2e-4 s control step of four 5e-5 s steps).
Per control step the host reads the wing angles from the engine's packet and sends Flybody's force control
ctrl[wing] = target - qpos[wing] (FlightImitation.before_step, flybody/tasks/flight_imitation.py), every other ctrl 0;
target is the WingBeatPatternGenerator table at the base beat frequency (models/flight_wpg.py, flight venv).

check  the same open-loop rollout in MuJoCo (same MJCF, same ctrl rule from its own state) for 0.1 s (500 control
       steps, 22 wing beats): engine root position within 1 % of MuJoCo's displacement at every step, wing angles
       over the first beat within 1e-3 rad. Mutant: target shifted by half a beat must FAIL.
watch  live.py-style mpv window of the flapping fly (slow motion).

usage: gpu-wait run --priority 90 --timeout 1800 -- .venv/bin/python -m agent.body_engine.flight check|watch [STEPS]
"""
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from kinestride.model import FEET, export, load, meshes
from agent.body_engine.spider_link import build

R = Path(__file__).resolve().parents[2]
FL = R / '.work/flight-body'
OUT = R / '.work/agent/body_engine/flight'
K = 4  # flight dt 5e-5, Flybody control step 2e-4


def free_mjcf():
    """flight.xml with contacts off (the fly is airborne; no floor) and the pyramidal cone the export gate takes (no
    contacts, so the cone changes nothing), without the ghost fly. Written as FL/free.xml."""
    if not (FL / 'flight.xml').is_file():
        subprocess.run([str(R / '.work/neural-flies/flight-venv/bin/python'), str(R / 'agent/body_engine/models/flight_export.py'), str(FL)], check=True)
    if not (FL / 'wpg.npz').is_file():
        subprocess.run([str(R / '.work/neural-flies/flight-venv/bin/python'), str(R / 'agent/body_engine/models/flight_wpg.py'), str(FL)], check=True)
    tree = ET.parse(FL / 'flight.xml')
    root = tree.getroot()
    opt = root.find('option') if root.find('option') is not None else ET.SubElement(root, 'option')
    opt.set('cone', 'pyramidal')
    for g in root.iter('geom'):
        g.set('contype', '0'); g.set('conaffinity', '0')
    wb = root.find('worldbody')  # Flybody's ghost (the reference trajectory's marker fly): no joints, drawn only
    for b in [b for b in wb.findall('body') if b.get('name', '').startswith('ghost')]:
        wb.remove(b)
    for parent in root.iter():  # and everything naming it (sensors, exclusions)
        for e in [e for e in parent if any(v.startswith('ghost/') for k, v in e.attrib.items() if k != 'name')]:
            parent.remove(e)
    tree.write(FL / 'free.xml')
    return str(FL / 'free.xml')


def prepare():
    mjcf = free_mjcf()
    w = dict(np.load(FL / 'wpg.npz'))
    w = remap(load(str(FL / 'flight.xml')), load(mjcf), w)
    m = export(mjcf, OUT, FEET, True)  # FEET: the contact blocks STAGE 7 reads (the flight MJCF has no explicit pairs)
    meshes(mjcf, OUT, 40)
    w['qpos0'].astype(np.float32).reshape(-1, 1).tofile(OUT / 'qpos.bin')
    w['qvel0'].astype(np.float32).reshape(-1, 1).tofile(OUT / 'qvel.bin')
    np.zeros((m.nu, 1), np.float32).tofile(OUT / 'ctrl.bin')
    return mjcf, m, w


def remap(full, free, w):
    """wpg.npz indexes the env's model (with the ghost's free joint): move state and wing indices to the free model by name."""
    q, v = np.zeros(free.nq), np.zeros(free.nv)
    adr = {}
    for j in range(free.njnt):
        k = mujoco.mj_name2id(full, mujoco.mjtObj.mjOBJ_JOINT, free.joint(j).name)
        n = {0: 7, 1: 4}.get(int(free.jnt_type[j]), 1)
        q[free.jnt_qposadr[j]:free.jnt_qposadr[j] + n] = w['qpos0'][full.jnt_qposadr[k]:full.jnt_qposadr[k] + n]
        nd = {0: 6, 1: 3}.get(int(free.jnt_type[j]), 1)
        v[free.jnt_dofadr[j]:free.jnt_dofadr[j] + nd] = w['qvel0'][full.jnt_dofadr[k]:full.jnt_dofadr[k] + nd]
        adr[int(full.jnt_qposadr[k])] = int(free.jnt_qposadr[j])
    act = [mujoco.mj_name2id(free, mujoco.mjtObj.mjOBJ_ACTUATOR, full.actuator(a).name) for a in w['wing_act']]
    return w | {'qpos0': q, 'qvel0': v, 'wing_qadr': np.array([adr[int(a)] for a in w['wing_qadr']]), 'wing_act': np.array(act)}


def ctrl_of(m, w, q, i, shift=0):
    u = np.zeros(m.nu)
    t = w['target'][(i + shift) % len(w['target'])]
    u[w['wing_act']] = t - q[w['wing_qadr']]
    return u


def camera(m, q0, dist):
    d = mujoco.MjData(m)
    d.qpos[:] = q0
    mujoco.mj_forward(m, d)
    look = d.xpos[1]
    a, e = np.radians(100), np.radians(-25)
    eye = look - dist * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    return [*map(str, eye), *map(str, look), str(m.vis.global_.fovy)]


def engine(m, w, steps, shift=0, cam=None, fd4=None):
    """Open-loop rollout on the card: root qpos (steps, 7) and wing qpos (steps, 6) after each control step."""
    b = build(OUT, 1, K, 1, 960 if fd4 else 64, 540 if fd4 else 64)
    nq, nv, ns = m.nq, m.nv, int(dict(f[2:].split('=') for f in (OUT / 'sizes.txt').read_text().split()).get('NSLOT', 0))
    extra = dict(pass_fds=(4,), preexec_fn=lambda: __import__('os').dup2(fd4, 4)) if fd4 else {}
    p = subprocess.Popen([str(b), str(OUT) + '/', str(steps), *(cam or camera(m, w['qpos0'], 6.0)), '1' if fd4 else '0', '1'],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, **extra)
    q, root, wing = w['qpos0'].astype(np.float32), [], []
    for i in range(steps):
        try:
            p.stdin.write(ctrl_of(m, w, q, i, shift).astype(np.float32).tobytes()); p.stdin.flush()
            pk = p.stdout.read(4 * (m.nu + nq + nv + ns))
        except BrokenPipeError:
            break  # watch: mpv closed
        if len(pk) < 4 * (m.nu + nq + nv + ns):
            break
        q = np.frombuffer(pk, np.float32)[m.nu:m.nu + nq].astype(np.float64)
        root.append(q[:7]); wing.append(q[w['wing_qadr']])
    p.wait()
    return np.array(root), np.array(wing)


def mujoco_ref(mjcf, w, steps):
    m = load(mjcf)
    d = mujoco.MjData(m)
    d.qpos[:], d.qvel[:] = w['qpos0'], w['qvel0']
    root, wing = [], []
    for i in range(steps):
        d.ctrl[:] = ctrl_of(m, w, d.qpos, i)
        mujoco.mj_step(m, d, K)
        root.append(d.qpos[:7].copy()); wing.append(d.qpos[w['wing_qadr']].copy())
    return np.array(root), np.array(wing)


def compare(ref, got, w):
    beat = int(round(1 / 218 / 2e-4))  # control steps per wing beat
    disp = np.linalg.norm(ref[0][:, :3] - w['qpos0'][:3], axis=1).max()
    pos = np.linalg.norm(got[0][:, :3] - ref[0][:, :3], axis=1).max() / disp
    ang = np.abs(got[1][:beat] - ref[1][:beat]).max()
    return {'pass': bool(pos <= 0.01 and ang <= 1e-3), 'root_err_rel': float(pos), 'wing_err_rad': float(ang),
            'displacement': float(disp), 'z_end_ref': float(ref[0][-1, 2]), 'z_end_engine': float(got[0][-1, 2]), 'steps': len(got[0])}


def main(cmd='check', steps='500'):
    mjcf, m, w = prepare()
    n = int(steps)
    if cmd == 'watch':
        mpv = subprocess.Popen(['mpv', '--demuxer=rawvideo', '--demuxer-rawvideo-w=960', '--demuxer-rawvideo-h=540',
                                '--demuxer-rawvideo-mp-format=rgba', '--no-cache', '-'], stdin=subprocess.PIPE)
        engine(m, w, n or 1 << 30, cam=camera(m, w["qpos0"], 4.0), fd4=mpv.stdin.fileno())
        mpv.stdin.close(); mpv.wait()
        return
    ref = mujoco_ref(mjcf, w, n)
    r = compare(ref, engine(m, w, n), w)
    bad = compare(ref, engine(m, w, n, shift=int(round(1 / 218 / 2e-4 / 2))), w)
    print(('PASS' if r['pass'] else 'FAIL'), 'flight open loop', r)
    print(('PASS' if not bad['pass'] else 'FAIL'), 'mutant (half-beat phase shift) must fail', bad)
    if not r['pass'] or bad['pass']:
        sys.exit('FAIL flight')
    print('PASS flight: engine open-loop flapping flight matches MuJoCo')


if __name__ == '__main__':
    main(*sys.argv[1:])
