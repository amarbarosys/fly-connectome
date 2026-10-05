"""Brain<->body link for the spider (docs/plans/SPIDER_BRAIN.md unit 1): live.mojo -DBRAIN=1 (ctrl over stdin, host brain)
and -DBRAIN=2 (spider_brain.mojo gait kernel on the card), sensor packets on stdout.

  python -m agent.body_engine.spider_link check     gates 1-3 below (under gpu-wait)
  python -m agent.body_engine.spider_link watch    one spider walking a circle under the card brain, in an mpv window (under gpu-wait)
  python -m agent.body_engine.spider_link time      measured ms per control step, transport a vs b, K, animals (not a gate)

gates: (1) kernel gait ctrl == spider_gait.gait (oracle) to 1e-5; (2) engine rollout under the brain's ctrl stream == MuJoCo
mj_step under the SAME stream (20 control steps, K=1; bar max(1e-5, 3 x the float32 input floor) of the field, the
`model.check` convention); (3) closed loop: speed 1 walks > 1 body length (5.6 mm) in 1 s with tilt < 20 deg, speed 0 stands,
turn +0.8 turns left > 45 deg, contact packet shows feet down.
"""
import hashlib
import subprocess
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

from agent.body_engine.live import prepare
from agent.body_engine.model import ENGINE, load
from agent.body_engine.models.spider import build as spider_build
from agent.body_engine.render_check import MOJO
from agent.body_engine.spider_gait import NU, gait

OUT = Path('.work/agent/body_engine/spider-link')
CAM = ['0', '0', '10', '0', '0', '0', '45']
DT = 1e-4
BODY_MM = 5.6  # prosoma + opisthosoma, models/spider.py


def files(F, name=None, prey=False):
    """Engine + mesh files for F copies of the spider standing (prepare's state), servos at the stance, drives speed 1 turn 0."""
    d = OUT / (name or f'f{F}')
    if not (d / 'drive.bin').exists():
        cam, m = prepare(d, mjcf=str(spider_build(prey=prey)), dist=15.0)
        for n in ('qpos', 'qvel', 'ctrl'):
            a = np.fromfile(d / f'{n}.bin', np.float32).reshape(-1, 1)
            np.tile(a, (1, F)).astype(np.float32).tofile(d / f'{n}.bin')
        np.r_[np.ones(F), np.zeros(F)].astype(np.float32).tofile(d / 'drive.bin')
    return d


def build(d, F, k, brain, w=64, h=64, eyes=False, gaze=False):
    flags = (d / 'sizes.txt').read_text().split() + (d / 'rsizes.txt').read_text().split()
    nb = int([f for f in flags if f.startswith('-DNBODY=')][0][8:])
    src = [ENGINE / f for f in ('live.mojo', 'smooth.mojo', 'render.mojo', 'spider_brain.mojo', 'eyes.mojo')]
    tag = hashlib.sha1(b''.join(f.read_bytes() for f in src + [d / 'sizes.txt', d / 'rsizes.txt'])).hexdigest()[:8]  # content (incl. the model's sizes), not mtime: mutant-check restores old mtimes
    b = d / f'link_b{brain}_k{k}_{w}x{h}{"_eyes" if eyes else ""}{"_gaze" if gaze else ""}_{tag}'
    if not b.is_file():
        subprocess.run([MOJO, 'build', '-I', str(ENGINE), *flags, f'-DFLIES={F}', '-DSTAGE=7', f'-DK={k}', '-DWAVE=1', f'-DW={w}', f'-DH={h}',
                        f'-DPROWS={7 * (nb - 1)}', '-DBOX_MM=0', f'-DBRAIN={brain}', f'-DEYES={int(eyes)}', f'-DGAZE={int(gaze)}', str(ENGINE / 'live.mojo'), '-o', str(b)], check=True,
                       capture_output=True)
    return b


def sizes(d):
    s = dict(f[2:].split('=') for f in (d / 'sizes.txt').read_text().split())
    return int(s['NQ']), int(s['NV']), int(s['NSLOT'])


def run(d, F, k, brain, steps, packet=0, ctrl=None, eyes=False):
    """Run `steps` control steps; returns the packets [(u, q, v, valid)] (every `packet`-th step and the last) and seconds.
    brain 1: ctrl(step) -> (F, NU) written to stdin each step."""
    nq, nv, ns = sizes(d)
    p = subprocess.Popen([str(build(d, F, k, brain, eyes=eyes)), str(d) + '/', str(steps), *CAM, '0', str(packet)], stdin=subprocess.PIPE if brain == 1 else None,
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    row = lambda n: np.empty((n, F), np.float32)
    sent = [i for i in range(steps) if (packet and i % packet == 0) or i + 1 == steps]
    out, t0 = [], time.perf_counter()
    for i in range(steps):
        if brain == 1:
            p.stdin.write(np.ascontiguousarray(ctrl(i).T, np.float32).tobytes())
            p.stdin.flush()
        if i in sent:
            pk = tuple(np.frombuffer(p.stdout.read(4 * n * F), np.float32).reshape(n, F) for n in (NU, nq, nv, ns))
            if eyes:  # retinas, uint8 [animal][eye][H x W] (eyes.mojo)
                pk += (np.frombuffer(p.stdout.read(F * 6 * 64 * 64), np.uint8).reshape(F, 6, 64, 64),)
            out.append(pk)
    dt = time.perf_counter() - t0
    if p.wait() != 0:
        raise SystemExit(f'FAIL link: engine exited {p.returncode}')
    return out, dt


def yaw_of(q):
    w, x, y, z = q[3:7]
    return float(np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))))


def tilt_of(q):
    w, x, y, z = q[3:7]
    return float(np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1))))


def mujoco_chain(d, F, steps, ctrl, round32=False):
    m = load(str(spider_build()))
    m.opt.noslip_iterations = 0
    s = mujoco.MjData(m)
    r = (lambda x: x.astype(np.float32).astype(np.float64)) if round32 else (lambda x: x)
    s.qpos[:] = r(np.fromfile(d / 'qpos.bin', np.float32).reshape(-1, F)[:, 0].astype(np.float64))
    s.qvel[:] = r(np.fromfile(d / 'qvel.bin', np.float32).reshape(-1, F)[:, 0].astype(np.float64))
    for i in range(steps):
        s.ctrl[:] = r(ctrl(i)[0].astype(np.float64))
        mujoco.mj_step(m, s)
    return s.qpos.copy(), s.qvel.copy()


def check():
    fail = []
    d, F = files(8), 8
    drive = lambda i: (np.ones(F) * 0.7, np.linspace(-0.5, 0.5, F))
    open(d / 'drive.bin', 'wb').write(np.r_[drive(0)[0], drive(0)[1]].astype(np.float32).tobytes())
    # (1) kernel gait vs oracle at the last of 60 control steps (K=10)
    pk, _ = run(d, F, 10, 2, 60)
    u = pk[-1][0]
    want = gait(59 * 10 * DT, *drive(0)).T
    e = float(np.abs(u - want).max())
    print(f'{"PASS" if e <= 1e-5 else "FAIL"} (1) brain kernel ctrl vs oracle, max abs err {e:.2e} (bar 1e-5)')
    fail += [] if e <= 1e-5 else ['gait kernel']
    # (2) engine vs MuJoCo under the same ctrl stream, host brain, K=1
    steps = 20
    ctrl = lambda i: gait(i * DT * 10, *drive(0))  # 10x faster gait clock: the servos move every step
    pk, _ = run(d, F, 1, 1, steps, ctrl=ctrl)
    _, q, v, _ = pk[-1]
    qm, vm = mujoco_chain(d, F, steps, ctrl)
    qa, va = mujoco_chain(d, F, steps, ctrl, round32=True)
    for name, got, want_, alt in (('qpos', q[:, 0], qm, qa), ('qvel', v[:, 0], vm, va)):
        scale = np.abs(want_).max()
        err, floor = float(np.abs(got - want_).max() / scale), float(np.abs(alt - want_).max() / scale)
        bar = max(1e-5, 3 * floor)
        print(f'{"PASS" if err <= bar else "FAIL"} (2) {name} engine vs MuJoCo after {steps} steps: rel err {err:.2e}, float32 floor {floor:.2e}, bar {bar:.2e}')
        fail += [] if err <= bar else [f'parity {name}']
    # (3) closed loop, 1 s, speed 1 / 0, turn +0.8, K=10
    d1, F = files(4), 4
    np.r_[[1, 0, 1, 1], [0, 0, 0.8, -0.8]].astype(np.float32).tofile(d1 / 'drive.bin')  # speed 1 0 1 1, turn 0 0 +0.8 -0.8
    pk, _ = run(d1, F, 10, 2, 1000)
    u, q, v, valid = pk[-1]
    q0 = np.fromfile(d1 / 'qpos.bin', np.float32).reshape(-1, F)
    disp = np.linalg.norm(q[:2] - q0[:2], axis=0)
    yaw = np.array([yaw_of(q[:, i]) for i in range(F)])
    tilt = np.array([tilt_of(q[:, i]) for i in range(F)])
    feet = (valid > 0).sum(0)
    ok = disp[0] > BODY_MM and tilt.max() < 20 and disp[1] < 1.0 and yaw[2] > 45 and yaw[3] < -45 and (feet > 0).all()
    print(f'{"PASS" if ok else "FAIL"} (3) closed loop 1 s: displacement mm {np.round(disp, 1)} (walker > {BODY_MM}, zero drive < 1), yaw deg {np.round(yaw, 0)}, '
          f'tilt max {tilt.max():.1f}, valid contact slots {feet}')
    fail += [] if ok else ['closed loop']
    if fail:
        raise SystemExit(f'FAIL spider_link: {fail}')
    print('PASS all spider_link checks')


def timing():
    for F in (1, 64):
        d = files(F)
        for k in (1, 10):
            for brain in (1, 2):
                st = 200
                ctrl = lambda i: gait(i * k * DT, np.ones(F), np.zeros(F))
                pk = 1 if brain == 1 else 0  # a closed loop on the host reads the sensors every step; the card brain reads none
                run(d, F, k, brain, 20, packet=pk, ctrl=ctrl)  # warm: compile, load
                _, s = run(d, F, k, brain, st, packet=pk, ctrl=ctrl)
                print(f'F {F:3d} K {k:2d} transport {"a host" if brain == 1 else "b card"}: {1e3 * s / st:7.3f} ms per control step, '
                      f'{1e3 * s / st / k:6.3f} ms per physics step')


def watch(k=10):
    """One spider circling under the on-card brain (speed 0.6, turn 0.8), drawn every control step into mpv (K steps = K x 0.1 ms per frame)."""
    w, h = 960, 540
    d = files(1, 'watch')
    np.array([0.6, 0.8], np.float32).tofile(d / 'drive.bin')
    q0 = np.fromfile(d / 'qpos.bin', np.float32)
    look = q0[:3] + np.array([0.0, 6.0, 0.0])  # the circle's centre, left of the start
    cam = [str(x) for x in (*(look + np.array([0.0, -22.0, 14.0])), *look, 45)]
    mpv = subprocess.Popen(['mpv', '--demuxer=rawvideo', f'--demuxer-rawvideo-w={w}', f'--demuxer-rawvideo-h={h}',
                            '--demuxer-rawvideo-mp-format=rgba', '--no-cache', '-'], stdin=subprocess.PIPE)
    eng = subprocess.Popen(['/usr/bin/bash', '-c', f'exec "$@" 4>&{mpv.stdin.fileno()}', 'engine', str(build(d, 1, k, 2, w, h)), str(d) + '/', '0', *cam, '1', '0'],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, pass_fds=(mpv.stdin.fileno(),))
    mpv.wait()
    eng.terminate()


if __name__ == '__main__':
    {'check': check, 'time': timing, 'watch': watch}[sys.argv[1]]()
