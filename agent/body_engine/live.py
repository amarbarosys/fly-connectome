"""Live view (live.mojo): prepare a standing fly, build, and check the path a user meets, or watch it.

check (under gpu-wait):
  frame  live's first frame (K=1: pose of the starting state) against render.mojo drawing the same state from
         smooth's pose file (STAGE 7, no K), same camera: silhouette IoU >= 0.999
  stand  300 frames at K=10 (0.3 s), servos holding the standing pose: the fly's silhouette (alpha of the raw
         stream) in the last frame overlaps the first at IoU >= 0.9, and frame-stats (iTools) motion over the last
         100 frames is < 0.02 (settled). Control arm: the same fly with only the 6 feet colliding (FEET 'tarsus5',
         which sinks through the floor in MuJoCo too) must fail
  leak   10,000 streamed frames, verify_render_leak's sampler at 100 and 10,000: fds equal, VRAM not up, RSS < +1 MB
watch: live.py watch [K] [cube] [spider]  -> mpv window (cube: inside the closed 40 mm cube of model.cube, walls
       drawn; spider: the body from models/spider.py instead of the fly) (slow motion: the engine is ~140x slower than real time at one fly)

usage: gpu-wait run --priority 90 --timeout 1800 -- .venv/bin/python -m agent.body_engine.live check
"""
import json
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

from agent.body_engine.model import ENGINE, FEET, cube, export, meshes, standing
from agent.body_engine.render_check import DIST, MJCF, MOJO
from agent.body_engine.verify_render_leak import sample

OUT = Path('.work/agent/body_engine/live')
W, H = 960, 540
STATS = Path.home() / 'iTools/video/frame-stats/frame-stats'


def prepare(out=OUT, pairs=FEET, mjcf=MJCF, dist=DIST):
    """Engine + mesh files for one fly standing still: state from model.standing, servos holding its joint angles,
    adhesion on. Returns the camera args (eye, target, fovy) and the model."""
    out.mkdir(parents=True, exist_ok=True)
    m = export(mjcf, out, pairs, True)
    meshes(mjcf, out)
    q, v = standing(m, 2)
    q, v = q[:1], v[:1]
    u = np.where(m.actuator_trntype == mujoco.mjtTrn.mjTRN_BODY, 1.0, q[0][m.jnt_qposadr[np.minimum(m.actuator_trnid[:, 0], m.njnt - 1)]])
    q.astype(np.float32).T.copy().tofile(out / 'qpos.bin')
    v.astype(np.float32).T.copy().tofile(out / 'qvel.bin')
    u.astype(np.float32).reshape(-1, 1).tofile(out / 'ctrl.bin')
    d = mujoco.MjData(m)
    d.qpos[:] = q[0]
    mujoco.mj_forward(m, d)
    look = d.xpos[1]
    a, e = np.radians(90), np.radians(-20)
    eye = look - dist * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    return [*map(str, eye), *map(str, look), str(m.vis.global_.fovy)], m


def build(out, m, k, box=0):
    flags = (out / 'sizes.txt').read_text().split() + (out / 'rsizes.txt').read_text().split()
    binary = out / f'live_k{k}_b{box}'
    src = [ENGINE / f for f in ('live.mojo', 'smooth.mojo', 'render.mojo')] + [out / 'sizes.txt', out / 'rsizes.txt']
    if binary.is_file() and binary.stat().st_mtime > max(f.stat().st_mtime for f in src):
        return binary  # built on the CPU before the GPU job
    subprocess.run([MOJO, 'build', '-I', str(ENGINE), *flags, '-DFLIES=1', '-DSTAGE=7', f'-DK={k}', f'-DW={W}', f'-DH={H}',
                    f'-DPROWS={7 * (m.nbody - 1)}', f'-DBOX_MM={box}', str(ENGINE / 'live.mojo'), '-o', str(binary)], check=True)
    return binary


def frame_check(out, m, cam):
    live = build(out, m, 1)
    subprocess.run([str(live), str(out) + '/', '1', *cam, str(out / 'live.rgba')], check=True, capture_output=True)
    flags = (out / 'sizes.txt').read_text().split()
    subprocess.run([MOJO, 'build', *flags, '-DFLIES=1', '-DSTAGE=7', str(ENGINE / 'smooth.mojo'), '-o', str(out / 'pose1')], check=True)
    subprocess.run([str(out / 'pose1'), str(out) + '/'], check=True, capture_output=True)
    prows = (out / 'smooth.bin').stat().st_size // 4
    subprocess.run([MOJO, 'build', f'-DW={W}', f'-DH={H}', f'-DNBODY={m.nbody}', '-DFLIES=1', f'-DPROWS={prows}',
                    *(out / 'rsizes.txt').read_text().split(), str(ENGINE / 'render.mojo'), '-o', str(out / 'render1')], check=True)
    subprocess.run([str(out / 'render1'), str(out) + '/', str(out / 'smooth.bin'), str(out / 'ref.rgba'), '1', *cam, '0'],
                   check=True, capture_output=True)
    a, b = (np.fromfile(out / f, np.uint8).reshape(H, W, 4)[..., 3] == 255 for f in ('live.rgba', 'ref.rgba'))
    iou = (a & b).sum() / max((a | b).sum(), 1)
    return {'iou': float(iou), 'pixels': int(a.sum()), 'byte_equal': (out / 'live.rgba').read_bytes() == (out / 'ref.rgba').read_bytes(),
            'pass': bool(iou >= 0.999 and a.sum() > 0)}


def stand(out, binary, cam, frames, video):
    """(silhouette IoU first vs last frame, mean frame-stats motion over the last third) of one streamed run."""
    live = subprocess.Popen([str(binary), str(out) + '/', str(frames), *cam], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', f'{W}x{H}', '-r', '30', '-i', '-',
                            '-c:v', 'libx264', '-qp', '0', '-pix_fmt', 'yuv420p', str(video)], stdin=subprocess.PIPE)
    first = last = None
    for _ in range(frames):
        buf = live.stdout.read(W * H * 4)
        if len(buf) != W * H * 4:
            raise SystemExit('FAIL stand: stream ended early')
        enc.stdin.write(buf)
        alpha = np.frombuffer(buf, np.uint8).reshape(H, W, 4)[..., 3] == 255
        first = alpha if first is None else first
        last = alpha
    enc.stdin.close()
    if live.wait() != 0 or enc.wait() != 0:
        raise SystemExit('FAIL stand: live or ffmpeg failed')
    lines = subprocess.run([str(STATS), '--sw', str(video)], capture_output=True, text=True, check=True).stdout.strip().splitlines()
    motion = [json.loads(x)['motion'] for x in lines[:-1]]
    return float((first & last).sum() / max((first | last).sum(), 1)), float(np.mean(motion[-frames // 3:]))


def stand_check(out, m, cam):
    iou, settle = stand(out, build(out, m, 10), cam, 300, out / 'held.mkv')
    ctrl_out = out.parent / 'live-feet'
    cam6, m6 = prepare(ctrl_out, 'tarsus5')
    iou6, settle6 = stand(ctrl_out, build(ctrl_out, m6, 10), cam6, 300, ctrl_out / 'feet.mkv')
    ok, control_fails = iou >= 0.9 and settle < 0.02, not (iou6 >= 0.9 and settle6 < 0.02)
    return {'iou_first_last': iou, 'motion_last_third': settle, 'feet_only_control': {'iou': iou6, 'motion': settle6},
            'pass': bool(ok and control_fails)}


def leak_check(out, m, cam, frames=10_000, early=100):
    binary = out / 'live_k1_b0'
    p = subprocess.Popen([str(binary), str(out) + '/', str(frames), *cam], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    got = []
    for n in range(1, frames + 1):
        if len(p.stdout.read(W * H * 4)) != W * H * 4:
            raise SystemExit(f'FAIL leak: stream ended at frame {n}')
        if n in (early, frames):
            got.append(sample(p.pid))
    p.stdout.close()
    if p.wait() != 0:
        raise SystemExit(f'FAIL leak: live exited {p.returncode}')
    (f0, v0, r0), (f1, v1, r1) = got
    return {'fds': [f0, f1], 'vram_kib': [v0, v1], 'rss_kib': [r0, r1], 'pass': bool(f0 == f1 and v1 <= v0 and r1 - r0 < 1024)}


def main(cmd, *args):
    if cmd == 'watch':
        k = int(args[0]) if args else 10
        box = 40 if 'cube' in args else 0
        body = MJCF
        if 'spider' in args:
            from agent.body_engine.models.spider import build as spider
            body = str(spider())
        out = OUT.parent / ('live' + ('-cube' if box else '') + ('-spider' if 'spider' in args else ''))
        cam, m = prepare(out, mjcf=cube(body, box) if box else body, dist=15.0 if 'spider' in args else DIST)
        if box:  # from inside a corner of the cube, looking at the animal
            look = np.array(cam[3:6], float)
            cam = [*map(str, look + np.array([12., -12., 8.]) * (1.6 if 'spider' in args else 1)), *cam[3:]]
        live = subprocess.Popen([str(build(out, m, k, box)), str(out) + '/', '0', *cam], stdout=subprocess.PIPE)
        subprocess.run(['mpv', '--demuxer=rawvideo', f'--demuxer-rawvideo-w={W}', f'--demuxer-rawvideo-h={H}',
                        '--demuxer-rawvideo-mp-format=rgba', '-'], stdin=live.stdout)
        return
    cam, m = prepare()
    result = {'frame': frame_check(OUT, m, cam)}
    result['stand'] = stand_check(OUT, m, cam)
    result['leak'] = leak_check(OUT, m, cam)
    for name, r in result.items():
        print(('PASS' if r['pass'] else 'FAIL'), name, json.dumps(r))
    bad = [n for n, r in result.items() if not r['pass']]
    if bad:
        sys.exit(f'FAIL {len(bad)}/{len(result)} live checks: {" ".join(bad)}')
    print('PASS live: frame, stand, leak')


if __name__ == '__main__':
    main(*(sys.argv[1:] or ['check']))
