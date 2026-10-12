"""Live view (live.mojo): prepare a standing fly, build, and check the path a user meets, or watch it.

check (under gpu-wait):
  frame  live's first frame (K=1: pose of the starting state) against render.mojo drawing the same state from
         smooth's pose file (STAGE 7, no K), same camera: silhouette IoU >= 0.999
  stand  300 frames at K=10 (0.3 s), servos holding the standing pose: the fly's silhouette (alpha of the raw
         stream) in the last frame overlaps the first at IoU >= 0.9, and frame-stats (iTools) motion over the last
         100 frames is < 0.02 (settled). Control arm: the same fly with only the 6 feet colliding (FEET 'tarsus5',
         which sinks through the floor in MuJoCo too) must fail
  leak   10,000 streamed frames, verify_render_leak's sampler at 100 and 10,000: fds equal, VRAM not up, RSS < +1 MB
fps: live.py fps K [...as watch]: 300 frames, ms per frame. watch: live.py watch [K] [cube] [spider] [world [SEED]] | [world-game]  -> mpv window (world-game: watch_world.py, everything on, fps reported) (cube: inside the closed 40 mm cube of model.cube, walls
       drawn; spider: the body from models/spider.py instead of the fly) (slow motion: the engine is ~140x slower than real time at one fly)

usage: gpu-wait run --priority 90 --timeout 1800 -- .venv/bin/python -m agent.body_engine.live check
"""
import json
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

from kinestride.model import ENGINE, FEET, combine, cube, export, load, meshes, standing, terrain
from kinestride.render_check import DIST, MJCF, MOJO
from kinestride.verify_render_leak import sample
from kinestride.model import world_height_grad as height_grad  # world.mojo CLI (world.py is its oracle)
APP = Path(__file__).parent  # this app's Mojo sources (live, spectate, gait check); the physics core is kinestride's ENGINE
IMKIT = Path.home() / 'Projects/mojo/mojo-imkit/src'  # the GUI package live.mojo imports (imkit: window, font, ui)

OUT = Path('.work/agent/body_engine/live')
W, H = 960, 540
STATS = Path.home() / 'iTools/video/frame-stats/frame-stats'


WORLD = (0, 0, 0.1, 8.0, 4, 1, 2.0, 0.5, 0.5, 40.0, 5.0, 40.0)  # 0.1 mm bumps (0.3 mm floors the fly), cones, seed 1


def roots(m):
    """qpos address of every free joint (one per animal)."""
    return [int(m.jnt_qposadr[j]) for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]


def world_camera(out, m, fovy):
    """Eye, target, fovy framing every animal: target the roots' mean, distance fitting their bounding sphere (+5 mm)
    to the fov, azimuth the first of 12 whose sight lines to every root clear the terrain (landmark cones occlude)."""
    q = np.fromfile(out / 'qpos.bin', np.float32).astype(float)
    t = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_NUMERIC, 'terrain')
    p = m.numeric_data[m.numeric_adr[t]:m.numeric_adr[t] + 16]
    look = np.mean([q[a:a + 3] for a in roots(m)], 0)
    dist = (max(np.linalg.norm(q[a:a + 3] - look) for a in roots(m)) + 5.0) / np.sin(np.radians(float(fovy)) / 2)
    az = np.radians(np.arange(-130, 230, 30))
    eyes = look + dist * np.stack([0.85 * np.cos(az), 0.85 * np.sin(az), np.full_like(az, 0.52)], 1)
    top = np.array([q[a:a + 3] for a in roots(m)])
    line = eyes[:, None, None] + np.linspace(0, 1, 200)[:190, None] * (top[None, :, None] - eyes[:, None, None])  # az, root, point
    clear = ((line[..., 2] > 0) & (line[..., 2] > height_grad(line[..., 0], line[..., 1], p)[0] + 0.2)).all((1, 2))  # one world.mojo run
    eye = eyes[np.argmax(clear) if clear.any() else -1]
    return [*map(str, eye), *map(str, look), str(fovy)]


def lift(m, q):
    """in place: every free root of qpos q raised so its animal's bodies sit 0.05 mm above the terrain under them."""
    t = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_NUMERIC, 'terrain')
    d = mujoco.MjData(m)
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    j = np.flatnonzero(m.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    h = height_grad(d.xpos[:, 0], d.xpos[:, 1], m.numeric_data[m.numeric_adr[t]:m.numeric_adr[t] + 16])[0]  # one world.mojo run
    for a, r in zip(roots(m), m.body_rootid[m.jnt_bodyid[j]]):  # this animal's bodies: the ground under them, not the
        q[a + 2] += float(h[m.body_rootid == r].max()) + 0.05  # highest within 4 mm (a 0.57 mm drop on a 0.1 ramp)


def prepare(out=OUT, pairs=FEET, mjcf=MJCF, dist=DIST, flat=None, convex=False, world=None):
    """Engine + mesh files for one fly standing still: state from model.standing, servos holding its joint angles,
    adhesion on. Returns the camera args (eye, target, fovy) and the model. flat: on terrain, stand on this flat-floor
    MJCF instead and lift the animal 0.05 mm above the highest ground within 4 mm of it (MuJoCo's own rollout on the
    sampled heightfield jams thin feet through its prisms: root 2.05 mm up, feet 0.81 mm under the surface).
    convex: also export box/ellipsoid-vs-mesh pairs (GJK + EPA, check.sh convexstep), else they are dropped."""
    out.mkdir(parents=True, exist_ok=True)
    m = load(mjcf)
    m.opt.noslip_iterations = 0  # normal friction: noslip 5 makes the standing fly tremble, in MuJoCo too (Brain 2026-10-05)
    m = export(m, out, pairs, True, convex=convex)
    meshes(mjcf, out, world=world)  # world: the world's walls, stones and assets drawn too (model.meshes)
    q, v = standing(load(flat or mjcf), 2)  # shipped noslip 5: with noslip 0 the settling rollout collapses onto the belly
    q, v = q[:1], v[:1]
    if flat:  # every free root (fly, spider) lifted above its own ground
        lift(m, q[0])
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
    flags += ['-DWAVE=1']  # one animal on a 32-lane wave (smooth.mojo WAVE): the live step is latency, not throughput
    flags += ['-DTERRAIN=1'] if any(f.startswith('-DHFGEOM') for f in flags) else []
    binary = out / f'live_k{k}_b{box}'
    src = [APP / 'live.mojo'] + [ENGINE / f for f in ('smooth.mojo', 'render.mojo')] + [out / 'sizes.txt', out / 'rsizes.txt']
    if binary.is_file() and binary.stat().st_mtime > max(f.stat().st_mtime for f in src):
        return binary  # built on the CPU before the GPU job
    subprocess.run([MOJO, 'build', '-I', str(ENGINE), '-I', str(IMKIT), *flags, '-DFLIES=1', '-DSTAGE=7', f'-DK={k}', f'-DW={W}', f'-DH={H}',
                    f'-DPROWS={7 * (m.nbody - 1)}', f'-DBOX_MM={box}', str(APP / 'live.mojo'), '-o', str(binary)], check=True)
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


def upright(out, m, cam, frames=300, flip=False):
    """Per animal (free root): (cos of its root's tilt from the upright start pose, lowest vertex of the root body's
    first geom above the terrain: fly thorax, spider prosoma) after `frames` frames; flip starts every animal upside
    down (rolled 180 deg about its root's x axis, the gate's negative control)."""
    q0 = np.fromfile(out / 'qpos.bin', np.float32)
    if flip:
        q1 = q0.copy()
        for a in roots(m):
            r = np.zeros(4)
            mujoco.mju_mulQuat(r, q0[a + 3:a + 7].astype(float), np.array([0., 1, 0, 0]))
            q1[a + 3:a + 7], q1[a + 2] = r, q1[a + 2] + 1.0
        q1.tofile(out / 'qpos.bin')
    subprocess.run([str(build(out, m, 10)), str(out) + '/', str(frames), *cam, str(out / 'last.rgba')], check=True, stderr=subprocess.DEVNULL)
    d = mujoco.MjData(m)
    d.qpos[:] = np.fromfile(out / 'last.rgba.qpos', np.float32)
    mujoco.mj_kinematics(m, d)
    t = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_NUMERIC, 'terrain')
    p = m.numeric_data[m.numeric_adr[t]:m.numeric_adr[t] + 16] if t >= 0 else (0, 0, 0, 1, 0, 0, 2, 0.5)
    res = []
    for j in range(m.njnt):
        if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
            continue
        a, b = m.jnt_qposadr[j], m.jnt_bodyid[j]
        r0 = np.zeros(9)
        mujoco.mju_quat2Mat(r0, q0[a + 3:a + 7].astype(float))
        g, k = m.body_geomadr[b], m.geom_dataid[m.body_geomadr[b]]
        v = m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k] + m.mesh_vertnum[k]] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        res.append((float(r0[2::3] @ d.xmat[b].reshape(3, 3)[:, 2]), float((v[:, 2] - height_grad(v[:, 0], v[:, 1], p)[0]).min())))
    return res


def upright_check(terrain_mjcf, flat=MJCF):
    """Stands on terrain: dropped from the lifted flat pose, after 300 frames (3000 steps) the root has tilted under
    30 deg from its start pose (the standing thorax frame is itself pitched, up z 0.42) and the thorax clears the
    ground by at least half its flat standing clearance and 0.2 mm (flat 0.48 mm; the floor catches a flat run that
    itself lies down, 74485d5; a fly lying on its belly, feet down or not,
    is at 0: 0.3 mm bumps measured 0.002). The same gate must PASS on the flat floor and FAIL started upside down.
    (ctrl 0 is no negative control: angle 0 is near the standing pose, the fly stood with it.)"""
    res, ref = {}, None
    for name, mjcf, off in (('flat', flat, False), ('terrain', terrain_mjcf, False), ('terrain flipped (must fail)', terrain_mjcf, True)):
        out = OUT.parent / ('upright-' + name.split()[0] + ('-flipped' if off else ''))
        cam, m = prepare(out, mjcf=mjcf, flat=None if mjcf == flat else flat)
        got = upright(out, m, cam, flip=off)
        ref = [c for _, c in got] if ref is None else ref
        res[name] = {'cos_tilt': [t for t, _ in got], 'clearance': [c for _, c in got], 'flat_clearance': ref,
                     'pass': bool(all(t > np.cos(np.radians(30)) and c > max(0.5 * r, 0.2) for (t, c), r in zip(got, ref)))}
        print(('PASS' if res[name]['pass'] else 'FAIL'), name, json.dumps(res[name]))
    good = res['flat']['pass'] and res['terrain']['pass'] and not res['terrain flipped (must fail)']['pass']
    print('PASS upright on terrain' if good else 'FAIL upright on terrain')
    return good


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
    if cmd in ('watch', 'fps'):  # fps: live.py fps K [world ...]: 300 frames to a file, prints ms per frame
        k = int(args[0]) if args else 10
        if 'world-game' in args:  # spider under the card brain + follow camera + game bars + plume text and mini-map, one window (watch_world.py)
            from agent.body_engine import watch_world
            return (watch_world.watch if cmd == 'watch' else watch_world.fps)(k)
        box = 40 if 'cube' in args else 0
        body = MJCF
        if 'spider' in args:
            from kinestride.models.spider import build as spider
            body = str(spider())
        out = OUT.parent / ('live' + ('-cube' if box else '') + ('-spider' if 'spider' in args else '') + ('-world' if 'world' in args else ''))
        if 'world' in args:  # fly + spider on the endless seeded terrain: live.py watch K world [SEED]
            from kinestride.models.spider import build as spider
            body = combine(MJCF, str(spider()))
            seed = next((int(x) for x in args[1:] if x.isdigit()), 1)
            cam, m = prepare(out, mjcf=terrain(body, WORLD[:5] + (seed,) + WORLD[6:])[0], flat=body)
            cam = world_camera(out, m, cam[6])
        else:
            cam, m = prepare(out, mjcf=cube(body, box) if box else body, dist=15.0 if 'spider' in args else DIST)
        if box:  # from inside a corner of the cube, looking at the animal
            look = np.array(cam[3:6], float)
            cam = [*map(str, look + np.array([12., -12., 8.]) * (1.6 if 'spider' in args else 1)), *cam[3:]]
        if cmd == 'fps':
            print(subprocess.run([str(build(out, m, k, box)), str(out) + '/', '300', *cam, str(out / 'fps.rgba')], check=True, capture_output=True, text=True).stdout.strip())
            return
        live = subprocess.Popen([str(build(out, m, k, box)), str(out) + '/', '0', *cam], stdout=subprocess.PIPE)
        subprocess.run(['mpv', '--demuxer=rawvideo', f'--demuxer-rawvideo-w={W}', f'--demuxer-rawvideo-h={H}',
                        '--demuxer-rawvideo-mp-format=rgba', '-'], stdin=live.stdout)
        return
    if cmd == 'upright':  # live.py upright TERRAIN_MJCF [FLAT_MJCF]
        sys.exit(0 if upright_check(*args) else 1)
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
