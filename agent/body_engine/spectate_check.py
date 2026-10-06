"""Spectator gates for live.mojo (-DFOLLOW=1 -DHUD=1 -DPROFILE=1; docs/plans/RENDERER.md): the camera follows the moving spider,
the HUD bars show the state buffer, hooks off changes no byte, and the frame cost and leak are measured in the integrated path.

  python -m agent.body_engine.spectate_check build        compile every variant on the CPU (run before the GPU job)
  gpu-wait run --priority 60 --timeout 1800 -- .venv/bin/python -m agent.body_engine.spectate_check check
  gpu-wait run ... -- .venv/bin/python -m agent.body_engine.spectate_check clip     (the framing gate alone: the mutant's gate)
  gpu-wait run ... -- .venv/bin/python -m agent.body_engine.spectate_check hud      (the bar-pixel gate alone)
  gpu-wait run ... -- .venv/bin/python -m agent.body_engine.spectate_check off      (the hooks-off byte gate alone)

check = (1) hooks off: a frame of the starting state from live.mojo at main (pre-hook) vs the current file with FOLLOW=HUD=0,
byte-equal; (2) the 300-frame clip of the spider walking (merged brain, BRAIN=2): spectator_integration_check.py on the receipt
(moving clip framed and centred, root travel, frame cost split, leak at frames 100 and 10,000); (3) HUD bars: pixel extent of each
fill == hud.normalized_extent of the state buffer, dead state grey, and HUD pixels carry alpha 254 (255 stays "animal").
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from agent.body_engine import spider_link
from agent.body_engine.model import ENGINE
from agent.body_engine.render_check import MOJO
from agent.body_engine.verify_render_leak import sample

OUT = Path('.work/agent/body_engine/spectate')
W, H, K = 640, 360, 10
CAM = ['0', '0', '10', '0', '0', '0', '45']  # overwritten every frame by follow
ROOT_BODY, ROOT_RADIUS = 1, 8.0  # the spider's root body and a sphere around it covering the legs (mm)
CHECKER = ENGINE / 'spectator_integration_check.py'
PATCH = Path('/home/<user>/Projects/science/fly-connectome/.work/team/status/spectator-hooks/integration-gates.patch')


def hud_geometry():
    src = (ENGINE / 'spectate.mojo').read_text()
    return {k: int(re.search(rf'comptime {k} = (\d+)', src).group(1)) for k in ('BAR_X', 'BAR_Y', 'BAR_W', 'BAR_H', 'BAR_PITCH')}


def files():
    d = spider_link.files(1, 'spectate')
    np.array([ROOT_BODY, ROOT_RADIUS], np.float32).tofile(d / 'roots.bin')
    np.array([1.0, 0.0], np.float32).tofile(d / 'drive.bin')  # speed 1, turn 0: a straight walk
    set_state(d, 1, 1, 1, 0)
    return d


def set_state(d, energy, health, stamina, dead):
    np.array([energy, health, stamina, dead], np.float32).tofile(d / 'game.bin')


def build(d, tag, flags, k=K, src=None, brain=2, flies=1):
    """Content-addressed build (mutant-check restores old mtimes): live.mojo `src` (default the current one), every engine
    module it may import, and the model's sizes (a re-export with other sizes, e.g. convex pairs on, needs its own binary)."""
    src = src or ENGINE / 'live.mojo'
    deps = [src] + sorted(f for f in ENGINE.glob('*.mojo') if f != src)
    sz = (d / 'sizes.txt').read_text().split() + (d / 'rsizes.txt').read_text().split()
    h = hashlib.sha1(b''.join(f.read_bytes() for f in deps) + ' '.join(flags + sz + [f'F{flies}']).encode() + bytes([brain])).hexdigest()[:8]
    b = d / f'{tag}_k{k}_{W}x{H}_{h}'
    if not b.is_file():
        nb = int([f for f in sz if f.startswith('-DNBODY=')][0][8:])
        subprocess.run([MOJO, 'build', '-I', str(ENGINE), *sz, f'-DFLIES={flies}', '-DSTAGE=7', f'-DK={k}', '-DWAVE=1', f'-DW={W}', f'-DH={H}', f'-DPROWS={7 * (nb - 1)}',
                        '-DBOX_MM=0', f'-DBRAIN={brain}', *flags, str(src), '-o', str(b)], check=True, capture_output=True)
    return b


def pre_hook_source():
    """live.mojo as merged before the spectator hooks (the byte-equal reference), beside the engine files it imports."""
    p = OUT / 'pre' / 'live.mojo'
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(subprocess.run(['git', 'show', '5faa668:agent/body_engine/live.mojo'], check=True, capture_output=True, text=True).stdout)
    return p


def variants(d):
    return {'pre': build(d, 'pre', [], 1, pre_hook_source()), 'off': build(d, 'off', [], 1),
            'spec': build(d, 'spec', ['-DFOLLOW=1', '-DHUD=1', '-DPROFILE=1']), 'leak': build(d, 'leak', ['-DFOLLOW=1', '-DHUD=1'], 1)}


def run(binary, d, frames, out, extra=(), cam=CAM):
    """One engine run with the frames on fd 4 -> file `out`; returns (stderr, seconds)."""
    t = time.perf_counter()
    r = subprocess.run(['bash', '-c', 'exec "$0" "$@" 4>"$FRAMES" >/dev/null', str(binary), str(d) + '/', str(frames), *cam, '1', '0', *extra],
                       stderr=subprocess.PIPE, text=True, env={**os.environ, 'FRAMES': str(out)})  # the frames are fd 4 (preexec_fn dup2s are closed again by close_fds)
    if r.returncode:
        raise SystemExit(f'FAIL live exited {r.returncode}: {r.stderr[-400:]}')
    if Path(out).stat().st_size != frames * W * H * 4:
        raise SystemExit(f'FAIL live wrote {Path(out).stat().st_size} bytes for {frames} frames of {W}x{H}')
    return r.stderr, time.perf_counter() - t


def leak(binary, d, frames=10_000, early=100):
    p = subprocess.Popen(['bash', '-c', 'exec "$0" "$@" 4>&1 >/dev/null', str(binary), str(d) + '/', str(frames), *CAM, '1', '0'],
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    got, size = [], W * H * 4
    with p.stdout as pipe:
        for n in range(1, frames + 1):
            left = size
            while left:
                chunk = pipe.read(left)
                if not chunk:
                    raise SystemExit(f'FAIL leak: stream ended at frame {n}')
                left -= len(chunk)
            if n in (early, frames):
                got.append(sample(p.pid))
    if p.wait() != 0:
        raise SystemExit(f'FAIL leak: live exited {p.returncode}')
    (f0, v0, r0), (f1, v1, r1) = got
    return {'sample_frames': [early, frames], 'fds': [f0, f1], 'vram_kib': [v0, v1], 'rss_kib': [r0, r1]}


def hooks_off(d, v):
    run(v['pre'], d, 1, OUT / 'baseline-before.rgba')
    run(v['off'], d, 1, OUT / 'baseline-disabled.rgba')
    same = (OUT / 'baseline-before.rgba').read_bytes() == (OUT / 'baseline-disabled.rgba').read_bytes() and (OUT / 'baseline-before.rgba').stat().st_size == W * H * 4
    print(('PASS' if same else 'FAIL'), 'hooks off: frame byte-equal to the pre-hook live.mojo (5faa668)')
    return same


def clip(d, v, frames=300):
    err, _ = run(v['spec'], d, frames, OUT / 'follow.rgba', [str(OUT / 'roots.f32')])
    cost = re.search(r'COST physics (\S+) neural_control (\S+) render (\S+) copy (\S+) display (\S+) frames (\d+)', err)
    names = ('physics', 'neural_control', 'render', 'copy', 'display')
    ms = dict(zip(names, map(float, cost.groups()[:5])))
    total = sum(ms.values())
    return {'width': W, 'height': H, 'clip_frames': frames, 'animals': 1, 'baseline_reference_rgba': 'baseline-before.rgba',
            'baseline_disabled_rgba': 'baseline-disabled.rgba', 'moving_clip_rgba': 'follow.rgba', 'root_trace_f32': 'roots.f32',
            'min_root_travel_mm': 1.0, 'rendered_animals': 1, 'animal_count_semantics': 'bodies in a shared physical scene',
            'frame_cost_ms': ms, 'achieved_fps': 1000 / total, 'real_time_factor': K * 1e-4 / (total / 1000)}


def hud_gate(d, v):
    g = hud_geometry()
    ok = True
    for name, state in (('alive', (0.75, 0.25, 0.5, 0)), ('dead', (0.75, 0.25, 0.5, 1))):
        set_state(d, *state)
        run(v['spec'], d, 1, OUT / f'hud-{name}.rgba', [str(OUT / 'hud-roots.f32')])
        img = np.fromfile(OUT / f'hud-{name}.rgba', np.uint8).reshape(H, W, 4)
        for i, value in enumerate(state[:3]):
            y = g['BAR_Y'] + g['BAR_PITCH'] * i + g['BAR_H'] // 2
            row = img[y, g['BAR_X']:g['BAR_X'] + g['BAR_W']]
            track = (row[:, :3] == (0x28, 0x28, 0x28)).all(1)
            fill = ~track
            want = int(round(value * g['BAR_W']))
            hud_px = (row[:, 3] == 254).all()
            grey = bool((row[fill][:, :3] == 0x80).all()) if state[3] else bool((row[fill][:, :3] != 0x80).all())
            good = int(fill.sum()) == want and hud_px and grey and bool(fill[:want].all())
            ok &= good
            print(('PASS' if good else 'FAIL'), f'HUD {name} bar {i}: {int(fill.sum())} px filled (normalized_extent {value} x {g["BAR_W"]} = {want}), alpha 254, colour {"grey" if state[3] else "bar"}')
    set_state(d, 1, 1, 1, 0)
    return ok


def tight(path):
    """Framing tighter than the integration checker's (centre 0.15-0.85, no edge clipping, which a camera 30 mm off the animal still passes, 2.7 % of
    the frame at 0.38/0.42): in all frames the silhouette centroid within 0.4-0.6 of both axes and the silhouette at least 2 % of the frame."""
    a = np.memmap(path, np.uint8, 'r').reshape(-1, H, W, 4)
    lo, hi, size = np.ones(2), np.zeros(2), 1.0
    for f in a:
        ys, xs = np.nonzero(f[..., 3] == 255)
        c = np.array([xs.mean() / W, ys.mean() / H]) if len(xs) else np.array([-1.0, -1.0])
        lo, hi, size = np.minimum(lo, c), np.maximum(hi, c), min(size, len(xs) / (W * H))
    ok = bool((lo >= 0.4).all() and (hi <= 0.6).all() and size >= 0.02)
    print(('PASS' if ok else 'FAIL'), f'follow framing tight: centroid x {lo[0]:.3f}..{hi[0]:.3f} y {lo[1]:.3f}..{hi[1]:.3f} (0.4-0.6), smallest silhouette {100 * size:.2f} % of the frame (>= 2 %)')
    return ok


def integration(receipt):
    if not CHECKER.exists():
        subprocess.run(['git', 'apply', str(PATCH.resolve())], check=True)
    r = subprocess.run([sys.executable, str(CHECKER), str(receipt)], capture_output=True, text=True)
    print(r.stdout + r.stderr, end='')
    return r.returncode == 0


def main(cmd):
    OUT.mkdir(parents=True, exist_ok=True)
    d = files()
    v = variants(d)
    if cmd == 'build':
        print('built', *map(str, v.values()))
        return
    if cmd == 'hud':
        sys.exit(0 if hud_gate(d, v) else 1)
    if cmd == 'off':
        sys.exit(0 if hooks_off(d, v) else 1)
    ok = True
    if cmd == 'check':
        ok &= hooks_off(d, v)
        ok &= hud_gate(d, v)
    receipt = clip(d, v)
    if cmd == 'check':
        receipt['leak'] = leak(v['leak'], d)
        print(('PASS' if receipt['leak']['fds'][0] == receipt['leak']['fds'][1] else 'FAIL'), 'leak', json.dumps(receipt['leak']))
    else:  # clip: the framing gate alone (no leak run), so the receipt keeps the last leak numbers if any
        old = OUT / 'receipt.json'
        receipt['leak'] = json.loads(old.read_text())['leak'] if old.exists() else {'sample_frames': [100, 10000], 'fds': [0, 0], 'vram_kib': [0, 0], 'rss_kib': [0, 0]}
        if not (OUT / 'baseline-before.rgba').exists():
            hooks_off(d, v)
    (OUT / 'receipt.json').write_text(json.dumps(receipt, indent=1))
    ok &= integration(OUT / 'receipt.json')
    ok &= tight(OUT / 'follow.rgba')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'check')
