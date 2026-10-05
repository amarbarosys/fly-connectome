"""`live.py watch K world-game`: one command to watch everything the live view has: the spider walking a circle under the card brain (BRAIN=2),
the camera following it, energy/health/stamina bars from the engine's own game rows (work drain, death gates the servos), the plume sampled at
its leg I tarsus tips with the concentrations printed and the sources drawn in a mini-map, all in one mpv window, with the frame rate reported.

  gpu-wait run --priority 90 --timeout 1800 -- .venv/bin/python -m agent.body_engine.live watch 10 world-game   (mpv window; ms per frame and fps every 100 frames)
  gpu-wait run --priority 90 --timeout 1800 -- .venv/bin/python -m agent.body_engine.live fps 10 world-game     (300 frames, no window: the fps; timing: exclusive)
  gpu-wait run --shared --vram 1 --priority 60 --timeout 1800 -- .venv/bin/python -m agent.body_engine.watch_world check   (what a user meets, checked: below)

Demo rules (not the gate's): the energy drains in about 1.7 s of simulated time (60 s of wall time at K=10), so the bars visibly move and the
animal dies and goes limp; the plume descriptor is world_check's (food 40 mm behind the start, decoy 20 mm behind and 60 mm to the left).
check (640 x 360): 100 frames of the combined build (FOLLOW + HUD + WORLD + GAME + PROFILE, K=10): the energy bar is exactly round(engine energy row x 120)
pixels and the energy fell, the concentration text is pixel-exact for the sampled values, both mini-map sources are drawn, the camera keeps the animal
centred (spectate_check.tight), and the engine's own ms-per-frame line is reported.
"""
import os
import re
import shutil
import subprocess
import sys

import numpy as np

from agent.body_engine import game_check, spectate_check as sc, world_check as wk

BASAL, RATE = 3e-5, 3.3e-7  # per physics step: energy 1 -> 0 in ~17,000 steps (1.7 s) with the spider's ~90 work per step
FLAGS = ['-DFOLLOW=1', '-DHUD=1', '-DWORLD=1', '-DGAME=1', '-DNSITE=2']


def setup(k, big):
    """Spider dir (world_check's descriptor and sites, the demo rules at the end of mf.bin, drive speed 0.6 turn 0.8) and its binary; big = 960 x 540."""
    if big:
        sc.W, sc.H = 960, 540
    d, m, w = wk.prepare('spider')
    shutil.copy(d / 'mf.bin', d / 'mf.base')  # prepare re-exports each call: the rules ride at the end of a fresh mf.bin
    game_check.set_rules(d, BASAL, RATE)
    np.array([0.6, 0.8], np.float32).tofile(d / 'drive.bin')
    return d, sc.build(d, 'worldgame', FLAGS, k, brain=2)


def fps_line(err):
    ms = [float(x) for x in re.findall(r'([\d.]+) ms per frame', err)]
    return f'{ms[-1]:.1f} ms per frame = {1000 / ms[-1]:.1f} fps' if ms else 'no frame-rate line'


def watch(k=10):
    d, b = setup(k, True)
    mpv = subprocess.Popen(['mpv', '--demuxer=rawvideo', f'--demuxer-rawvideo-w={sc.W}', f'--demuxer-rawvideo-h={sc.H}', '--demuxer-rawvideo-mp-format=rgba',
                            '--no-cache', '-'], stdin=subprocess.PIPE)
    eng = subprocess.Popen(['bash', '-c', 'exec "$0" "$@" 4>&1 >/dev/null', str(b), str(d) + '/', '0', *sc.CAM, '1', '0'], stdout=mpv.stdin, stderr=subprocess.PIPE, text=True)
    for line in eng.stderr:  # live.mojo prints ms per frame and simulated time every 100 frames
        print(line.strip(), f'({fps_line(line)})', flush=True)
        if mpv.poll() is not None:
            break
    eng.terminate()


def fps(k=10, frames=300):
    d, b = setup(k, True)
    p = subprocess.run(['bash', '-c', 'exec "$0" "$@" 4>/dev/null >/dev/null', str(b), str(d) + '/', str(frames), *sc.CAM, '1', '0'], capture_output=True, text=True)
    if p.returncode:
        raise SystemExit(f'FAIL live exited {p.returncode}: {p.stderr[-300:]}')
    print(f'world-game K={k} {sc.W}x{sc.H}, {frames} frames: {fps_line(p.stderr)} (one spider, every overlay on)')


def check(k=10, frames=100):
    d, _ = setup(k, False)
    b = sc.build(d, 'worldgame_probe', FLAGS + ['-DPROFILE=1'], k, brain=2)
    out = wk.OUT
    p = subprocess.run(['bash', '-c', 'exec "$0" "$@" 4>"$FRAMES" >/dev/null', str(b), str(d) + '/', str(frames), *sc.CAM, '1', '0',
                        str(out / 'wg-roots.f32'), str(out / 'wg-game.f32'), str(out / 'wg-plume.f32')],
                       capture_output=True, text=True, env={**os.environ, 'FRAMES': str(out / 'wg-frames.rgba')})
    if p.returncode:
        raise SystemExit(f'FAIL live exited {p.returncode}: {p.stderr[-300:]}')
    img = np.fromfile(out / 'wg-frames.rgba', np.uint8).reshape(-1, sc.H, sc.W, 4)
    g = wk.geometry()
    last = img[-1]
    game = np.fromfile(out / 'wg-game.f32', np.float32).reshape(-1, 6)
    plume = np.fromfile(out / 'wg-plume.f32', np.float32).reshape(-1, 2, 2)
    ok = True
    y = g['BAR_Y'] + g['BAR_H'] // 2  # energy bar == the engine's own row
    row = last[y, g['BAR_X']:g['BAR_X'] + g['BAR_W']]
    fill = int((~(row[:, :3] == 0x28).all(1)).sum())
    want = int(round(float(game[-1, 0]) * g['BAR_W']))
    bars = fill == want and game[-1, 0] < game[0, 0]
    ok &= bars
    print(('PASS' if bars else 'FAIL'), f'energy bar {fill} px == round(row {game[-1, 0]:.4f} x {g["BAR_W"]}) = {want}; energy fell {game[0, 0]:.4f} -> {game[-1, 0]:.4f} in {frames} frames')
    mask = wk.text_mask(g, plume[-1])  # concentration text pixel-exact for the sampled values
    region = np.zeros((sc.H, sc.W), bool)
    region[g['TXT_Y']:g['TXT_Y'] + 2 * g['BAR_PITCH'], :g['BAR_X'] + 104] = True
    text = bool((((last[..., 3] == 254) & region) == mask).all())
    ok &= text
    print(('PASS' if text else 'FAIL'), f'concentration text pixel-exact for the sampled values food {plume[-1, 0, 0]:.3f} {plume[-1, 1, 0]:.3f}, decoy {plume[-1, 0, 1]:.3f} {plume[-1, 1, 1]:.3f}')
    crop = last[g['MAP_Y']:g['MAP_Y'] + g['MAP'], g['MAP_X']:g['MAP_X'] + g['MAP']]  # both sources in the mini-map
    seen = [int(((crop[..., :3] == c).all(-1) & (crop[..., 3] == 254)).sum()) for c in wk.COLOR.values()]
    mm = all(n >= 25 for n in seen)
    ok &= mm
    print(('PASS' if mm else 'FAIL'), f'mini-map: food {seen[0]} px, decoy {seen[1]} px (>= 25 each)')
    ok &= sc.tight(out / 'wg-frames.rgba')  # the camera keeps the spider centred
    print('engine frame-rate line (PROFILE build syncs between phases): ' + (fps_line(p.stderr) if p.stderr.strip() else 'none'))
    if not ok:
        sys.exit('FAIL watch_world')
    print('PASS watch_world: follow camera, game bars from the engine rows, plume text and mini-map in one frame')


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'check'
    {'check': check, 'watch': watch, 'fps': fps}[cmd](*(int(x) for x in sys.argv[2:3]))
