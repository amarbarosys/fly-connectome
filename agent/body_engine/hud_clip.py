"""One 300-frame clip with every HUD overlay on, checked frame by frame, plus the hooks-off byte gate against main's live.mojo (live-hud brief step 2).

  gpu-wait run --shared --vram 1 --priority 60 --timeout 1800 --env PYTHONPATH=$PWD -- .venv/bin/python -m agent.body_engine.hud_clip check

The spider walks a circle under the card brain (BRAIN=2, K=10, 640 x 360) with FOLLOW + HUD + WORLD + GAME on (watch_world's build, demo rules: the energy falls
visibly). From the engine's own traces of the same frames (game rows, sampled plume, root pose) every one of the 300 frames must show: each of the three state bars
exactly round(row x 120) px wide; the concentration text pixel-equal to the glyphs of the sampled values; both plume sources' mini-map squares at hud.project_source
(descriptor, animal root of that frame) within 0.55 px in their kind colour; the animal centred (spectate_check.tight). Hooks off (no flag): the first frame of the starting
state is byte-equal to the one main's live.mojo (git show main) draws. One frame goes to hud-frame.png for a human look.
check.sh hudclip adds the mutants: the mini-map source x and y swapped (marker at the wrong place), a state bar one pixel short.
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from agent.body_engine import spectate_check as sc, watch_world as ww, world_check as wk

OUT = Path('.work/agent/body_engine/hud-clip')
PNG_FRAME = 150


def hooks_off():
    """First frame with every hook flag off == the frame main's live.mojo draws (the same engine files, K=1, spider walker)."""
    OUT.mkdir(parents=True, exist_ok=True)
    d = sc.files()
    src = OUT / 'pre' / 'live.mojo'
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(subprocess.run(['git', 'show', 'main:agent/body_engine/live.mojo'], check=True, capture_output=True, text=True).stdout)
    sc.run(sc.build(d, 'pre_main', [], 1, src), d, 1, OUT / 'main.rgba')
    sc.run(sc.build(d, 'off_main', [], 1), d, 1, OUT / 'off.rgba')
    a, b = (OUT / 'main.rgba').read_bytes(), (OUT / 'off.rgba').read_bytes()
    ok = bool(a) and a == b
    print(('PASS' if ok else 'FAIL'), f'hooks off: first frame byte-equal to main live.mojo ({len(a)} bytes)')
    return ok


def clip(frames=300):
    d, b = ww.setup(10, False)
    b = sc.build(d, 'worldgame_probe', ww.FLAGS + ['-DPROFILE=1'], 10, brain=2)
    tr = [str(OUT / f) for f in ('roots.f32', 'game.f32', 'plume.f32')]
    p = subprocess.run(['bash', '-c', 'exec "$0" "$@" 4>"$FRAMES" >/dev/null', str(b), str(d) + '/', str(frames), *sc.CAM, '1', '0', *tr],
                       capture_output=True, text=True, env={**os.environ, 'FRAMES': str(OUT / 'clip.rgba')})
    if p.returncode:
        raise SystemExit(f'FAIL live exited {p.returncode}: {p.stderr[-300:]}')
    img = np.memmap(OUT / 'clip.rgba', np.uint8, 'r').reshape(-1, sc.H, sc.W, 4)
    game = np.fromfile(OUT / 'game.f32', np.float32).reshape(-1, 6)
    plume = np.fromfile(OUT / 'plume.f32', np.float32).reshape(-1, 2, 2)
    roots = np.fromfile(OUT / 'roots.f32', np.float32).reshape(-1, 3)
    g = wk.geometry()
    _, _, w = wk.prepare('spider')
    region = np.zeros((sc.H, sc.W), bool)
    region[g['TXT_Y']:g['TXT_Y'] + 2 * g['BAR_PITCH'], :g['BAR_X'] + 104] = True
    bad = {'bars': [], 'text': [], 'map': []}
    for i in range(frames):
        f = img[i]
        for k in range(3):
            y = g['BAR_Y'] + g['BAR_PITCH'] * k + g['BAR_H'] // 2
            fill = int((~(f[y, g['BAR_X']:g['BAR_X'] + g['BAR_W'], :3] == 0x28).all(1)).sum())
            if fill != int(round(float(game[i, k]) * g['BAR_W'])):
                bad['bars'].append((i, k))
        if not ((((f[..., 3] == 254) & region) == wk.text_mask(g, plume[i])).all()):
            bad['text'].append(i)
        crop = f[g['MAP_Y']:g['MAP_Y'] + g['MAP'], g['MAP_X']:g['MAP_X'] + g['MAP']]
        for s in w.sources:
            cx = g['MAP'] / 2 + (s.x_mm - roots[i, 0]) * g['MAP_SCALE']
            cy = g['MAP'] / 2 - (s.y_mm - roots[i, 1]) * g['MAP_SCALE']
            ys, xs = np.nonzero((crop[..., :3] == wk.COLOR[s.kind]).all(-1) & (crop[..., 3] == 254))
            if len(xs) < 25 or abs(xs.mean() + 0.5 - cx) > 0.55 or abs(ys.mean() + 0.5 - cy) > 0.55:
                bad['map'].append((i, s.kind))
    ok = not any(bad.values())
    for name, got in bad.items():
        print(('PASS' if not got else 'FAIL'), f'{name}: {frames - len({x[0] if isinstance(x, tuple) else x for x in got})} of {frames} frames exact' + (f'; first bad {got[:3]}' if got else ''))
    print(f'state over the clip: energy {game[0, 0]:.4f} -> {game[-1, 0]:.4f}, stamina {game[0, 2]:.4f} -> {game[-1, 2]:.4f}; food concentration L {plume[0, 0, 0]:.3f} -> {plume[-1, 0, 0]:.3f}; animal travelled '
          f'{float(np.linalg.norm(roots[-1, :2] - roots[0, :2])):.1f} mm')
    Image.fromarray(np.array(img[PNG_FRAME])).save(OUT / 'hud-frame.png')
    print(f'frame {PNG_FRAME} -> {OUT / "hud-frame.png"}')
    ok &= sc.tight(OUT / 'clip.rgba')
    return ok


def main():
    ok = hooks_off()
    ok &= clip()
    if not ok:
        sys.exit('FAIL hud_clip')
    print('PASS hud_clip: 300 frames, every overlay exact per frame, hooks off byte-equal to main')


if __name__ == '__main__':
    main()
