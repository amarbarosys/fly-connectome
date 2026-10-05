"""Plume-world gates for the live view (-DWORLD=1; docs/plans/WORLD.md): the odor sites sample the plume, the HUD prints the sampled
concentration and draws the sources, all from one descriptor.

  python -m agent.body_engine.world_check build [spider|fly]
  gpu-wait run --priority 60 --timeout 1800 -- .venv/bin/python -m agent.body_engine.world_check check [spider|fly]

One descriptor per run (world_content.build(SEED), translated so the food source lies 40 mm behind the animal's start and the decoy placed 20 mm
behind and 60 mm to the left, both inside the 160 mm mini-map; the animal walks away, so the numbers fall) is written once as src.bin; the engine sampler (spectate.sense, on the card, from the engine's own
pose rows) and the mini-map read it, the python oracle reads the same PlumeWorld. Sensing sites: fly = the MJCF's own sites
fly/l_odor, fly/r_odor (stands, brain 0: the fly's gait is not merged); spider = the leg I tarsus tips (l1_tarsus, r1_tarsus at local
+0.6 mm x: the generated model has no palps and no sites), walking under the merged brain (BRAIN=2).
  (1) sense: per site, food and decoy of the engine's last frame == world_content.concentration_xy_mm at the site's MuJoCo kinematics
      position (engine's final qpos), abs err <= 2e-5
  (2) text: the 'd.ddd' text under the bars is exactly the glyphs python draws from the oracle value, and nothing else is drawn there
  (3) map: every source square sits at hud.project_source(descriptor) (centroid within 0.55 px: the squares sit on whole pixels) in its kind colour; the animal at the centre
check.sh world_hud adds mutants: food/decoy swapped in sense, the presence flag ignored, mini-map y flipped.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

from agent.body_engine import live, spectate_check as sc, world_content as wc
from agent.body_engine.models.spider import build as spider_build
from agent.body_engine.render_check import MJCF

OUT = Path('.work/agent/body_engine/world-hud')
W, H = sc.W, sc.H
SEED, BEHIND_MM, FRAMES = 3, 40.0, 60
FONT = dict(zip('0123456789.FDLR', (0x7B6F, 0x2C97, 0x73E7, 0x73CF, 0x5BC9, 0x79CF, 0x79EF, 0x7249, 0x7BEF, 0x7BCF, 0x0002, 0x79A4, 0x6B6E, 0x4927, 0x6B75)))
SPIDER_SITES = (('l1_tarsus', 0.6), ('r1_tarsus', 0.6))  # body, local +x (mm): leg I tarsus tip
COLOR = {0: (255, 190, 64), 1: (96, 160, 255)}  # world_content colours, food and decoy


def geometry():
    src = (sc.ENGINE / 'spectate.mojo').read_text()
    g = sc.hud_geometry()
    g['MAP'] = int(re.search(r'comptime MAP = (\d+)', src).group(1))
    g['MAP_SCALE'] = int(re.search(r'MAP_SCALE_E3", (\d+)', src).group(1)) * 0.001
    g['PANEL_H'] = g['BAR_Y'] + 3 * g['BAR_PITCH']
    g['TXT_Y'] = g['PANEL_H'] + 4
    g['MAP_X'], g['MAP_Y'] = W - g['MAP'] - 8, 8
    return g


def sites_of(m):
    """(body id, local xyz mm) per odor site; fly: the model's own sites, spider: the leg I tarsus tips."""
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(m.nsite)]
    if 'fly/l_odor' in names:
        return [(int(m.site_bodyid[names.index(n)]), m.site_pos[names.index(n)].astype(float)) for n in ('fly/l_odor', 'fly/r_odor')]
    return [(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b), np.array([x, 0.0, 0.0])) for b, x in SPIDER_SITES]


def prepare(name, absent=False):
    d = OUT / name
    cam, m = live.prepare(d, mjcf=str(spider_build()) if name == 'spider' else MJCF, dist=15.0 if name == 'spider' else live.DIST)
    q0 = np.fromfile(d / 'qpos.bin', np.float32)
    np.array([1, 2.0 if name == 'fly' else 8.0], np.float32).tofile(d / 'roots.bin')
    np.array([[b, *p] for b, p in sites_of(m)], np.float32).tofile(d / 'sites.bin')
    np.array([1.0, 0.0], np.float32).tofile(d / 'drive.bin')
    np.array([1, 1, 1, 0], np.float32).tofile(d / 'game.bin')
    w = wc.build(SEED)
    f = w.sources[0]
    dx, dy = float(q0[0]) - BEHIND_MM - f.x_mm, float(q0[1]) - f.y_mm  # the food source 40 mm behind the start: descriptor translated once
    shifted = [wc.Source(s.source_id, s.x_mm + dx, s.y_mm + dy, s.kind, s.color_rgb) for s in w.sources]
    shifted[1] = wc.Source(shifted[1].source_id, float(q0[0]) - 20.0, float(q0[1]) + 60.0, shifted[1].kind, shifted[1].color_rgb)  # the decoy: seed 3 puts it 686 mm away, off the map
    w = wc.PlumeWorld(w.seed, tuple(shifted), wc.ResourceState(w.state.remaining, (True, not absent)))  # absent: the decoy emits nothing and is drawn grey
    np.array([[s.x_mm, s.y_mm, s.kind, float(w.state.present[i]), w.state.remaining[i]] for i, s in enumerate(w.sources)], np.float32).tofile(d / 'src.bin')
    return d, m, w


def binary(d, name):
    return sc.build(d, 'world', ['-DFOLLOW=1', '-DHUD=1', '-DPROFILE=1', '-DWORLD=1', '-DNSITE=2'], 10, brain=2 if name == 'spider' else 0)


def run(d, name, frames=FRAMES):
    """The engine for `frames` control steps; returns (last frame RGBA (H, W, 4), final qpos, root trace (frames, 3), plume trace (frames, 2 sites x 2))."""
    b = binary(d, name)
    traces = [str(OUT / f) for f in ('roots.f32', 'game.f32', 'plume.f32')]
    if name == 'spider':  # frames on fd 4, one sensor packet on stdout at the last step
        p = subprocess.run(['bash', '-c', 'exec "$0" "$@" 4>"$FRAMES"', str(b), str(d) + '/', str(frames), *sc.CAM, '1', '0', *traces],
                           capture_output=True, env={**os.environ, 'FRAMES': str(OUT / 'frames.rgba')})
        nq = int(re.search(r'-DNQ=(\d+)', (d / 'sizes.txt').read_text()).group(1))
        nu = int(re.search(r'-DNU=(\d+)', (d / 'sizes.txt').read_text()).group(1))
        q = np.frombuffer(p.stdout, np.float32)[nu:nu + nq]
        img = np.fromfile(OUT / 'frames.rgba', np.uint8).reshape(-1, H, W, 4)[-1]
    else:  # to_file: frame FRAMES-1 and its qpos
        p = subprocess.run([str(b), str(d) + '/', str(frames), *sc.CAM, str(OUT / 'last.rgba'), '0', *traces], capture_output=True)
        q = np.fromfile(str(OUT / 'last.rgba') + '.qpos', np.float32)
        img = np.fromfile(OUT / 'last.rgba', np.uint8).reshape(H, W, 4)
    if p.returncode:
        raise SystemExit(f'FAIL live exited {p.returncode}: {p.stderr[-300:].decode()}')
    roots = np.fromfile(OUT / 'roots.f32', np.float32).reshape(-1, 3)
    plume = np.fromfile(OUT / 'plume.f32', np.float32).reshape(-1, 2, 2)  # frame, site, (food, decoy)
    return img, q, roots, plume


def site_xy(m, q, d):
    s = mujoco.MjData(m)
    s.qpos[:] = q
    mujoco.mj_kinematics(m, s)
    return np.array([(s.xpos[b] + s.xmat[b].reshape(3, 3) @ p)[:2] for b, p in sites_of(m)])


def text_mask(g, values):
    """Expected (H, W) mask of the concentration text: rows F (food) and D (decoy), labels and 'd.ddd' per site, 2x glyphs."""
    mask = np.zeros((H, W), bool)

    def draw(ch, x0, y0):
        for gy in range(5):
            for gx in range(3):
                if (FONT[ch] >> (14 - (3 * gy + gx))) & 1:
                    mask[y0 + 2 * gy:y0 + 2 * gy + 2, x0 + 2 * gx:x0 + 2 * gx + 2] = True

    for line, label in enumerate('FD'):
        y0 = g['TXT_Y'] + g['BAR_PITCH'] * line
        draw(label, g['BAR_X'] - 12, y0)
        for site in range(values.shape[0]):
            n = int(np.clip(values[site, line], 0, 9.999) * 1000 + 0.5)
            for c, ch in enumerate(f'{n // 1000 % 10}.{n // 100 % 10}{n // 10 % 10}{n % 10}'):
                draw(ch, g['BAR_X'] + 48 * site + 8 * c, y0)
    return mask


def scenario(name, absent):
    OUT.mkdir(parents=True, exist_ok=True)
    d, m, w = prepare(name, absent)
    tag = f'{name}{" decoy absent" if absent else ""}'
    g = geometry()
    img, q, roots, plume = run(d, name)
    ok = True
    # (1) sense
    want = wc.concentration_xy_mm(site_xy(m, q, d), w)
    err = float(np.abs(plume[-1] - want).max())
    moved = float(np.abs(plume[-1] - plume[0]).max())
    one = err <= 2e-5 and (want > 0).any() and (not absent or bool((plume[:, :, 1] == 0).all()))  # absent: the decoy's concentration is exactly 0 in every frame
    ok &= one
    print(('PASS' if one else 'FAIL'), f'{tag} (1) sense: engine concentration at {len(want)} sites == world_content at MuJoCo site positions, max abs err {err:.2e} '
          f'(bar 2e-5); food L/R {plume[-1, 0, 0]:.4f} {plume[-1, 1, 0]:.4f}, decoy {plume[-1, 0, 1]:.4f} {plume[-1, 1, 1]:.4f}; change over {FRAMES} frames {moved:.4f}')
    # (2) text
    mask = text_mask(g, want)
    hud = img[..., 3] == 254
    region = np.zeros((H, W), bool)
    region[g['TXT_Y']:g['TXT_Y'] + 2 * g['BAR_PITCH'], :g['BAR_X'] + 96 + 8] = True
    shown = hud & region
    two = bool((shown == mask).all() and (img[mask][:, :3] == 255).all())
    ok &= two
    print(('PASS' if two else 'FAIL'), f'{tag} (2) text: {int(mask.sum())} glyph pixels expected, {int(shown.sum())} drawn, equal={bool((shown == mask).all())}')
    # (3) map: each source square at project_source, in its kind colour
    tx, ty = float(roots[-1, 0]), float(roots[-1, 1])
    three = True
    for i, s in enumerate(w.sources):
        cx = g['MAP'] / 2 + (s.x_mm - tx) * g['MAP_SCALE']  # hud.project_source: centre + (source - target) x scale, y flipped
        cy = g['MAP'] / 2 - (s.y_mm - ty) * g['MAP_SCALE']
        inside = 4 <= cx <= g['MAP'] - 4 and 4 <= cy <= g['MAP'] - 4
        crop = img[g['MAP_Y']:g['MAP_Y'] + g['MAP'], g['MAP_X']:g['MAP_X'] + g['MAP']]
        hit = (crop[..., :3] == ((128, 128, 128) if not w.state.present[i] else COLOR[s.kind])).all(-1) & (crop[..., 3] == 254)
        ys, xs = np.nonzero(hit)
        if not inside:
            print(f'NOTE {tag} source {i} (kind {s.kind}) at map ({cx:.1f}, {cy:.1f}) is outside the {g["MAP"]} px map')
            three &= len(xs) == 0
            continue
        got = (xs.mean() + 0.5, ys.mean() + 0.5) if len(xs) else (np.nan, np.nan)
        good = len(xs) >= 25 and abs(got[0] - cx) <= 0.55 and abs(got[1] - cy) <= 0.55
        three &= bool(good)
        print(('PASS' if good else 'FAIL'), f'{tag} (3) map source {i} kind {s.kind}: expected ({cx:.2f}, {cy:.2f}) got ({got[0]:.2f}, {got[1]:.2f}), {len(xs)} px')
    ok &= three
    return bool(ok)


def check(name):
    return scenario(name, False) and scenario(name, True)


def main(cmd, *names):
    names = names or ('spider', 'fly')
    OUT.mkdir(parents=True, exist_ok=True)
    if cmd == 'build':
        for n in names:
            binary(prepare(n)[0], n)
        return
    res = [check(n) for n in names]
    if not all(res):
        sys.exit('FAIL world: ' + ' '.join(n for n, r in zip(names, res) if not r))
    print('PASS world: ' + ' '.join(names))


if __name__ == '__main__':
    main(*(sys.argv[1:] or ['check']))
