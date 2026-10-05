"""Video of the spider closed loop (agent/spider_eye_loop.py): what AM_L sees, every 10 ms, for the same prey trial driven
by the trained pathway (left panel) and by the flipped-teacher mutant (right panel). Crosshair = gaze centre; the prey is
the dark dot. 1 s of simulated time (100 steps), shown 10x slower. Seed 2, prey speed 8 deg/s.
usage: gpu-wait run --shared --vram 4 --priority 60 --timeout 2400 --env PYTHONPATH=$PWD -- .venv/bin/python -m agent.spider_eye_video
  -> .work/agent/spider/eye-loop.mp4
"""
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from agent import spider_eye_loop as L

OUT = Path('.work/agent/spider/eye-video')
STEPS, SPEED, TGT0 = 100, 8.0, 0.25


def trial(net, rec):
    """one batch (B spiders, all the same trial); returns AM_L images of spider 0 and its gaze error per step."""
    reset, step = net
    reset()
    sp, tgt0 = np.full(L.G.B, SPEED), np.full(L.G.B, TGT0)
    eyes = L.Eyes(tgt0, sp, STEPS)
    gaze, eye_v, prev, frames, errs = np.zeros(L.G.B), np.zeros(L.G.B), None, [], []
    for k in range(STEPS):
        eyes.p.stdin.write(np.ascontiguousarray(eyes.ctrl).tobytes() + np.asarray(gaze, np.float32).tobytes())
        eyes.p.stdin.flush()
        eyes.p.stdout.read(4 * (L.NU + eyes.nq + eyes.nv + eyes.ns) * L.G.B)
        ret = np.frombuffer(eyes.p.stdout.read(L.G.B * L.E.NE * L.H * L.W), np.uint8).reshape(L.G.B, L.E.NE, L.H, L.W)
        frames.append(ret[0, 0].copy())
        now = L.receptors(rec, ret[:, 0].min(1).astype(np.float32))
        d = step(L.P.motion(prev, now).astype(np.float32)) if prev is not None else np.zeros(L.G.B)
        prev = now
        eye_v = eye_v + L.P.KA * d
        gaze = gaze + eye_v * L.P.DT
        errs.append(float(abs(tgt0[0] + sp[0] * (k + 1) * L.P.DT - gaze[0])))
    eyes.close()
    return frames, errs


def panel(img, title, err, k):
    im = Image.fromarray(img).convert('RGB').resize((L.W * 2, L.H * 2), Image.NEAREST)
    d = ImageDraw.Draw(im)
    cx, cy = L.W, L.H
    d.line([(cx - 14, cy), (cx + 14, cy)], fill=(230, 40, 40), width=1)
    d.line([(cx, cy - 14), (cx, cy + 14)], fill=(230, 40, 40), width=1)
    d.text((6, 4), title, fill=(255, 255, 255))
    d.text((6, L.H * 2 - 14), f't {k * 10:4d} ms   gaze error {err:.2f} deg', fill=(255, 255, 0))
    return im


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rec = L.P.retina(np.random.default_rng(0))
    runs = {}
    for arm, title in (('learn', 'trained brain'), ('flip', 'mutant: flipped teacher')):
        _, net = L.G.train(arm, 300, rec, 2, keep=True)
        runs[title] = trial(L.pathway(net), rec)
        net.close()
    names = list(runs)
    for k in range(STEPS):
        a, b = (panel(runs[n][0][k], n, runs[n][1][k], k) for n in names)
        frame = Image.new('RGB', (a.width * 2 + 8, a.height), (20, 20, 20))
        frame.paste(a, (0, 0)); frame.paste(b, (a.width + 8, 0))
        frame.save(OUT / f'{k:04d}.png')
    mp4 = OUT.parent / 'eye-loop.mp4'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', '10', '-i', str(OUT / '%04d.png'), '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
                    '-pix_fmt', 'yuv420p', str(mp4)], check=True)
    for n in names:
        print(f'{n}: gaze error at 0.2 / 0.5 / 1.0 s = {runs[n][1][19]:.2f} / {runs[n][1][49]:.2f} / {runs[n][1][99]:.2f} deg')
    print('written', mp4)


if __name__ == '__main__':
    main()
