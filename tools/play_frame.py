"""One rendered frame of play on the world, as the player sees it: W held for STEPS control steps, frames every 16 steps to
a file, the last one saved as PNG. usage: gpu-wait run --priority 60 --vram 4 --timeout 900 --env PYTHONPATH=$PWD --
.venv/bin/python tools/play_frame.py [SEED] [STEPS]  -> .work/agent/body_engine/play/frame-world{SEED}.png"""
import sys
import tomllib

import numpy as np
from PIL import Image

from agent.body_engine import fly_link as fl, play, spectate_check as sc

seed, steps = (int(sys.argv[1]) if len(sys.argv) > 1 else 1), (int(sys.argv[2]) if len(sys.argv) > 2 else 300)
raw = play.OUT / f'frames-world{seed}.rgba'
km = tomllib.loads(play.KEYMAP.read_text())
brain = play.Brain(1, km)
d, m, eng = play.setup(1, render=16, out=str(raw), seed=seed)
orbit = [float(x) for x in sys.argv[3:7]] or [1, -np.pi / 4, 0.35, 1]  # on, azimuth, elevation, zoom
np.array(orbit, np.float32).tofile(d / 'orbit.bin')
w = brain.names.index('w')
r = np.zeros((1, len(brain.names)), np.float32); r[0, w] = brain.hz[w]
play.loop(brain, eng, [fl.Gait(m, d, fl.K)], lambda i, game: r, steps)
brain.brain.close(); eng.close()
a = np.fromfile(raw, np.uint8)
n = a.size // (sc.W * sc.H * 4)
png = play.OUT / (f'frame-world{seed}' + ('-az' + sys.argv[4] if len(sys.argv) > 4 else '') + '.png')
Image.fromarray(a[(n - 1) * sc.W * sc.H * 4:n * sc.W * sc.H * 4].reshape(sc.H, sc.W, 4)[..., :3]).save(png)
print(f'{n} frames, last -> {png}')
