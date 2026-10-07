"""Wall seconds per sim second of play's loop (brain + body, F=1, no render), flat vs terrain: how far from real time.
usage: gpu-wait run --priority 60 --vram 4 --timeout 900 --env PYTHONPATH=$PWD -- .venv/bin/python tools/play_speed.py"""
import time
import tomllib

import numpy as np

from agent.body_engine import fly_link as fl, play

km = tomllib.loads(play.KEYMAP.read_text())
for seed in (None, 1):
    brain = play.Brain(1, km)
    d, m, eng = play.setup(1, seed=seed)
    g = [fl.Gait(m, d, fl.K)]
    w = brain.names.index('w')
    r = np.zeros((1, len(brain.names)), np.float32); r[0, w] = brain.hz[w]
    n, t = 300, {}
    t0 = time.perf_counter()
    play.loop(brain, eng, g, lambda i, game: r, n, lambda i, *a: t.setdefault('first', time.perf_counter()) if i == 0 else None)
    wall = time.perf_counter() - t.get('first', t0)
    print(f"{'flat' if seed is None else 'terrain'}: {wall / (n * fl.K * fl.DT):.1f} wall s per sim s ({wall / n * 1e3:.2f} ms per 1 ms control step)", flush=True)
    brain.brain.close(); eng.close()
