"""CPU check of agent/eye_render.py on play_eyes_view's probe poses (world 1). Sides are the fly's own, measured by
mj_ray from the thorax: (-17.5, -pi/2) has the wall 1.0 mm to its left, (-17.5, +pi/2) 0.7 mm to its right.
Gates: wall left -> left-eye cells (az > 0) fire more than right-eye cells, wall right the other way; the mirrored
eye map flips both (mutant). Prints coverage and ms/frame.
usage: PYTHONPATH=. .venv/bin/python tools/eye_render_check.py"""
import sys
import time

import numpy as np

from agent import fly_vision as fv
from agent.body_engine import fly_link as fl, live
from agent.body_engine.play import OUT
from agent.eye_render import Eyes

m = live.load(live.terrain(fl.MJCF, live.WORLD[:5] + (1,) + live.WORLD[6:])[0])
q0 = np.fromfile(OUT / 'fly-world1/qpos.bin', np.float32).astype(float)
ids, az, el = fv.receptors(fv.LAMINA)
L = az > 0
POSES = {'left': (-17.5, -np.pi / 2), 'right': (-17.5, np.pi / 2)}

res, r = {}, None
for mir in (False, True):
    e = Eyes(m, ids, az, el, mirror=mir, renderer=r)
    r = e.r
    t = time.perf_counter()
    for k, (x, yaw) in POSES.items():
        q = q0.copy()
        q[0], q[1], q[3:7] = x, 0, (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
        hz = e.rates(q)
        res[mir, k] = hz[L].mean() - hz[~L].mean()
    print(f"mirror={mir} seen {e.seen.mean():.2f} of {len(ids)} cells; left-eye minus right-eye Hz: wall left {res[mir, 'left']:+.1f}, "
          f"wall right {res[mir, 'right']:+.1f}; {(time.perf_counter() - t) / 2 * 1e3:.1f} ms/frame")
ok = [('wall left drives the left eye more, wall right the right eye', res[False, 'left'] > 0 > res[False, 'right']),
      ('mutant mirrored map flips both', res[True, 'left'] < 0 < res[True, 'right'])]
for k, v in ok:
    print(('PASS ' if v else 'FAIL ') + k)
if not all(v for _, v in ok):
    sys.exit('FAIL eye render')
print('PASS eye render')
