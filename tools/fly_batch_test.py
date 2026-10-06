"""Batched fly engine (fly_link.Engine F > 1) against F single-fly engines: the same ctrl streams (Gait, a different drive per
fly) for N steps; qpos must match to 1e-5 at every step. Then s/step and bytes/step at F 1, 32, 128 (one process each).
usage: gpu-wait run --shared --vram 4 --priority 60 --timeout 1200 --env PYTHONPATH=$PWD -- .venv/bin/python tools/fly_batch_test.py
"""
import sys
import time

import numpy as np

from agent.body_engine.fly_link import K, Engine, Gait, prepare

N, F = 200, 4
d, m, _ = prepare()
drives = [(0.6, 0.6), (0.9, 0.3), (0.3, 0.9), (0.0, 0.0)]
gaits = [Gait(m, d, K) for _ in range(F)]
ctrl = np.array([[g.ctrl(dr) for g, dr in zip(gaits, drives)] for _ in range(N)])  # (N, F, NU)
one = [Engine(d, K, f'one{i}', N) for i in range(F)]
q1 = []
for c in ctrl:
    for e, u in zip(one, c):
        e.send(u)
    q1.append([e.recv() for e in one])
for e in one:
    e.close()
bat = Engine(d, K, 'batch', N, F=F)
qb = []
for c in ctrl:
    bat.send(c)
    qb.append(bat.recv())
bat.close()
diff = float(np.abs(np.array(q1) - np.array(qb)).max())
print(f'batch F={F} vs {F} single engines, {N} steps: qpos max |diff| {diff:.2e} (bar 1e-5); root xy end {np.array(qb)[-1, :, :2].round(2).tolist()}')
for f in (1, 32, 128):
    e = Engine(d, K, f'time{f}', N, F=f)
    u = np.tile(ctrl[0, 0], (f, 1))
    t0 = time.perf_counter()
    for _ in range(N):
        e.send(u)
        e.recv()
    dt = (time.perf_counter() - t0) / N
    e.close()
    print(f'F {f}: {dt:.4f} s/step, {dt / f * 1e3:.3f} ms per fly-step, bytes/step {e.size + 4 * e.nu * f}')
ok = diff <= 1e-5
print(('PASS' if ok else 'FAIL') + ' batched fly engine equals single-fly engines')
sys.exit(0 if ok else 1)
