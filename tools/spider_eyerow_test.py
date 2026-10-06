"""-DEYEROW=1 (live.mojo + eyes.eye_rowmin) against the full retina path: the same scene, gaze and ctrl through Eyes(row=False)
and Eyes(row=True); AM_L's darkest pixel per column must match byte for byte at every step; per-step bytes and wall time reported.
usage: gpu-wait run --shared --vram 4 --priority 60 --timeout 900 --env PYTHONPATH=$PWD -- .venv/bin/python tools/spider_eyerow_test.py
"""
import sys
import time

import numpy as np

from agent import spider_eye_loop as L

N = 60
tgt0, sp = np.linspace(-1, 1, L.G.B), np.full(L.G.B, 8.0)
gaze = np.random.default_rng(3).normal(0, 0.5, (N, L.G.B))
out = {}
for row in (False, True):
    e = L.Eyes(tgt0, sp, N, row=row)
    t0, got = time.perf_counter(), []
    for k in range(N):
        got.append(e.sense(gaze[k]).copy())
    out[row] = (np.array(got), (time.perf_counter() - t0) / N)
    e.close()
full, rowm = out[False][0], out[True][0]
same = bool((full == rowm).all())
print(f"rows equal at all {N} steps x {L.G.B} animals: {same}; max |diff| {np.abs(full - rowm).max():.0f}; "
      f"s/step full {out[False][1]:.4f} row {out[True][1]:.4f} ({out[False][1] / out[True][1]:.1f}x); bytes/step {L.G.B * 6 * L.H * L.W} -> {L.G.B * L.W}")
print(('PASS' if same else 'FAIL') + ' eyerow: row mode equals the full retina path')
sys.exit(0 if same else 1)
