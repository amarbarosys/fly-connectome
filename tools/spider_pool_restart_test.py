"""Restarted spider_pool binary continues the random streams (growth fix B): a run stopped after one step and restarted
with GPUPathway(t=...) gives the same motor spikes as one continuous run; the mutant (restart from 0) replays step 1.
usage: gpu-wait run --shared --vram 1 --timeout 600 --env SLICE=1 --env PYTHONPATH=$PWD -- .venv/bin/python tools/spider_pool_restart_test.py
"""
import numpy as np, agent.spider_pool_gpu as S
idx, w, fb, ch = S.init(); w[:] = np.random.default_rng(4).normal(0, 0.15, S.NS).astype(np.float32)
m = np.random.default_rng(1).normal(0, 0.02, (S.B, 12)).astype(np.float32)
a = S.GPUPathway(idx, w, fb, ch); a.reset(); x1 = a.step(m); a.reset(); x2 = a.step(m); a.close()
b = S.GPUPathway(idx, w, fb, ch); b.reset(); y1 = b.step(m); t = b.t; b.close()
c = S.GPUPathway(idx, w, fb, ch, t=t); c.reset(); y2 = c.step(m); c.close()
z = S.GPUPathway(idx, w, fb, ch); z.reset(); z2 = z.step(m); z.close()  # restart WITHOUT the counter: replays step 1
print(('PASS' if (x2 == y2).all() else 'FAIL'), 'restart with counters = continuous run; motor spikes', int(x2.sum()), int(y2.sum()))
print(('PASS' if (z2 == x1).all() and not (x2 == x1).all() else 'FAIL'), 'mutant: restart from 0 replays step 1 (the bug B fixes)')
raise SystemExit(0 if (x2 == y2).all() and (z2 == x1).all() and not (x2 == x1).all() else "FAIL restart")
