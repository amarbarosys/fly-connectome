"""Per-kernel time of resident sniffs at the learning loop's size, under rocprofv3.

Same graph, groups, plastic planes and stimulation as agent/pain_loop.py (96 columns, 100 ms window,
MBON readout); one warm-up sniff, then SNIFFS timed sniffs of both odours.

usage: gpu-wait run ... -- rocprofv3 --kernel-trace --stats -d .work/agent/shiu/profile -o sniff \
           -- .venv/bin/python -m agent.profile_sniff
"""
import time

import numpy as np

from agent.learning_loop import WINDOW, setup
from agent.shiu import ShiuLive
from agent.verify_pain import brain

FLIES, SNIFFS = 96, 3

pre, post, w, n, grp, kc, approach, avoid, odour = setup()
live = brain(ShiuLive, pre, post, w, n, grp, kc, odour, FLIES)
mbon = np.concatenate([approach, avoid])
live.start([WINDOW], (np.arange(FLIES)[None] * n + mbon[:, None]).reshape(-1))
_, p = live._stim_arrays()
live.sniff(p)
t0 = time.perf_counter()
for _ in range(SNIFFS):
    live.sniff(p)
print(f'{FLIES} columns, {WINDOW} steps: {(time.perf_counter() - t0) / SNIFFS * 1000:.0f} ms per sniff', flush=True)
live.close()
