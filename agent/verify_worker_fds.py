"""Resident Shiu worker must not leak file descriptors: open fds after 50 and 400 sniffs must be equal.

Mojo 1.0 input() leaked one duplicate of the stdin pipe per call (58 -> 408 fds), which killed long
loops with "Too many open files" (the silent trial-4 crash); the worker now reads stdin with read(2).
"""
import os
import numpy as np
from agent.learning_loop import setup
from agent.verify_pain import brain
from agent.shiu import ShiuLive
pre, post, w, n, grp, kc, approach, avoid, odour = setup()
b = brain(ShiuLive, pre, post, w, n, grp, kc, odour, 4)
b.start([20], np.asarray(kc[:4]))
_, p = b._stim_arrays()
fds = []
for k in range(400):
    b.sniff(p)
    if k in (49, 399):
        fds.append(len(os.listdir(f'/proc/{b.proc.pid}/fd')))
        if k == 399:
            from collections import Counter
            print(Counter(os.path.basename(os.readlink(f'/proc/{b.proc.pid}/fd/{x}')) for x in os.listdir(f'/proc/{b.proc.pid}/fd')).most_common(4))
b.close()
ok = fds[1] == fds[0]
print("worker fds after 50 / 400 sniffs:", fds, "PASS" if ok else "FAIL leak")
raise SystemExit(0 if ok else 1)
