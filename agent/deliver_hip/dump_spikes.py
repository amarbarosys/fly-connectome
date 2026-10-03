"""Spike lists per step from the CPU oracle, for deliver_bench.hip.

Learning-loop graph (agent/learning_loop.setup, both plastic planes at 1.0), one column, both odours
at HZ, STEPS steps. Writes .work/agent/shiu/hipbench/: spikes.bin (int32 neuron ids, all steps in
order), offsets.bin (int32 STEPS + 1 prefix offsets) and graph.txt (the CSR directory).
"""
from pathlib import Path

import numpy as np

from agent.learning_loop import setup
from agent.neural_connectome import ROOT
from agent.shiu import ShiuBrain
from agent.verify_pain import brain

STEPS = 600
out = ROOT / '.work/agent/shiu/hipbench'
out.mkdir(parents=True, exist_ok=True)
pre, post, w, n, grp, kc, approach, avoid, odour = setup()
b = brain(ShiuBrain, pre, post, w, n, grp, kc, odour, 1)
spikes, offsets = [], [0]
for _ in range(STEPS):
    b.step()
    s = np.flatnonzero(b.last == b.t - 1).astype(np.int32)
    spikes.append(s)
    offsets.append(offsets[-1] + len(s))
np.concatenate(spikes).tofile(out / 'spikes.bin')
np.int32(offsets).tofile(out / 'offsets.bin')
(out / 'graph.txt').write_text(str(b.graph) + '\n')
assert offsets[-1] == int(b.counts.sum()), (offsets[-1], int(b.counts.sum()))
per = np.diff(offsets)
print(f'{offsets[-1]} spikes over {STEPS} steps (max {per.max()} per step), out-edges per step mean '
      f'{np.mean([int((b.row[s + 1] - b.row[s]).sum()) for s in spikes]):.0f} -> {out}')
