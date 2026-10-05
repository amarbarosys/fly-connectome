"""Falsifier before mcns unit 3 (learned MN -> actuator layer): does the MCNS Shiu model give the 381 leg MNs any
rhythm at stepping frequency under forward-DN drive? A linear readout of stationary rates cannot reproduce the gait's
periodic targets, so no rhythm = the layer cannot learn the gait (R^2 bar unreachable by construction).

Drive: the forward DN set (agent.mcns_mapping, DNg/oDN1 types) at RATE Hz Poisson, 1 s, counts per 2 ms.
Reports active MNs (> 1 Hz), and per active MN the fraction of rate-fluctuation power in 5-20 Hz (stepping band;
FlyGym's tripod is ~12 Hz) vs a Poisson-flat expectation (band share of the 0-250 Hz spectrum = 15/250 = 0.06).

usage: gpu-wait run --shared --vram 2 --priority 60 --timeout 900 -- .venv/bin/python tools/mcns_mn_rhythm.py [RATE]
"""
import json
import sys

import numpy as np

from agent import mcns, shiu_g1
from agent.mcns_mapping import descending_name_sets, leg_motor_groups
from agent.shiu import ShiuGPU

rate = float(sys.argv[1]) if len(sys.argv) > 1 else 150.0
data = mcns.graph()
graph = next(shiu_g1.arms(data, 'mcns'))[1]
n = data[4][0]
dn, _ = descending_name_sets()
mn_groups, _ = leg_motor_groups()
mn = np.concatenate(list(mn_groups.values()))
brain = ShiuGPU(*graph, n=n, flies=2, seed=7)  # 2 columns: the readout buffer is N x columns (381 x 500 bins > N)
brain.stimulate(dn['forward'], rate)
win = np.arange(20, 10001, 20)  # 2 ms bins, 1 s
cgrp = np.full(n, -1, np.int32); cgrp[mn] = 0
_, got = brain.run_groups(win, cgrp, mn)
c = np.diff(np.vstack([np.zeros(len(mn), np.int64), got.astype(np.int64)]), axis=0)  # spikes per bin (500, 381)
hz = c.sum(0) / 1.0
act = hz > 1
f = np.fft.rfftfreq(len(win), 2e-3)
band = (f >= 5) & (f <= 20)
share = []
for j in np.flatnonzero(act):
    x = c[100:, j] - c[100:, j].mean()  # skip 200 ms onset
    p = np.abs(np.fft.rfft(x, len(win))) ** 2
    share.append(p[band].sum() / max(p[1:].sum(), 1e-9))
share = np.array(share)
out = {'rate_hz': rate, 'forward_dn': int(len(dn['forward'])), 'mn': int(len(mn)), 'active_mn': int(act.sum()),
       'mn_rate_median_active': float(np.median(hz[act])) if act.any() else 0.0,
       'band_share_median': float(np.median(share)) if len(share) else 0.0,
       'band_share_p90': float(np.percentile(share, 90)) if len(share) else 0.0,
       'flat_expectation': float(band.sum() / (len(f) - 1)),
       'rhythmic_mn (share > 3x flat)': int((share > 3 * band.sum() / (len(f) - 1)).sum())}
print(json.dumps(out))
np.save('.work/mcns/mn_rhythm_counts.npy', c)
