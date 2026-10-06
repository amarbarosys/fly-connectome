"""Per-eye gain calibration for agent/fly_vision.py: the male CNS left eye path runs weaker (fewer placed L3, 10-20 % fewer
synapses, LPLC2 82k vs 111k input synapses) and dies at the LC4/LPLC2 threshold. The left eye's rate gain is fitted here on
a CALIBRATION stimulus only: a 40 deg dark disc at elevation +35, az +55 (left eye) or -55 (right eye), so the disc tests
(az +-70, elevation 0) and the readout's training discs (elevation 0) never see it. Full-field dimming was tried first: LC4/LPLC2
hardly fire to it (0-0.5 spikes at every gain), a vacuous balance.

Columns: left gain G in GRID x {left disc, right disc} x K repeats, right eye gain 1. Readout: all-DN spikes per disc (LC4 +
LPLC2 per side reported), 30-170 ms. Pick: the smallest G with DN(left disc) >= DN(right disc), right >= 5 spikes. Matching
LC4/LPLC2 first gave 1.6, which overshot at the DNs (disc test 561 vs 230 spikes, side decode 0.81). Output .work/agent/shiu/eye-gain.json.
usage: gpu-wait run --shared --vram 4 --priority 60 --timeout 600 --env PYTHONPATH=$PWD -- .venv/bin/python tools/fly_eye_calibrate.py
"""
import json
import sys
from pathlib import Path

import numpy as np

from agent import fly_vision as fv, mcns, shiu_g1
from agent.eye_map import annotations
from agent.mcns_mapping import annotation_table
from agent.shiu import ShiuGPU

OUT = Path('.work/agent/shiu/eye-gain.json')
GRID, K = (1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6), 4
a = annotations()
t, s = a['type'].fillna('').to_numpy(), a['somaSide'].fillna('').str[:1].str.upper().to_numpy()
cg = np.full(len(a), -1, np.int32)
for g, sd in enumerate('LR'):
    cg[np.isin(t, ['LC4', 'LPLC2']) & (s == sd)] = g
cg[annotation_table().superclass.fillna('').str.startswith('descending').to_numpy()] = 2  # all DNs: the level the gates read
data = mcns.graph()
n = data[4][0]
ids, az, el = fv.receptors(fv.LAMINA)
left = s[ids] == 'L'
imgs = (fv.disc(55, 35), fv.disc(-55, 35))
b = ShiuGPU(*next(shiu_g1.arms(data, 'mcns'))[1], n=n, flies=len(GRID) * 2 * K, seed=53)
for c, G in enumerate(GRID):
    for e, img in enumerate(imgs):
        for r in range(K):
            col = (c * 2 + e) * K + r
            for cells, hz in fv.drive(img, ids, az, el, np.where(left, G, 1.0)):
                b.stimulate(cells, hz, fly=col, key=col)
sums, _ = b.run_groups(np.array([300, 1700]), cg)
y = np.diff(np.asarray(sums, np.int64), axis=0)[0].T.reshape(len(GRID), 2, K, 3).mean(2)  # (gain, disc, group LC-L, LC-R, DN)
lc = np.stack([y[:, 0, 0], y[:, 1, 1]], 1)  # left LC under the left disc, right LC under the right disc
x = y[:, :, 2]  # all-DN spikes: left disc, right disc
ok = (x[:, 0] >= x[:, 1]) & (x[:, 1] >= 5)
out = {'grid': GRID, 'dn_left_disc': x[:, 0].tolist(), 'dn_right_disc': x[:, 1].tolist(), 'lc_left': lc[:, 0].tolist(), 'lc_right': lc[:, 1].tolist(),
       'gain': float(GRID[int(np.argmax(ok))]) if ok.any() else None}
print(json.dumps(out))
if out['gain'] is None:
    sys.exit('FAIL eye calibration: no gain in the grid balances the eyes')
OUT.write_text(json.dumps(out, indent=1) + '\n')
print('PASS eye calibration: left gain', out['gain'])
