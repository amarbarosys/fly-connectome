"""Per-region left-eye gain for agent/fly_vision.py (FULL_FLY U1 option 1). A scalar left gain cannot balance the male CNS
eyes: the left path is weaker by an amount that depends on field position (MCNS.md Unit 4; tools/fly_eye_calibrate.py,
1.6 at its calibration disc overshoots at the test disc). Here the left eye is split into four regions (frontal/lateral x
lower/upper, split at az 72.5 and el 5 deg) and each gets its own gain, fitted at the level the loop reads: the mirrored DN
turn readout T (mcns_link steer, .work/agent/mcns/vision_readout.npz), forward DNs 80 Hz, 0.5 s.
Calibration discs (radius 25 deg) sit at elevation -30 / +40, azimuth 50 / 95, so no test sees them: the vision gates and
the readout's training discs are at elevation 0, full_fly's walls are a scene, not a disc.
Per region: left disc at gain G in GRID (left cells of that region scaled; others 1) and the mirrored right disc at gain 1,
K repeats. Pick: the smallest G with T(left disc) >= |T(right disc)|, |T(right)| >= 0.2. Output .work/agent/shiu/eye-gain-region.json.
usage: gpu-wait run --shared --vram 14 --preemptible --timeout 1800 --env PYTHONPATH=$PWD -- .venv/bin/python tools/fly_eye_calibrate_region.py
"""
import json
import sys

import numpy as np

from agent import fly_vision as fv
from agent.body_engine.mcns_link import FWD_HZ, brain
from agent.eye_map import annotations
from agent.mcns_mapping import descending_name_sets

GRID, K, N = (1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4), 4, 500
CENTRES = {'frontal-lower': (50, -30), 'lateral-lower': (95, -30), 'frontal-upper': (50, 40), 'lateral-upper': (95, 40)}

ids, az, el = fv.receptors(fv.LAMINA)
left = annotations()['somaSide'].fillna('').str[:1].str.upper().to_numpy()[ids] == 'L'
reg = fv.eye_region(az, el)
v = np.load('.work/agent/mcns/vision_readout.npz')
dn, _ = descending_name_sets()
arms = [(r, g, k) for r in CENTRES for g in (*GRID, 'right') for k in range(K)]
b, _, _, _ = brain(cols=len(arms), seed=59)
for c, (r, g, _) in enumerate(arms):
    a0, e0 = CENTRES[r]
    b.stimulate(dn['forward'], FWD_HZ, fly=c, key=c)
    img = fv.disc(-a0 if g == 'right' else a0, e0, radius=25)
    gain = np.where(left & (reg == r), 1.0 if g == 'right' else g, 1.0)
    for cells, hz in fv.drive(img, ids, az, el, gain):
        b.stimulate(cells, hz, fly=c, key=c)
b.start([1], np.concatenate([c * b.n + v['dn'] for c in range(len(arms))]).astype(np.int32))
x = b.advance(N * 10).astype(float).reshape(len(arms), -1) * (100 / N)
b.close()
T = ((x - v['mu']) / v['sd']) @ v['w'] + v['y0']
out = {}
for r in CENTRES:
    t = {g: float(np.mean([T[c] for c, a in enumerate(arms) if a[0] == r and a[1] == g])) for g in (*GRID, 'right')}
    ok = [g for g in GRID if t[g] >= abs(t['right']) and abs(t['right']) >= 0.2]
    out[r] = {'T': t, 'gain': ok[0] if ok else None}
    print(r, {k: round(val, 2) for k, val in t.items()}, '-> gain', out[r]['gain'])
bad = [r for r in out if out[r]['gain'] is None]
if bad:
    sys.exit(f'FAIL region eye calibration: no gain in the grid balances {bad}')
from pathlib import Path
Path('.work/agent/shiu/eye-gain-region.json').write_text(json.dumps(out, indent=1) + '\n')
print('PASS region eye calibration:', {r: out[r]['gain'] for r in out})
