"""Gate for agent/fly_vision.py: a dark disc seen by the left vs the right eye (pixels -> R1-6 -> MCNS Shiu brain) must
be told apart in the descending neurons, a blank image must drive nothing, and a mirrored eye map must swap the sides.

Columns (one launch, K repeats each): disc at az +70 (left eye), disc at az -70 (right), blank, disc at +70 seen through the
mirrored map (mutant). DN counts 30-200 ms. Gates: >= 5 DNs with |z| > 4 left vs right; blank DN spikes <= 1 % of the
left-disc ones; leave-one-out nearest-centroid side decode >= 0.9; mirrored-left trials decode as right in >= 80 %.
usage: gpu-wait run --shared --vram 4 --priority 60 --timeout 900 --env PYTHONPATH=$PWD -- .venv/bin/python tools/fly_vision_test.py
"""
import json
import sys

import numpy as np

from agent import fly_vision as fv, mcns, shiu_g1
from agent.mcns_mapping import annotation_table
from agent.shiu import ShiuGPU
from agent.visual_side import decode

K = 8
ids, az, el = fv.receptors()
a = annotation_table()
dn = np.flatnonzero(a.superclass.fillna('').str.startswith('descending').to_numpy())
data = mcns.graph()
n = data[4][0]
cond = [('left', fv.drive(fv.disc(70), ids, az, el)), ('right', fv.drive(fv.disc(-70), ids, az, el)),
        ('blank', fv.drive(np.full((90, 180), fv.BG), ids, az, el)), ('mirrored', fv.drive(fv.disc(70), ids, -az, el))]
brain = ShiuGPU(*next(shiu_g1.arms(data, 'mcns'))[1], n=n, flies=len(cond) * K, seed=23)
for c, (_, groups) in enumerate(cond):
    for r in range(K):
        for cells, hz in groups:
            brain.stimulate(cells, hz, fly=c * K + r, key=c * K + r)
cg = np.full(n, -1, np.int32); cg[dn] = 0
_, got = brain.run_groups(np.array([300, 2000]), cg, np.concatenate([f * n + dn for f in range(len(cond) * K)]).astype(np.int32))
x = np.diff(got.astype(np.int64), axis=0)[0].reshape(len(cond), K, len(dn)).astype(float)  # (cond, repeat, dn)
L, R, B, M = x
z = (L.mean(0) - R.mean(0)) / np.sqrt(L.var(0, ddof=1) / K + R.var(0, ddof=1) / K + 1 / K)
v = np.concatenate([L, R]).T
side_dec = decode(v, np.repeat([0, 1], K))
u = lambda q: q / (np.linalg.norm(q) + 1e-9)
cL, cR = u(L.mean(0)), u(R.mean(0))
mut_right = float(np.mean([u(m) @ cR > u(m) @ cL for m in M]))
out = {'receptors': int(len(ids)), 'dns': int(len(dn)), 'dn_spikes': {k: float(q.sum(1).mean()) for k, q in zip(('left', 'right', 'blank', 'mirrored'), x)},
       'dns_z4': int((np.abs(z) > 4).sum()), 'max_abs_z': float(np.abs(z).max()), 'side_decode': side_dec, 'mirrored_decoded_right': mut_right}
print(json.dumps(out))
ok = [('>= 5 DNs |z| > 4, left vs right disc', out['dns_z4'] >= 5),
      ('blank drives <= 1 % of the left-disc DN spikes', out['dn_spikes']['blank'] <= 0.01 * out['dn_spikes']['left']),
      ('side decode >= 0.9', side_dec >= 0.9),
      ('mutant mirrored map: left disc decodes as right >= 80 %', mut_right >= 0.8)]
for k, v_ in ok:
    print(('PASS ' if v_ else 'FAIL ') + k)
if not all(v_ for _, v_ in ok):
    sys.exit('FAIL fly vision from pixels')
print('PASS fly vision from pixels')
