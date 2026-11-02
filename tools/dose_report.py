"""Score LEARNING_FIX_PLAN step 2. gap = learner valence - control valence.   python tools/dose_report.py FIRST"""
import json
import sys

import numpy as np

recs = json.load(open(f'.work/agent/shiu/specificity/dose-{sys.argv[1]}.json'))['records']
sel = lambda d: [x for x in recs if x['d'] == d]
gap = lambda r, st, o: np.array([np.mean(x[st][o]['valence'][:8]) - np.mean(x[st][o]['valence'][8:]) for x in r])
HELD = ('o2', 'o3', 'o4', 'o5')
sd = {o: gap(sel(0), 'after_A', o).std(ddof=1) for o in HELD}
print('noise sd', {k: round(float(v), 3) for k, v in sd.items()})
for d in (0.3, 0.6, 1):
    r = sel(d)
    a, b = gap(r, 'after_A', 'o1'), gap(r, 'after_B', 'o0')
    held = {o: float(np.median(np.abs(gap(r, 'after_A', o)))) for o in HELD}
    g1 = (a > 0.3).sum() >= 7
    g2 = all(held[o] < 0.5 * np.median(a) and held[o] < 3 * sd[o] + 0.1 for o in HELD)
    g3 = (b > 0.3).sum() >= 7
    print(d, 'A %.3f (%d/8)' % (np.median(a), (a > 0.3).sum()), {k: round(v, 3) for k, v in held.items()},
          'B %.3f (%d/8)' % (np.median(b), (b > 0.3).sum()), 'G1', bool(g1), 'G2', bool(g2), 'G3', bool(g3))
