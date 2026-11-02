"""Score prereg in LEARNING_FIX_PLAN.md: elig receipts, gap = control valence - learner valence, per odour.
    python tools/elig_report.py FIRST
"""
import json
import sys

import numpy as np

first = sys.argv[1]
recs = json.load(open(f'.work/agent/shiu/specificity/elig-{first}.json'))['records']
ok = True
for win in ([500], [500, 2000]):
    r = [x for x in recs if x['windows'] == win]
    gap = lambda o: np.array([np.mean(x['after_A'][o]['valence'][8:]) - np.mean(x['after_A'][o]['valence'][:8]) for x in r])
    a = gap('o1')
    held = {o: float(np.median(np.abs(gap(o)))) for o in ('o2', 'o3', 'o4', 'o5')}
    g1, g2 = int((a > 0.3).sum()) >= 7, all(v < min(0.5 * np.median(a), 0.15) for v in held.values())
    print(win, 'A median %.3f' % np.median(a), 'A>0.3 %d/%d' % ((a > 0.3).sum(), len(a)), {k: round(v, 3) for k, v in held.items()}, 'G1', g1, 'G2', g2)
    if win == [500]: ok = g1 and g2
print('VERDICT', 'PASS' if ok else 'FAIL')
