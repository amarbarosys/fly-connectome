"""Exploratory (private note TYPEBLOCK_PREREG plan 3): why does class_valence.2 keep the learning gap and .0/.1/.3/.4 not?
Reads stored receipts only. Per realisation, mean MBON spikes (learners, flies 0-7; controls 8-15) for odour A,
before and after training.    python tools/ladder_diff.py [LEVEL]
"""
import glob
import json
import sys

import numpy as np

level = sys.argv[1] if len(sys.argv) > 1 else 'class_valence'
tag = 'block{}' if level == 'block' else 'labels' + level + '.{}'
arms = [(tag.format(k), tag.format(k)) for k in range(5)]
print('arm  pre_ap pre_av | L_ap L_av | C_ap C_av | gap_seeds')
for name, tag in arms:
    fs = sorted(glob.glob(f'.work/agent/shiu/specificity/runs/*/run-{tag}-0.json'))
    if not fs:
        print(name, 'FAIL no receipt'); continue
    recs = [r for r in json.load(open(fs[-1]))['records'] if r.get('reward', True)]
    get = lambda st, key, f: np.mean([r[st]['o1'][key][f] for r in recs], 0)
    L, C = slice(0, 8), slice(8, 16)
    p, q = (lambda k: get('pre', k, slice(None)).mean()), (lambda k, f: get('after_A', k, f).mean())
    gaps = [np.mean(r['after_A']['o1']['valence'][8:]) - np.mean(r['after_A']['o1']['valence'][:8]) for r in recs]
    print(f"{name:18s} {p('approach'):7.1f} {p('avoid'):7.1f} | {q('approach', L):7.1f} {q('avoid', L):7.1f} | "
          f"{q('approach', C):7.1f} {q('avoid', C):7.1f} | {np.round(gaps, 2)}")
