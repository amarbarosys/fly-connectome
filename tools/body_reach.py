"""Interface inventory for LINK_BODY plan: neurons per FlyWire class, and share of the brain within N synapses of
every sensory neuron (forward) and of every motor/descending neuron (backward).   PYTHONPATH=. .venv/bin/python tools/body_reach.py"""
import csv
from collections import Counter

import numpy as np

from agent.neural_connectome import ANNOTATIONS
from agent.verify_shiu import _reference_graph

pre, post, _, index = _reference_graph()
pre, post = np.asarray(pre), np.asarray(post)
n = max(int(pre.max()), int(post.max())) + 1
sup, cla = np.full(n, '', dtype=object), np.full(n, '', dtype=object)
for r in csv.DictReader(open(ANNOTATIONS), delimiter='\t'):
    i = index.get(int(r['root_id']))
    if i is not None: sup[i], cla[i] = r['super_class'], r['cell_class']
print('super_class', Counter(sup).most_common())
print('sensory cell_class', Counter(cla[sup == 'sensory']).most_common(20))


def reach(start, a, b):
    seen = np.zeros(n, bool); seen[start] = True; out = []
    for hop in range(1, 7):
        new = np.zeros(n, bool); new[b[seen[a]]] = True; new &= ~seen; seen |= new
        out.append(round(float(seen.mean()), 3))
    return out


print('forward from sensory, hops 1-6', reach(np.isin(sup, ['sensory']), pre, post))
print('backward from motor+descending, hops 1-6', reach(np.isin(sup, ['motor', 'descending']), post, pre))
