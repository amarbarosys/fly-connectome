"""Why do DNa01/DNa02 not lateralize the MCNS leg MNs (tools/mcns_mn_drive.py: L/R 1911/1611 vs 1917/1574 at 150 Hz)?
Wiring check on CPU: per source cell, signed synaptic reach onto left and right leg MNs over 1, 2 and 3 hops (weights as
the Shiu graph, >= 5 synapses, sign by transmitter), for the turning DNs vs the forward DNs of each side.
Reach_k = row of W^k summed over the target MNs; per-cell means so sets of different size compare.
usage: .venv/bin/python tools/mcns_dna_reach.py
"""
import json

import numpy as np
import scipy.sparse as sp

from agent import mcns
from agent.mcns_mapping import descending_name_sets, leg_motor_groups

pre, post, w, names, (n, *_) = mcns.graph()
W = sp.csr_matrix((w.astype(np.float64), (pre, post)), shape=(n, n))
A = sp.csr_matrix((np.abs(w).astype(np.float64), (pre, post)), shape=(n, n))
dn, _ = descending_name_sets()
mg, _ = leg_motor_groups()
mn = {s: np.concatenate([mg[k] for k in mg if k.endswith('_' + s)]) for s in 'LR'}
sets = {'DNa01/02 left': dn['turn_left'], 'DNa01/02 right': dn['turn_right'],
        'forward left': dn['forward_left'], 'forward right': dn['forward_right']}
out = {}
for name, cells in sets.items():
    x = sp.csr_matrix((np.ones(len(cells)), (np.zeros(len(cells), int), cells)), shape=(1, n))
    row = {}
    for k in (1, 2, 3):
        x = x @ W
        xa = (abs(x)).toarray().ravel()
        xs = x.toarray().ravel()
        row[f'hop{k}'] = {s: {'signed': float(xs[mn[s]].sum() / len(cells)), 'abs': float(xa[mn[s]].sum() / len(cells))} for s in 'LR'}
    out[name] = {'cells': int(len(cells)), **row}
print(json.dumps(out, indent=1))
