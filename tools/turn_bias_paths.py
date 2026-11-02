"""Why leg senses bias the DN turn readout: linear signed path gain from each leg's sensory cells to readout T, by hop.
Weights row-normalised by total input; hop 1 = sensory -> DN, hop n = via n-1 interneurons. Connectivity only, no spikes.
    PYTHONPATH=. .venv/bin/python tools/turn_bias_paths.py
"""
import numpy as np
import scipy.sparse as sp

from agent import fly_legsense as ls, mcns

pre, post, w, _, (n, *_x) = mcns.graph()
v = np.load('.work/agent/mcns/vision_readout.npz', allow_pickle=True)
print('readout keys', list(v), 'y0', float(v['y0']))
ids = v['ids'] if 'ids' in v else v['dn']
gain = np.zeros(n); gain[ids] = v['w'] / v['sd']
tot = np.bincount(post, np.abs(w), n); tot[tot == 0] = 1
A = sp.csr_matrix((w / tot[post], (post, pre)), (n, n)).T.tocsr()  # A[i, j]: i -> j
C = ls.cells()
print('hop-wise T gain per group; total split by DN-side of the weight')
g = gain.copy()
for hop in range(1, 6):
    leg = {l: float(sum(g[np.asarray(c)].sum() for k, c in C.items() if k[0] == l)) for l in ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')}
    typ = {t: float(sum(g[np.asarray(c)].sum() for k, c in C.items() if k[1] == t)) for t in ('touch', 'load', 'angle')}
    L, R = sum(leg[l] for l in ('lf', 'lm', 'lh')), sum(leg[l] for l in ('rf', 'rm', 'rh'))
    print('hop', hop, 'L %.4f R %.4f L+R %.4f' % (L, R, L + R), {k: round(x, 4) for k, x in leg.items()}, {k: round(x, 4) for k, x in typ.items()})
    g = A @ g
