"""Where each male CNS visual neuron looks: eye column and viewing direction per neuron.

Male CNS v1.0 annotates optic-lobe hex columns (assignedOlHex1/2) for columnar cells but not for
photoreceptors or LC/LPLC projection neurons. Method from flyverse (tel-0s/flyverse-core, MIT,
flyverse/connectome.py and retina.py): a photoreceptor takes the column (side, hex1, hex2) of its
strongest column-assigned postsynaptic partners (synapses summed per column); a column's viewing
direction comes from the hex embedding X = h1 - h2/2, Y = h2 sqrt(3)/2, (h1 + h2) dorsal, (h1 - h2)
anterior, 4.6 deg per column, each eye centred 55 deg lateral (azimuth + = left). Added here: a
non-columnar visual neuron's receptive-field centre is the synapse-weighted mean direction of its
column-assigned inputs (one hop). Output .work/agent/shiu/eye-map-mcns.npz: az, el (deg, NaN unknown).
"""
import os
import sys

import numpy as np
import pandas as pd

from agent import mcns
from agent.neural_connectome import ROOT

DEG, CENTRE = 4.6, 55.0
PATH = ROOT / '.work/agent/shiu/eye-map-mcns.npz'


def annotations():
    a = pd.read_feather(mcns.DATA / 'body-annotations-male-cns-v1.0-minconf-0.5.feather',
                        columns=['status', 'type', 'somaSide', 'assignedOlHex1', 'assignedOlHex2'])
    return a[a['status'].isin(['Traced', 'Anchor'])].reset_index(drop=True)  # mcns.graph() order


def build():
    a = annotations()
    pre, post, w, _, _ = mcns.graph()
    n = len(a)
    side = a['somaSide'].fillna('').str[:1].str.upper().to_numpy()
    h1, h2 = a['assignedOlHex1'].to_numpy(float), a['assignedOlHex2'].to_numpy(float)
    hexed = ~np.isnan(h1) & np.isin(side, ['L', 'R'])
    # photoreceptors inherit the column with the most synapses onto column-assigned partners
    pr = a['type'].fillna('').str.match(r'^R[1-8]').to_numpy()
    e = pr[pre] & hexed[post]
    votes = pd.DataFrame({'pr': pre[e], 's': side[post[e]], 'a': h1[post[e]], 'b': h2[post[e]], 'w': np.abs(w[e])})
    best = votes.groupby(['pr', 's', 'a', 'b']).w.sum().reset_index().sort_values('w').drop_duplicates('pr', keep='last')
    side, h1, h2 = side.copy(), h1.copy(), h2.copy()
    side[best.pr], h1[best.pr], h2[best.pr] = best.s, best.a, best.b
    hexed = ~np.isnan(h1) & np.isin(side, ['L', 'R'])
    # hex column -> azimuth, elevation (flyverse retina.build_retina), centred per eye
    az, el = np.full(n, np.nan), np.full(n, np.nan)
    X, Y = h1 - h2 / 2, h2 * np.sqrt(3) / 2
    dv, ap = .5 * X + np.sqrt(3) / 2 * Y, np.sqrt(3) / 2 * X - .5 * Y
    for s, sign in (('L', 1), ('R', -1)):
        m = hexed & (side == s)
        cols = np.unique(np.c_[h1[m], h2[m]], axis=0)  # centre on the eye's columns, not its cells
        cx, cy = cols[:, 0] - cols[:, 1] / 2, cols[:, 1] * np.sqrt(3) / 2
        az[m] = sign * (CENTRE - (ap[m] - (np.sqrt(3) / 2 * cx - .5 * cy).mean()) * DEG)
        el[m] = (dv[m] - (.5 * cx + np.sqrt(3) / 2 * cy).mean()) * DEG
    # non-columnar neurons: synapse-weighted mean direction of column-assigned inputs
    e = hexed[pre] & ~hexed[post] & (w > 0)
    unit = np.c_[np.cos(np.radians(el)) * np.cos(np.radians(az)), np.cos(np.radians(el)) * np.sin(np.radians(az)), np.sin(np.radians(el))]
    acc = np.zeros((n, 3))
    np.add.at(acc, post[e], unit[pre[e]] * w[e][:, None])
    got = (np.linalg.norm(acc, axis=1) > 0) & ~hexed
    acc[got] /= np.linalg.norm(acc[got], axis=1, keepdims=True)
    az[got] = np.degrees(np.arctan2(acc[got, 1], acc[got, 0]))
    el[got] = np.degrees(np.arcsin(np.clip(acc[got, 2], -1, 1)))
    np.savez(PATH, az=az, el=el, column=hexed)
    return a, az, el, hexed, pr


def main():
    a, az, el, column, pr = build()
    if os.environ.get('MUTANT') == 'mirror':
        az = -az
    t = a['type'].fillna('').to_numpy()
    soma = a['somaSide'].fillna('').str[:1].str.upper().to_numpy()
    print(f"photoreceptors with a column {int((pr & column).sum())}/{int(pr.sum())}")
    for s in ('L', 'R'):
        m = pr & column & (np.sign(az) == (1 if s == 'L' else -1))
        print(f"  eye {s}: azimuth {np.nanmin(az[m]):.0f}..{np.nanmax(az[m]):.0f}, elevation {np.nanmin(el[m]):.0f}..{np.nanmax(el[m]):.0f} deg, "
              f"{int(m.sum())} photoreceptors")
    side = {}
    for kind in ('LC4', 'LPLC2', 'LC9', 'LC31a'):
        m = t == kind
        ok = m & ~np.isnan(az)
        agree = np.mean(np.sign(az[ok]) == np.where(soma[ok] == 'L', 1, -1)) if ok.any() else float('nan')
        side[kind] = agree
        print(f"  {kind:6s} {int(ok.sum())}/{int(m.sum())} with a field; azimuth sign matches soma side {agree:.2f}; "
              f"az {np.nanmin(az[ok]):.0f}..{np.nanmax(az[ok]):.0f}")
    ok = [('photoreceptors placed >= 3000', int((pr & column).sum()) >= 3000),
          ('LC4 field on soma side >= 0.95', side['LC4'] >= 0.95),
          ('LPLC2 field on soma side >= 0.90', side['LPLC2'] >= 0.90)]
    for k, v in ok:
        print(('PASS ' if v else 'FAIL ') + k)
    if not all(v for _, v in ok):
        sys.exit('FAIL eye map')
    print('PASS eye map')


if __name__ == '__main__':
    main()
