"""Fit readout DN pair gains from probe dump (FULL_FLY DN pair gains).  PYTHONPATH=. python tools/dn_gain_fit.py SEED"""
import sys

import numpy as np

from agent.body_engine import fly_link as fl

z = np.load(fl.OUT / f'hybrid-dn-{sys.argv[1]}.npz')
v = np.load('.work/agent/mcns/vision_readout.npz')
r = np.maximum(z['dn'][z['arms'] == 'all'].mean(0), 0.5)
_, grp = np.unique(np.round(np.abs(v['w']), 9), return_inverse=True)
mean = np.bincount(grp, r) / np.bincount(grp)
g = np.clip(mean[grp] / r, 0.25, 4)
np.save(fl.OUT / 'dn-gain.npy', g)
print('groups', grp.max() + 1, 'group sizes>2', int((np.bincount(grp) > 2).sum()), 'g range', g.min().round(2), g.max().round(2))
