"""Mushroom-body output valence labels, from the literature, in FlyWire MBON type names.

Aso et al. 2014a (eLife 3:e04577, Table 1): MBON-01..07 glutamate, 08..11 GABA, 12..19 acetylcholine.
Aso et al. 2014b (eLife 3:e04580): "all MBONs eliciting aversion were glutamatergic and all the MBONs
eliciting attraction were either GABAergic or cholinergic". The avoid MBONs sit in compartments fed by
PAM (reward) dopamine neurons (gamma5, beta'2, beta2, gamma4, beta1, alpha1), so reward depresses
KC -> avoid-MBON synapses. FlyWire's top_nt calls MBON10 glutamatergic; Aso's table (GABA) is used.
MBON20 and above have no label here and stay out of the readout. Full texts in .work/papers/.
"""
import numpy as np

AVOID = tuple(f'MBON{i:02d}' for i in range(1, 8))
APPROACH = tuple(f'MBON{i:02d}' for i in range(8, 20))


def mbons(cells):
    """(approach, avoid) neuron indices; FlyWire types like 'MBON15-like' count with their type."""
    base = np.array([t.split('-')[0] for t in cells.type])
    mbon = cells.klass == 'MBON'
    return np.flatnonzero(mbon & np.isin(base, APPROACH)), np.flatnonzero(mbon & np.isin(base, AVOID))
