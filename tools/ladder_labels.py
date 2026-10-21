"""Label arrays for the block-shuffle resolution ladder (docs/plans/TYPEBLOCK_PREREG.md), in the learning graph's
node order (_reference_graph index, same as cells()). A blank label falls back to the next coarser one.
class_valence = cell_class with MBONs split into approach / avoid (learning_specificity.mbons).

    PYTHONPATH=. .venv/bin/python tools/ladder_labels.py   -> .work/agent/shiu/ladder-labels.npz
"""
import csv
import numpy as np
import agent.learning_specificity as L
from agent.experiment import cells, _reference_graph
from agent.neural_connectome import ANNOTATIONS, ROOT

KEYS = ('cell_type', 'supertype', 'cell_sub_class', 'cell_class', 'super_class', 'ito_lee_hemilineage')
LEVELS = {'cell_type': ('cell_type', 'cell_class', 'super_class'),
          'supertype': ('supertype', 'cell_type', 'cell_class', 'super_class'),
          'sub_class': ('cell_sub_class', 'cell_class', 'super_class'),
          'cell_class': ('cell_class', 'super_class'),
          'hemilineage': ('ito_lee_hemilineage', 'cell_class', 'super_class')}


def main():
    c = cells()
    index = _reference_graph()[3]
    n = len(c.klass)
    cols = {k: np.full(n, '', dtype=object) for k in KEYS}
    for r in csv.DictReader(open(ANNOTATIONS), delimiter='\t'):
        i = index.get(int(r['root_id']))
        if i is not None:
            for k in KEYS: cols[k][i] = r[k]
    assert (cols['cell_type'].astype(str) == np.asarray(c.type).astype(str)).all(), 'FAIL order: differs from cells()'
    out = {}
    for name, chain in LEVELS.items():
        lab = cols[chain[0]].copy()
        for k in chain[1:]:
            blank = lab == ''
            lab[blank] = cols[k][blank]
        lab[lab == ''] = 'unannotated'
        out[name] = lab.astype(str)
    approach, avoid = L.mbons(c)
    cv = out['cell_class'].astype(object).copy()
    cv[approach], cv[avoid] = 'MBON_approach', 'MBON_avoid'
    out['class_valence'] = cv.astype(str)
    np.savez(ROOT / '.work/agent/shiu/ladder-labels.npz', **out)
    print({k: len(np.unique(v)) for k, v in out.items()})


if __name__ == '__main__':
    main()
