"""How low can learning push an odour's MBON response? The readout check for learning in the loop.

The loop will steer toward lower MBON drive (dopamine depresses KC -> MBON from the rewarded odour's
KCs), so the food odour must be able to fall below the decoy. On the flood-free graph (ALLN x LN),
for each odour of each pair: scale every KC -> MBON edge from that odour's 200 ms KC set by DEPTHS
(repeated rewards compound) and read MBON spikes for both odours of the pair.
"""
import json

import numpy as np

from agent.brain_bench import ODORS6
from agent.experiment import Experiment, cells
from agent.neural_connectome import ROOT
from agent.shiu_learning import HZ, K, LN, PAIRS
from agent.verify_shiu import _reference_graph

DEPTHS = (1, .5, .25, .125, 0)


def main():
    pre, post, w, _ = _reference_graph()
    c = cells()
    names = c.klass
    is_mbon = names == 'MBON'
    kc, mbon = np.flatnonzero(names == 'Kenyon_Cell'), np.flatnonzero(is_mbon)
    stimuli = {'+'.join(o): (np.flatnonzero((names == 'olfactory') & np.isin(c.type, ['ORN_' + g for g in o])), HZ) for o in ODORS6}
    ln = lambda p, q, x: (names[p] == 'ALLN') & (x < 0)
    base = Experiment([(pre, post, w)], len(names), stimuli, (2000,), {'kc': kc}, {'ln': ln}, [{'ln': LN}], K, ('kc',)).run()
    x = base['cells']['kc'][-1, :, 0, 0]  # (kc, odour, repeat)
    sets = [kc[(x[:, o] > 0).mean(1) >= .5] for o in range(len(ODORS6))]
    out = []
    for a, b in PAIRS:
        for own, other in ((a, b), (b, a)):
            learn = np.zeros(len(names), bool)
            learn[sets[own]] = True
            res = Experiment([(pre, post, w)], len(names), stimuli, (2000,), {'mbon': mbon},
                             {'ln': ln, 'learn': lambda p, q, x: learn[p] & is_mbon[q]},
                             [{'ln': LN, 'learn': d} for d in DEPTHS], K).run()
            m = res['sums']['mbon'][-1, 0].mean(2)  # (depth, odour)
            row = {'learned': '+'.join(ODORS6[own]), 'other': '+'.join(ODORS6[other]), 'kcs': len(sets[own]),
                   'depths': DEPTHS, 'mbon_learned': m[:, own].tolist(), 'mbon_other': m[:, other].tolist()}
            out.append(row)
            print(f"{row['learned']:14s} ({row['kcs']:4d} KCs) MBON own " + ' '.join(f'{v:6.1f}' for v in row['mbon_learned']) +
                  f"   other {row['other']:14s} " + ' '.join(f'{v:6.1f}' for v in row['mbon_other']), flush=True)
    (ROOT / '.work/agent/shiu/mbon-floor.json').write_text(json.dumps(out, indent=2) + '\n')


if __name__ == '__main__':
    main()
