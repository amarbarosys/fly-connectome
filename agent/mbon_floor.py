"""How low can learning push an odour's MBON response? The readout check for learning in the loop.

Total MBON drive cannot (a813237): with every KC -> MBON edge of an odour removed, non-KC input keeps
the high odour above its partner. Aso et al. 2014 (eLife 3:e04580): MBONs whose activation repels are
glutamatergic, those that attract are GABAergic or cholinergic. So the readout is the balance
approach - avoid (labels in agent/mb_valence.py), and reward depresses KC -> avoid-MBON edges only.
With KC -> MBON at the authors' weight the avoid MBONs barely fire (0.2-3.4 spikes for 5 of 6 odours)
and depressing their KC input changes nothing, so KC -> MBON is swept by GAINS. On the flood-free
graph (ALLN x LN), for each odour of each pair: scale the KC -> avoid-MBON edges from that odour's
200 ms KC set by DEPTHS on top of the gain; read approach and avoid spikes for both odours.
"""
import json

import numpy as np

from agent.brain_bench import ODORS6
from agent.experiment import Experiment, cells
from agent.mb_valence import mbons
from agent.neural_connectome import ROOT
from agent.shiu_learning import HZ, K, LN, PAIRS
from agent.verify_shiu import _reference_graph

DEPTHS = (1, .25, 0)
GAINS = (1, 2, 4, 8)


def main():
    pre, post, w, _ = _reference_graph()
    c = cells()
    names = c.klass
    kc, is_kc = np.flatnonzero(names == 'Kenyon_Cell'), names == 'Kenyon_Cell'
    approach, avoid_ids = mbons(c)
    is_mbon, avoid = np.zeros(len(names), bool), np.zeros(len(names), bool)
    is_mbon[approach], is_mbon[avoid_ids], avoid[avoid_ids] = True, True, True
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
            res = Experiment([(pre, post, w)], len(names), stimuli, (2000,), {'approach': approach, 'avoid': avoid_ids},
                             {'ln': ln, 'gain': lambda p, q, x: is_kc[p] & is_mbon[q], 'learn': lambda p, q, x: learn[p] & avoid[q]},
                             [{'ln': LN, 'gain': g, 'learn': g * d} for g in GAINS for d in DEPTHS], K).run()
            app, avo = (res['sums'][g][-1, 0].mean(2).reshape(len(GAINS), len(DEPTHS), -1) for g in ('approach', 'avoid'))  # (gain, depth, odour)
            row = {'learned': '+'.join(ODORS6[own]), 'other': '+'.join(ODORS6[other]), 'kcs': len(sets[own]), 'gains': GAINS,
                   'depths': DEPTHS, 'approach': app[:, :, [own, other]].tolist(), 'avoid': avo[:, :, [own, other]].tolist()}
            out.append(row)
            print(f"{row['learned']:14s} ({row['kcs']:4d} KCs) vs {row['other']}", flush=True)
            for i, g in enumerate(GAINS):
                print(f"   gain {g}: own approach/avoid " + '  '.join(f'{app[i, j, own]:5.1f}/{avo[i, j, own]:5.1f}' for j in range(len(DEPTHS))) +
                      f"   other {app[i, 0, other]:5.1f}/{avo[i, 0, other]:5.1f}", flush=True)
    (ROOT / '.work/agent/shiu/mbon-balance.json').write_text(json.dumps(out, indent=2) + '\n')


if __name__ == '__main__':
    main()
