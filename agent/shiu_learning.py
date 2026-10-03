"""Mushroom-body learning on the Shiu brain, step 1-2: is a reward to odour A specific to A?

On the authors' graph the Kenyon-cell layer floods (about 3,400 of 5,177 KCs fire for every odour by
200 ms), so a reward to A moved B's MBON response as much as A's at every eligibility window (FAIL,
3d3c5a3). Antennal-lobe inhibition (ALLN negative edges x LN, agent/shiu_apl.py) ends the flood.
On that graph, for each odour pair (A, B) in ODORS6 and eligibility window: reward A once by scaling
every KC -> MBON edge from A's KC set (KCs firing by the window in at least half of A's repeats) by
1 - DEPRESS (dopamine LTD, KC-only eligibility as e2dd362). Sham: as many KCs drawn from those active
for neither odour. Baseline, learned and sham run as weight variants of one graph with common random
numbers, so every difference is paired. Pass per pair: A loses more than twice what B loses, more
than twice what A loses under the sham, and at least 2 MBON spikes per column.
"""
import json

import numpy as np

from agent.brain_bench import ODORS6
from agent.experiment import Experiment, cells
from agent.neural_connectome import ROOT
from agent.shiu_loop import HZ, PAIRS
from agent.verify_shiu import _reference_graph

K, DEPRESS, LATE = 8, .5, 2000  # repeats per odour; LTD factor; 200 ms
EARLY = (500, 2000)  # eligibility windows: 50, 200 ms
LN = 12  # antennal-lobe (ALLN) inhibition gain that ends the flood (agent/shiu_apl.py); 1 = authors' graph


def jaccard(a, b):
    return len(a & b) / max(len(a | b), 1)


def main():
    pre, post, w, _ = _reference_graph()
    c = cells()
    names = c.klass
    is_kc, is_mbon = names == 'Kenyon_Cell', names == 'MBON'
    kc, mbon = np.flatnonzero(is_kc), np.flatnonzero(is_mbon)
    stimuli = {'+'.join(o): (np.flatnonzero((names == 'olfactory') & np.isin(c.type, ['ORN_' + g for g in o])), HZ)
               for o in ODORS6}
    ln = lambda p, q, x: (names[p] == 'ALLN') & (x < 0)

    def run(edges, variants, per_cell=()):
        return Experiment([(pre, post, w)], len(names), stimuli, EARLY, {'kc': kc, 'mbon': mbon}, {'ln': ln, **edges},
                          [{'ln': LN, **v} for v in variants], K, per_cell).run()

    base = run({}, [{}], ('kc',))['cells']['kc'][:, :, 0, 0]  # (window, kc, odour, repeat)
    sets = [[set(kc[(base[i, :, o] > 0).mean(1) >= .5].tolist()) for o in range(len(ODORS6))] for i in range(len(EARLY))]
    rng = np.random.default_rng(5)
    out = {'kcs': len(kc), 'mbons': len(mbon), 'depress': DEPRESS, 'alln_gain': LN, 'pairs': []}
    for i, window in enumerate(EARLY):
        for a, b in PAIRS:
            learn = np.zeros(len(names), bool)
            learn[list(sets[i][a])] = True
            sham = np.zeros(len(names), bool)
            spare = np.setdiff1d(kc, list(sets[-1][a] | sets[-1][b]))  # active for neither odour at 200 ms
            sham[rng.choice(spare, len(sets[i][a]), replace=False)] = True
            mb = run({'learn': lambda p, q, x: learn[p] & is_mbon[q], 'sham': lambda p, q, x: sham[p] & is_mbon[q]},
                     [{}, {'learn': 1 - DEPRESS}, {'sham': 1 - DEPRESS}])['sums']['mbon'][-1, 0]  # (variant, odour, repeat)
            per_column = mb.mean(2)
            drop = lambda v, o: float(per_column[0, o] - per_column[v, o])
            shared = np.zeros(len(names), bool)
            shared[list(sets[i][b])] = True
            row = {'A': list(ODORS6[a]), 'B': list(ODORS6[b]), 'window_ms': window / 10,
                   'kc': [len(sets[i][a]), len(sets[i][b])], 'jaccard': jaccard(sets[i][a], sets[i][b]),
                   'edges_changed': int((learn[pre] & is_mbon[post]).sum()), 'edges_also_B': int(((learn & shared)[pre] & is_mbon[post]).sum()),
                   'mbon_spikes_A_B_before': [float(per_column[0, a]), float(per_column[0, b])],
                   'drop_A': drop(1, a), 'drop_B': drop(1, b), 'sham_drop_A': drop(2, a), 'sham_drop_B': drop(2, b)}
            row['pass'] = row['drop_A'] > 2 * max(abs(row['drop_B']), abs(row['sham_drop_A']), 1.)
            out['pairs'].append(row)
            print(f"{window / 10:4.0f}ms {'+'.join(row['A']):12s} vs {'+'.join(row['B']):12s} KC {row['kc']} J {row['jaccard']:.2f}  "
                  f"edges {row['edges_changed']} (B-shared {row['edges_also_B']})  MBON before {row['mbon_spikes_A_B_before'][0]:.0f}/{row['mbon_spikes_A_B_before'][1]:.0f}  "
                  f"drop A {row['drop_A']:+.1f} B {row['drop_B']:+.1f}  sham A {row['sham_drop_A']:+.1f} B {row['sham_drop_B']:+.1f}  "
                  f"{'PASS' if row['pass'] else 'FAIL'}", flush=True)
    out['status'] = 'PASS' if all(r['pass'] for r in out['pairs']) else 'FAIL'
    print(out['status'])
    (ROOT / '.work/agent/shiu/learning-early.json').write_text(json.dumps(out, indent=2) + '\n')


if __name__ == '__main__':
    main()
