"""Mushroom-body learning on the Shiu brain, step 1-2: does an early-window KC code make reward specific?

The Shiu Kenyon-cell layer floods (most KCs fire for every odour by 200 ms), which is why a
KC-eligibility rule rewarded for A moved B's MBON response more than A's (f3e46c9). Here a KC is
eligible only if it fires in the first 30 ms of the odour, before the flood. For each odour pair
(A, B) in ODORS6, on the graph with antennal-lobe inhibition x LN (no flood): reward A once by depressing every KC -> MBON edge from A's early KC set by DEPRESS
(dopamine LTD, KC-only eligibility as e2dd362), then compare the MBON spikes lost for A and B.
Sham: depress as many KCs drawn from those active for neither odour. Pass per pair: A loses more
than twice what B loses, and more than twice what A loses under the sham, and at least 2 spikes
(absolute, since B's MBON response can be 1 spike). Eligibility windows of 50 and 200 ms.
Authors' graph (LN 1): FAIL at 20/25/30 ms, because the 200 ms readout floods (3d3c5a3). Real wiring only.
"""
import csv
import json

import numpy as np

from agent import shiu_g1
from agent.brain_bench import ODORS6
from agent.neural_connectome import ANNOTATIONS, ROOT
from agent.shiu import ShiuGPU
from agent.shiu_loop import HZ, PAIRS, cells as olfactory
from agent.verify_shiu import _reference_graph

K, DEPRESS, LATE = 8, .5, 2000  # columns per odour; LTD factor; 200 ms
EARLY = (500, 2000)  # eligibility windows: 50, 200 ms
LN = 12  # antennal-lobe (ALLN) inhibition gain that ends the flood (agent/shiu_apl.py); 1 = authors' graph


def mb_cells():
    index, kc, mbon = _reference_graph()[3], [], []
    for r in csv.DictReader(open(ANNOTATIONS), delimiter='\t'):
        i = index.get(int(r['root_id']))
        if i is not None and r['cell_class'] == 'Kenyon_Cell':
            kc.append(i)
        if i is not None and r['cell_class'] == 'MBON':
            mbon.append(i)
    return np.array(kc), np.array(mbon)


def respond(pre, post, w, n, orn, seed):
    """Counts at each EARLY window and at LATE, (n, 6 * K): K columns per odour in ODORS6 order."""
    brain = ShiuGPU(pre, post, w, n=n, flies=len(ODORS6) * K, seed=seed)
    for c in range(len(ODORS6) * K):
        brain.stimulate([i for g in ODORS6[c // K] for i in orn[g]], HZ, fly=c)
    return brain.run([*EARLY, LATE])


def jaccard(a, b):
    return len(a & b) / max(len(a | b), 1)


def main():
    pre, post, w, names, _ = shiu_g1.flywire()
    n = len(names)
    from agent.shiu_apl import masks, mb_variant  # shiu_apl builds on this module
    is_kc, apl = masks(n)[:2]
    w = mb_variant(pre, post, w, names, is_kc, apl, g_ln=LN)
    orn, _ = olfactory('flywire')
    kc, mbon = mb_cells()
    *earlies, late = respond(pre, post, w, n, orn, seed=51)
    by_odour = lambda x, o: x[:, o * K:(o + 1) * K]
    late_set = [set(kc[(by_odour(late, o)[kc] > 0).mean(1) >= .5].tolist()) for o in range(len(ODORS6))]
    out = {'kcs': len(kc), 'mbons': len(mbon), 'depress': DEPRESS, 'alln_gain': LN, 'pairs': []}
    is_mbon = np.zeros(n, bool)
    is_mbon[mbon] = True
    rng = np.random.default_rng(5)

    def depress(cells):
        mask = np.zeros(n, bool)
        mask[list(cells)] = True
        edges = mask[pre] & is_mbon[post]
        learned = w.copy()
        learned[edges] = np.round(learned[edges] * (1 - DEPRESS)).astype(w.dtype)
        return edges, respond(pre, post, learned, n, orn, seed=51)[-1]

    def drop(after, o):  # MBON spikes lost per column for odour o, 0-200 ms
        return float(by_odour(late, o)[mbon].sum(0).mean() - by_odour(after, o)[mbon].sum(0).mean())

    for window, early in zip(EARLY, earlies):
        early_set = [set(kc[(by_odour(early, o)[kc] > 0).mean(1) >= .5].tolist()) for o in range(len(ODORS6))]
        for a, b in PAIRS:
            edges, after = depress(early_set[a])
            # control: as many KCs, drawn from those active for neither odour of the pair at 200 ms
            spare = np.setdiff1d(kc, list(late_set[a] | late_set[b]))
            _, sham = depress(rng.choice(spare, len(early_set[a]), replace=False))
            b_eligible = np.zeros(n, bool)
            b_eligible[list(early_set[b])] = True
            row = {'A': list(ODORS6[a]), 'B': list(ODORS6[b]), 'window_ms': window / 10,
                   'early_kc': [len(early_set[a]), len(early_set[b])], 'late_kc': [len(late_set[a]), len(late_set[b])],
                   'early_jaccard': jaccard(early_set[a], early_set[b]), 'late_jaccard': jaccard(late_set[a], late_set[b]),
                   'edges_changed': int(edges.sum()), 'edges_also_B_early': int((edges & b_eligible[pre]).sum()),
                   'mbon_spikes_A_B_before': [float(by_odour(late, o)[mbon].sum(0).mean()) for o in (a, b)],
                   'drop_A': drop(after, a), 'drop_B': drop(after, b), 'sham_drop_A': drop(sham, a), 'sham_drop_B': drop(sham, b)}
            row['pass'] = row['drop_A'] > 2 * max(abs(row['drop_B']), abs(row['sham_drop_A']), 1.)
            out['pairs'].append(row)
            print(f"{window / 10:4.0f}ms {'+'.join(row['A']):12s} vs {'+'.join(row['B']):12s} KC {row['early_kc']} J {row['early_jaccard']:.2f}  "
                  f"edges {row['edges_changed']} (B-shared {row['edges_also_B_early']})  MBON before {row['mbon_spikes_A_B_before'][0]:.0f}/{row['mbon_spikes_A_B_before'][1]:.0f}  "
                  f"drop A {row['drop_A']:+.1f} B {row['drop_B']:+.1f}  sham A {row['sham_drop_A']:+.1f} B {row['sham_drop_B']:+.1f}  "
                  f"{'PASS' if row['pass'] else 'FAIL'}", flush=True)
    out['status'] = 'PASS' if all(r['pass'] for r in out['pairs']) else 'FAIL'
    print(out['status'])
    (ROOT / '.work/agent/shiu/learning-early.json').write_text(json.dumps(out, indent=2) + '\n')


if __name__ == '__main__':
    main()
