"""Mushroom-body learning on the Shiu brain, step 1-2: does an early-window KC code make reward specific?

The Shiu Kenyon-cell layer floods (most KCs fire for every odour by 200 ms), which is why a
KC-eligibility rule rewarded for A moved B's MBON response more than A's (f3e46c9). Here a KC is
eligible only if it fires in the first 30 ms of the odour, before the flood. For each odour pair
(A, B) in ODORS6: reward A once by depressing every KC -> MBON edge from A's early KC set by DEPRESS
(dopamine LTD, KC-only eligibility as e2dd362), then compare the MBON response change for A and B.
Pass per pair: A's response moves more than twice B's (the f3e46c9 edge-ratio bar of 2,
applied to the response). Swept over eligibility windows of 20, 25 and 30 ms. Real wiring only.
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
EARLY = (200, 250, 300)  # eligibility windows: 20, 25, 30 ms


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
    orn, _ = olfactory('flywire')
    kc, mbon = mb_cells()
    *earlies, late = respond(pre, post, w, n, orn, seed=51)
    by_odour = lambda x, o: x[:, o * K:(o + 1) * K]
    late_set = [set(kc[(by_odour(late, o)[kc] > 0).mean(1) >= .5].tolist()) for o in range(len(ODORS6))]
    out = {'kcs': len(kc), 'mbons': len(mbon), 'depress': DEPRESS, 'pairs': []}
    is_mbon = np.zeros(n, bool)
    is_mbon[mbon] = True
    for window, early in zip(EARLY, earlies):
        early_set = [set(kc[(by_odour(early, o)[kc] > 0).mean(1) >= .5].tolist()) for o in range(len(ODORS6))]
        for a, b in PAIRS:
            eligible = np.zeros(n, bool)
            eligible[list(early_set[a])] = True
            edges = eligible[pre] & is_mbon[post]
            learned = w.copy()
            learned[edges] = np.round(learned[edges] * (1 - DEPRESS)).astype(w.dtype)
            after = respond(pre, post, learned, n, orn, seed=51)[-1]
            move = []
            for o in (a, b):
                r0, r1 = by_odour(late, o)[mbon].mean(1), by_odour(after, o)[mbon].mean(1)
                move.append(float(np.linalg.norm(r1 - r0) / (np.linalg.norm(r0) + 1e-9)))
            b_eligible = np.zeros(n, bool)
            b_eligible[list(early_set[b])] = True
            row = {'A': list(ODORS6[a]), 'B': list(ODORS6[b]),
                   'early_kc': [len(early_set[a]), len(early_set[b])], 'late_kc': [len(late_set[a]), len(late_set[b])],
                   'early_jaccard': jaccard(early_set[a], early_set[b]), 'late_jaccard': jaccard(late_set[a], late_set[b]),
                   'edges_changed': int(edges.sum()), 'edges_also_B_early': int((edges & b_eligible[pre]).sum()),
                   'mbon_spikes_A_B_before': [float(by_odour(late, o)[mbon].sum(0).mean()) for o in (a, b)],
                   'window_ms': window / 10, 'response_move_A': move[0], 'response_move_B': move[1],
                     'specificity': move[0] / max(move[1], 1e-9), 'pass': move[0] > 2 * move[1]}
            out['pairs'].append(row)
            print(f"{window / 10:4.0f}ms {'+'.join(row['A']):12s} vs {'+'.join(row['B']):12s} KC early {row['early_kc']} late {row['late_kc']} "
                  f"J early {row['early_jaccard']:.2f} late {row['late_jaccard']:.2f}  edges {row['edges_changed']} "
                  f"(B-shared {row['edges_also_B_early']})  MBON {row['mbon_spikes_A_B_before'][0]:.0f}/{row['mbon_spikes_A_B_before'][1]:.0f}  "
                  f"move A {move[0]:.3f} B {move[1]:.3f}  {'PASS' if row['pass'] else 'FAIL'}", flush=True)
    out['status'] = 'PASS' if all(r['pass'] for r in out['pairs']) else 'FAIL'
    print(out['status'])
    (ROOT / '.work/agent/shiu/learning-early.json').write_text(json.dumps(out, indent=2) + '\n')


if __name__ == '__main__':
    main()
