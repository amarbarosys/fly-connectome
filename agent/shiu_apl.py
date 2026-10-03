"""Stop the Shiu Kenyon-cell flood: which mushroom-body edges cause it, and what APL gain ends it?

In the authors' graph each KC gets as much excitation from other KCs (median +69 signed synapses,
axo-axonic in the lobes in the animal) as from its real inputs (+82), plus 60,657 DAN->KC synapses
counted as fast excitation (dopamine), against -25 of inhibition (APL about -17 to -20 per KC).
Variants zero edges in place (edge count unchanged, so all run side by side in one worker) and scale
KC->APL (gain in) and APL->KC (gain out). Per variant, 6 odours x K columns at the variant's ORN rate, 200 ms:
KC active fraction, A/B Jaccard of the KC sets per odour pair, MBON, APL and DN spikes;
KC and ALPN active fraction and pair Jaccard at each window, to see where odour sets merge.
Target: KC 5-10% active, Jaccard < 0.3, MBONs fire.
First sweep (APL alone): APL out x4 gives KC 6% but J 0.83, because ALPNs are already 66% active
with J 0.96 at 30 ms; the antennal lobe floods before the KCs. Second sweep: ALLN inhibition x16
(APL x1, 100 Hz) gives KC 3.9%, J 0.06, MBONs and DNs live; this sweep tunes around it.
"""
import csv
import json

import numpy as np

from agent import shiu_g1
from agent.brain_bench import ODORS6
from agent.neural_connectome import ANNOTATIONS, ROOT
from agent.shiu import ShiuGPU
from agent.shiu_learning import K, LATE, jaccard, mb_cells
from agent.shiu_loop import PAIRS, cells as olfactory, combine
from agent.verify_shiu import _reference_graph

VARIANTS = [  # (KC->KC kept, DAN->KC kept, APL gain in, APL gain out, ALLN inhibition gain, ORN Hz)
    (1, 1, 1, 1, 1, 200), (1, 1, 1, 1, 16, 100), (1, 1, 1, 1, 8, 100), (1, 1, 1, 1, 10, 100), (1, 1, 1, 1, 12, 100),
    (1, 1, 1, 1, 16, 200), (1, 1, 1, 1, 12, 200), (1, 1, 1, 1, 10, 200), (1, 1, 1, 1, 8, 200),
    (1, 1, 1, 2, 10, 200), (1, 1, 1, 2, 8, 200)]
WINDOWS = (300, 500, 1000, 2000)  # 30, 50, 100, 200 ms


def masks(n):
    index, apl = _reference_graph()[3], np.zeros(n, bool)
    for r in csv.DictReader(open(ANNOTATIONS), delimiter='\t'):
        i = index.get(int(r['root_id']))
        if i is not None and r['cell_type'] == 'APL':
            apl[i] = True
    kc, mbon = mb_cells()
    is_kc = np.zeros(n, bool)
    is_kc[kc] = True
    return is_kc, apl, kc, mbon, np.flatnonzero(apl)


def mb_variant(pre, post, w, names, is_kc, apl, kckc=1, dan=1, g_in=1, g_out=1, g_ln=1, hz=None):
    """Copy of w with KC->KC and DAN->KC zeroed when not kept, the APL loop and the antennal-lobe
    local neurons' inhibition (ALLN, negative edges) scaled; hz is the arm's ORN rate, not a graph edit."""
    v = w.copy()
    v[is_kc[pre] & is_kc[post]] *= kckc
    v[(names[pre] == 'DAN') & is_kc[post]] *= dan
    v[is_kc[pre] & apl[post]] *= g_in
    v[apl[pre] & is_kc[post]] *= g_out
    v[(names[pre] == 'ALLN') & (w < 0)] *= g_ln
    return v


def main():
    pre, post, w, names, _ = shiu_g1.flywire()
    n = len(names)
    orn, dn = olfactory('flywire')
    is_kc, apl, kc, mbon, apl_ids = masks(n)
    pn = np.flatnonzero(names == 'ALPN')
    graphs = [(pre, post, mb_variant(pre, post, w, names, is_kc, apl, *v)) for v in VARIANTS]
    arms, cols = len(VARIANTS), len(ODORS6) * K
    brain = ShiuGPU(*combine(graphs, n), n=arms * n, flies=cols, seed=61)
    for c in range(cols):
        ids = [i for g in ODORS6[c // K] for i in orn[g]]
        for a in range(arms):
            brain.stimulate([a * n + i for i in ids], VARIANTS[a][5], fly=c)
    windows = brain.run(list(WINDOWS))  # each (arms * n, cols), cumulative from 0
    counts = windows[-1]

    def code(x, cells):  # active fraction and max pair Jaccard of the per-odour sets
        sets = [set(cells[(x[cells, o * K:(o + 1) * K] > 0).mean(1) >= .5].tolist()) for o in range(len(ODORS6))]
        return float((x[cells] > 0).mean()), [jaccard(sets[p], sets[q]) for p, q in PAIRS]

    out = []
    for a, v in enumerate(VARIANTS):
        x = counts[a * n:(a + 1) * n]
        sets = [set(kc[(x[kc, o * K:(o + 1) * K] > 0).mean(1) >= .5].tolist()) for o in range(len(ODORS6))]
        time = {f'{t / 10:.0f}ms': {name: code(wx[a * n:(a + 1) * n], cells) for name, cells in (('kc', kc), ('pn', pn))}
                for t, wx in zip(WINDOWS, windows)}
        row = {'kckc': v[0], 'dan': v[1], 'apl_in': v[2], 'apl_out': v[3], 'ln': v[4], 'orn_hz': v[5],
               'kc_active': float((x[kc] > 0).mean()), 'jaccard': [jaccard(sets[p], sets[q]) for p, q in PAIRS],
               'mbon_spikes': float(x[mbon].sum(0).mean()), 'apl_spikes': float(x[apl_ids].sum(0).mean()),
               'dn_spikes': float(x[dn].sum(0).mean()), 'by_window': time}
        row['target'] = bool(.05 <= row['kc_active'] <= .10 and max(row['jaccard']) < .3 and row['mbon_spikes'] > 0)
        out.append(row)
        print(f"APL out {v[3]:2d} LN {v[4]:2d} ORN {v[5]:3d} Hz  KC {100 * row['kc_active']:5.1f}%  "
              f"J {max(row['jaccard']):.2f}  MBON {row['mbon_spikes']:7.1f}  APL {row['apl_spikes']:6.1f}  "
              f"DN {row['dn_spikes']:7.1f}  {'TARGET' if row['target'] else ''}", flush=True)
        print('    ' + '  '.join(f"{t} KC {100 * c['kc'][0]:4.1f}% J{max(c['kc'][1]):.2f} PN {100 * c['pn'][0]:4.1f}% J{max(c['pn'][1]):.2f}"
                                  for t, c in time.items()), flush=True)
    (ROOT / '.work/agent/shiu/apl-sweep.json').write_text(json.dumps(out, indent=2) + '\n')


if __name__ == '__main__':
    main()
