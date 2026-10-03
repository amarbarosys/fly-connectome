"""Slow neuromodulation (ShiuBrain.modulators): dopamine, octopamine and serotonin edges feed a decaying level that
scales the target's excitability instead of acting as fast synapses.

exact: GPU spike counts per cell equal the CPU oracle's at 50 steps with modulation ON (sugar input, 2 columns,
identical Poisson draws), as verify_shiu.exact_check does for the unmodulated model.

persistence (CPU oracle, which exact shows equals the GPU): sugar GRNs drive the brain throughout; all 210 octopamine
neurons are driven for 200 ms, then released. Spikes of their targets right after release (0-100 ms) and much later
(1.5-2.0 s) are compared with the same run without modulation. PASS: early >= 1.05 x unmodulated and late within 3%
(the level decayed). Bars set after the first run (early 1.13, late 0.99; mutant early 1.01): they separate a
lasting level from none, they are not a physiology target. The mutant (tau 0.1 ms) must fail.

usage: gpu-wait run --priority 90 --timeout 900 -- .venv/bin/python -m agent.verify_modulation
"""
import json
import re

from agent.experiment import cells
from agent.shiu import ShiuBrain, ShiuGPU
from agent.verify_shiu import REF, _reference_graph

import numpy as np

K = (0.05, 0.05, 0.05)  # gain per unit level: strong enough that modulation changes spikes within 200 steps
B = (0.1, 0.1, 0.1)  # slow current per unit level, on in the exact check so both terms are compared


def exact(steps=(50, 100, 200), columns=2):
    pre, post, w, index = _reference_graph()
    source = (REF / 'code/benchmark.py').read_text()
    sugar = [index[int(x)] for x in re.findall(r'7205\d{14}', source[source.index("'sugar'"):source.index("'p9'")])]
    nt, runs = cells().nt, {}
    for engine in (ShiuBrain, ShiuGPU):
        brain = engine(pre, post, w, n=len(index), flies=columns, seed=7)
        brain.stimulate(sugar, 200.0)
        brain.modulators(nt, k=K, b=B)
        runs[engine.__name__] = brain.run(list(steps))
    off = ShiuBrain(pre, post, w, n=len(index), flies=columns, seed=7)
    off.stimulate(sugar, 200.0)
    plain = off.run(list(steps))
    rows = [{'steps': s, 'cells_differing': int((a != b).any(axis=1).sum()), 'spikes_cpu': int(a.sum()), 'spikes_gpu': int(b.sum()),
             'cells_changed_by_modulation': int((a != c).any(axis=1).sum())}
            for s, a, b, c in zip(steps, runs['ShiuBrain'], runs['ShiuGPU'], plain)]
    return {'rows': rows, 'pass': rows[0]['cells_differing'] == 0 and rows[-1]['cells_changed_by_modulation'] > 0}


def persistence(tau_oa=1.0, k=1e-3, ms=1e-4):
    pre, post, w, index = _reference_graph()
    source = (REF / 'code/benchmark.py').read_text()
    sugar = [index[int(x)] for x in re.findall(r'7205\d{14}', source[source.index("'sugar'"):source.index("'p9'")])]
    nt = cells().nt
    oa = np.flatnonzero(nt == 'octopamine')
    spikes = {}
    for arm in ('modulated', 'plain'):
        b = ShiuBrain(pre, post, w, n=len(index), seed=7)
        b.stimulate(sugar, 200.0)
        b.stimulate(oa, 200.0)
        q = b.modulators(nt, tau=(0.5, tau_oa, 2.0), k=(0, k if arm == 'modulated' else 0, 0))
        targets = np.setdiff1d(np.unique(b.col[q == 1]), np.r_[oa, sugar])
        b.step(round(0.2 / ms))
        for cell in oa:
            del b.stim[int(cell)], b.key[int(cell)]
        windows = []
        for start, end in ((0.0, 0.1), (1.5, 2.0)):
            b.step(round(start / ms) - (b.t - round(0.2 / ms)))
            before = b.counts[targets].sum()
            b.step(round((end - start) / ms))
            windows.append(int(b.counts[targets].sum() - before))
        spikes[arm] = windows
    early, late = (spikes['modulated'][i] / max(spikes['plain'][i], 1) for i in (0, 1))
    return {'targets': int(len(targets)), 'spikes': spikes, 'early_ratio': early, 'late_ratio': late, 'tau_oa': tau_oa,
            'pass': early >= 1.05 and abs(late - 1) < 0.03}


if __name__ == '__main__':
    import sys
    if sys.argv[1:] == ['mutant']:  # tau 0.1 ms: the level vanishes within a step, so nothing may persist
        r = persistence(tau_oa=1e-4)
        print(json.dumps(r))
        raise SystemExit(0 if r['pass'] else 1)
    result = {'exact': exact(), 'persistence': persistence()}
    result['pass'] = all(r['pass'] for r in result.values())
    print(json.dumps(result))
    raise SystemExit(0 if result['pass'] else 1)
