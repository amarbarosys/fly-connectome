"""Slow neuromodulation (ShiuBrain.modulators): dopamine, octopamine and serotonin edges feed a decaying level that
scales the target's excitability instead of acting as fast synapses.

exact: GPU spike counts per cell equal the CPU oracle's at 50 steps with modulation ON (sugar input, 2 columns,
identical Poisson draws), as verify_shiu.exact_check does for the unmodulated model.

usage: gpu-wait run --priority 90 --timeout 900 -- .venv/bin/python -m agent.verify_modulation
"""
import json
import re

from agent.experiment import cells
from agent.shiu import ShiuBrain, ShiuGPU
from agent.verify_shiu import REF, _reference_graph

K = (0.05, 0.05, 0.05)  # gain per unit level: strong enough that modulation changes spikes within 200 steps


def exact(steps=(50, 100, 200), columns=2):
    pre, post, w, index = _reference_graph()
    source = (REF / 'code/benchmark.py').read_text()
    sugar = [index[int(x)] for x in re.findall(r'7205\d{14}', source[source.index("'sugar'"):source.index("'p9'")])]
    nt, runs = cells().nt, {}
    for engine in (ShiuBrain, ShiuGPU):
        brain = engine(pre, post, w, n=len(index), flies=columns, seed=7)
        brain.stimulate(sugar, 200.0)
        brain.modulators(nt, k=K)
        runs[engine.__name__] = brain.run(list(steps))
    off = ShiuBrain(pre, post, w, n=len(index), flies=columns, seed=7)
    off.stimulate(sugar, 200.0)
    plain = off.run(list(steps))
    rows = [{'steps': s, 'cells_differing': int((a != b).any(axis=1).sum()), 'spikes_cpu': int(a.sum()), 'spikes_gpu': int(b.sum()),
             'cells_changed_by_modulation': int((a != c).any(axis=1).sum())}
            for s, a, b, c in zip(steps, runs['ShiuBrain'], runs['ShiuGPU'], plain)]
    return {'rows': rows, 'pass': rows[0]['cells_differing'] == 0 and rows[-1]['cells_changed_by_modulation'] > 0}


if __name__ == '__main__':
    result = {'exact': exact()}
    result['pass'] = all(r['pass'] for r in result.values())
    print(json.dumps(result))
    raise SystemExit(0 if result['pass'] else 1)
