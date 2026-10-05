"""Exact Shiu CPU/GPU spike-count oracle on the male CNS graph."""
from __future__ import annotations

import json
import time

import numpy as np

from agent import mcns
from agent.shiu import ShiuBrain, ShiuGPU


def verify(steps=(100, 300, 1000), columns=2, seed=19):
    pre, post, weights, _, metadata = mcns.graph()
    n = int(metadata[0])
    descending = np.asarray(mcns.descending(), np.int64)
    # A reproducible subset makes the oracle non-vacuous while exercising full-graph delivery.
    cells = descending[::max(1, len(descending) // 64)][:64]
    runs = {}
    for engine in (ShiuBrain, ShiuGPU):
        brain = engine(pre, post, weights, n=n, flies=columns, seed=seed)
        brain.stimulate(cells, 200.0)
        start = time.perf_counter()
        counts = brain.run(list(steps))
        runs[engine.__name__] = (counts, time.perf_counter() - start)

    rows = []
    for i, step in enumerate(steps):
        cpu, gpu = runs['ShiuBrain'][0][i], runs['ShiuGPU'][0][i]
        rows.append({'steps': int(step),
                     'cells_differing': int(np.count_nonzero(np.any(cpu != gpu, axis=1))),
                     'cpu_spikes': int(cpu.sum()), 'gpu_spikes': int(gpu.sum())})
    result = {'dataset': mcns.RELEASE, 'neurons': n, 'edges': int(len(post)),
              'descending_inputs': int(len(cells)), 'seed': seed, 'rows': rows,
              'cpu_s': runs['ShiuBrain'][1], 'gpu_s': runs['ShiuGPU'][1],
              'pass': all(row['cells_differing'] == 0 for row in rows)
              and any(row['cpu_spikes'] > 0 for row in rows)}
    print(json.dumps(result, indent=2))
    if not result['pass']:
        raise SystemExit(1)
    return result


if __name__ == '__main__':
    verify()
