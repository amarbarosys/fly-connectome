"""WorkerBrain (GPU) against CPUBrain: odour drive, normalised gain, class gains, plasticity."""
import json
import time

import numpy as np

from agent.brain_bench import APL, odour_columns
from agent.mb_plasticity import mushroom_body_cells
from agent.neural_connectome import ROOT, CPUBrain, WorkerBrain
from agent import verify_sparse_learning

CONFIGS = (('raw', 1., 1., 1.), ('normalized', 35., 1., 1.), ('normalized', 35., 5., 80.))


def run(brain, drive, mode, gain, mbon_gain, dn_gain, cells):
    np.copyto(brain.graph.data, brain.base)
    brain.scale_edges(mbon_gain, post=cells['MBON'])
    brain.scale_edges(dn_gain, post=brain.descending)
    brain.gain = np.float32(gain)
    brain.reset()
    started = time.perf_counter()
    for _ in range(10):
        brain.step(np.zeros((drive.shape[1], 8), np.float32), extra_drive=drive)
    return brain.ever.copy(), brain.trace.copy(), time.perf_counter() - started


def verify():
    cells, drive, rows = mushroom_body_cells(), odour_columns(), []
    for mode in ('raw', 'normalized'):
        cpu = CPUBrain(flies=drive.shape[1], gain_mode=mode, gain=35)
        gpu = WorkerBrain(flies=drive.shape[1], gain_mode=mode, gain=35)
        cpu.base, gpu.base = cpu.graph.data.copy(), gpu.graph.data.copy()
        try:
            for config in (c for c in CONFIGS if c[0] == mode):
                ce, ct, cs = run(cpu, drive, *config, cells)
                ge, gt, gs = run(gpu, drive, *config, cells)
                rows.append({'config': config, 'ever_identical': bool(np.array_equal(ce, ge)),
                             'ever_cells_cpu': int(ce.sum()), 'trace_max_error': float(np.abs(ct - gt).max()),
                             'cpu_s': cs, 'gpu_s': gs})
        finally:
            gpu.close()
    learning = {}
    for name, cls in (('cpu', CPUBrain), ('gpu', WorkerBrain)):
        brain = cls(flies=3, gain_mode='normalized', gain=35)
        brain.normalized_base = brain.graph.data.copy()
        try:
            learning[name] = verify_sparse_learning.run_gain(
                brain, verify_sparse_learning.odor_drive(*verify_sparse_learning.ODORS), cells, 35., 5., 80.)
        finally:
            if name == 'gpu':
                brain.close()
    keys = ('kc_active_fraction', 'mbon_ever_count', 'A_over_B_learning_ratio', 'mbon_abs_response_delta')
    learning_match = all(np.allclose(learning['cpu'][k], learning['gpu'][k], rtol=1e-5, atol=1e-6) for k in keys)
    result = {'rows': rows, 'learning_cpu': {k: learning['cpu'][k] for k in keys},
              'learning_gpu': {k: learning['gpu'][k] for k in keys}, 'learning_match': bool(learning_match)}
    result['pass'] = bool(learning_match and all(r['ever_identical'] and r['trace_max_error'] <= 1e-5 for r in rows))
    out = ROOT / '.work/neural-flies/verify-worker-brain.json'
    out.write_text(json.dumps(result, indent=2, default=float) + '\n')
    print(json.dumps({'pass': result['pass'], 'rows': rows, 'learning_match': learning_match}, default=float))
    return result


if __name__ == '__main__':
    raise SystemExit(0 if verify()['pass'] else 1)
