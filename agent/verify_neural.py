"""Full-graph GPU/CPU boundary and state-isolation gate before embodiment."""
from pathlib import Path
import json
import time
import numpy as np
from agent.neural_connectome import CPUBrain, Worker, ASSETS, ROOT, digest


def verify():
    worker = Worker()
    oracle = CPUBrain()
    results = {}
    started = time.perf_counter()
    try:
        inputs = np.array([[.2,.7,.1,.2,.1,.2,.15,.5],[0]*8], np.float32)
        gpu = worker.step(inputs)
        cpu = oracle.step(inputs)
        error = float(np.max(np.abs(np.asarray(gpu['features'])-np.asarray(cpu['features']))))
        results['one_exchange_descending_max_abs_error'] = error
        results['one_exchange_active_counts_gpu'] = gpu['counts']
        results['one_exchange_active_counts_cpu'] = cpu['counts']
        assert error < 2e-4, f'GPU/CPU descending parity failed: {error}'
        assert np.allclose(gpu['counts'], cpu['counts'], atol=1), 'GPU/CPU count mismatch'
        assert not np.any(np.asarray(gpu['features'])[1]), 'cross-fly state leakage'
        assert gpu['counts'][1] == [0, 0], 'cross-fly spike leakage'
        results['independent_state_gate'] = 'PASS'
        worker.reset()
        one = np.repeat(inputs[:1], 2, axis=0)
        both = worker.step(one)
        assert np.array_equal(both['features'][0], both['features'][1]), 'identical inputs diverged without noise'
        results['identical_input_repeatability'] = 'PASS'
        worker.reset()
        null = worker.step(np.zeros((2,8)))
        assert not np.any(null['features']) and not np.any(null['counts'])
        results['null_stimulus_gate'] = 'PASS'
        disconnected = CPUBrain(disconnected=True)
        no_edges = disconnected.step(one)
        assert not np.any(no_edges['features']), 'sensory features bypassed graph into descending readout'
        results['disconnected_graph_descending_gate'] = 'PASS'
        results['status'] = 'PASS'
        results['worker_sha256'] = digest(ROOT/'.work/neural-flies/neural_worker')
        results['input_manifest_sha256'] = digest(ASSETS/'manifest.json')
        results['wall_seconds'] = time.perf_counter()-started
        results['scope'] = 'software parity/isolation; no biological validation'
        (ASSETS/'neural-gate.json').write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps(results,indent=2),flush=True)
    finally:
        worker.close()


if __name__ == '__main__':
    verify()
