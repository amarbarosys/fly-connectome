"""GPU worker at 64 flies and with FLY_DATA_DIR: feature parity with CPUBrain on the same graph."""
import json
import os

import numpy as np

from agent.neural_connectome import ROOT, CPUBrain, Worker

ARM = ROOT / '.work/agent/matched-baseline/block-s7'


def parity(binary, flies, data_dir=None, steps=5):
    if data_dir:
        os.environ['FLY_DATA_DIR'] = str(data_dir)
    rng = np.random.default_rng(7)
    worker, oracle = Worker(binary, flies=flies), CPUBrain(flies=flies)
    if data_dir:
        oracle.load_columns(data_dir)
    try:
        error = 0.0
        for _ in range(steps):
            inputs = rng.random((flies, 8), np.float32)
            gpu = np.asarray(worker.step(inputs)['features'], np.float32)
            cpu = np.asarray(oracle.step(inputs)['features'], np.float32)
            error = max(error, float(np.abs(gpu - cpu).max()))
        return error
    finally:
        worker.close()
        os.environ.pop('FLY_DATA_DIR', None)


def verify(build=ROOT / '.work/neural-flies/build-64'):
    result = {'c64_real_max_error': parity(build / 'neural_worker_c64', 64),
              'c2_block_s7_max_error': parity(build / 'neural_worker', 2, ARM)}
    result['pass'] = all(v <= 1e-5 for v in result.values())
    print(json.dumps(result))
    return result


if __name__ == '__main__':
    raise SystemExit(0 if verify()['pass'] else 1)
