"""Bounded CPU/GPU feature and learned-action parity on recorded sensory inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from agent.neural_connectome import CPUBrain, Readout, Worker, digest


def action(model, features):
    if np.max(np.abs(features)) < 1e-8:
        return 'straight'
    value = model.predict(features[None])[0]
    return 'left' if value[1] > value[0] else 'right'


def replay(brain, inputs):
    brain.reset()
    for values in inputs:
        packed = np.zeros((2, 8), np.float32); packed[0] = values
        result = brain.step(packed)
        assert result['counts'][1] == [0., 0.]
    return np.asarray(result['features'][0], np.float64)


def prepare(food, sequential, output):
    output.mkdir(parents=True, exist_ok=False)
    cases = []
    for path in sorted(food.glob('eval-*-trained.json')):
        row = json.loads(path.read_text())
        cases.append({'task': 'food', 'inputs': row['inputs'], 'source': str(path), 'sha256': digest(path)})
    for path in sorted(sequential.glob('train-*.json')):
        row = json.loads(path.read_text())
        cases.append({'task': 'avoidance_training', 'inputs': row['inputs'], 'source': str(path), 'sha256': digest(path)})
    assert len(cases) == 24
    models = {name: Readout(path) for name, path in [('food', food/'trained.npz'), ('sequential', sequential/'trained.npz')]}
    brain = CPUBrain(); features = []
    for case in cases:
        f = replay(brain, case['inputs']); features.append(f)
        case['expected_action'] = {name: action(model, f) for name, model in models.items()}
    np.savez(output/'oracle.npz', features=np.stack(features))
    receipt = {'cases': cases, 'models': {name: {'path': str(path), 'sha256': digest(path)}
        for name, path in [('food', food/'trained.npz'), ('sequential', sequential/'trained.npz')]},
        'feature_abs_tolerance': 1e-5, 'oracle_sha256': digest(output/'oracle.npz'),
        'scope': 'recorded sensory replay; no GPU physics or complete GPU body claim'}
    (output/'fixture.json').write_text(json.dumps(receipt, indent=2)+'\n')


def verify(output, backend):
    fixture = json.loads((output/'fixture.json').read_text())
    assert digest(output/'oracle.npz') == fixture['oracle_sha256']
    expected = np.load(output/'oracle.npz')['features']
    models = {}
    for name, row in fixture['models'].items():
        assert digest(Path(row['path'])) == row['sha256']
        models[name] = Readout(Path(row['path']))
    for row in fixture['cases']:
        assert digest(Path(row['source'])) == row['sha256']
    brain = Worker() if backend == 'gpu' else CPUBrain()
    started = time.perf_counter(); rows = []
    try:
        for index, case in enumerate(fixture['cases']):
            actual = replay(brain, case['inputs'])
            error = float(np.max(np.abs(actual-expected[index])))
            actions = {name: action(model, actual) for name, model in models.items()}
            rows.append({'task': case['task'], 'max_abs_error': error,
                         'actions_equal': actions == case['expected_action']})
        checks = {'features': max(row['max_abs_error'] for row in rows) <= fixture['feature_abs_tolerance'],
                  'learned_actions': all(row['actions_equal'] for row in rows)}
        receipt = {'status': 'PASS' if all(checks.values()) else 'FAIL', 'backend': backend,
            'checks': checks, 'cases': len(rows), 'neural_exchanges': len(rows)*8,
            'max_abs_feature_error': max(row['max_abs_error'] for row in rows),
            'wall_seconds_excluding_worker_startup': time.perf_counter()-started,
            'worker_ready': getattr(brain, 'ready', None), 'rows': rows, 'scope': fixture['scope']}
        (output/f'{backend}-verification.json').write_text(json.dumps(receipt, indent=2)+'\n')
        assert receipt['status'] == 'PASS', checks
        return receipt
    finally:
        if backend == 'gpu':
            brain.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'cpu', 'gpu'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--food-gate', type=Path)
    parser.add_argument('--sequential-gate', type=Path)
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args.food_gate, args.sequential_gate, args.output)
    else:
        print(json.dumps(verify(args.output, args.stage)))
