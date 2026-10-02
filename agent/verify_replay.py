"""CPU-only full-connectome replay, isolation and checkpoint corruption gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np

from agent.neural_connectome import CPUBrain, ROOT, digest, N
from scipy.sparse import csr_matrix
from agent.neural_checkpoint import STATE_FIELDS, identity, restore_checkpoint, save_checkpoint


def state(brain):
    return {name: getattr(brain, name).copy() for name in STATE_FIELDS}


def equal_state(brain, expected):
    return all(np.array_equal(getattr(brain, name), value) for name, value in expected.items())


def verify_tiny_semantics():
    """Hand-defined signed graph embedded in the existing reference dimensions."""
    brain = CPUBrain.__new__(CPUBrain)
    # Row is receiver; column is sender. Node 3 is inhibitory output.
    edges = [(2, 0, 10.), (2, 1, 5.), (3, 0, -10.)]
    brain.graph = csr_matrix(([e[2] for e in edges],
        ([e[0] for e in edges], [e[1] for e in edges])), shape=(N, N), dtype=np.float32)
    brain.ports = np.zeros((N, 8), np.float32)
    brain.ports[0, 0] = 1
    brain.ports[1, 1] = 1
    brain.descending = np.array([2, 3], np.int32)
    brain.reset()
    voltage = np.zeros(4, np.float32)
    spikes = np.zeros(4, np.float32)
    trace = np.zeros(4, np.float32)
    ever = np.zeros(4, bool)
    inhibitory_seen = False
    for tick in range(16):
        inputs = np.zeros((2, 8), np.float32)
        if tick < 8:
            inputs[0, :2] = [1., .5]
        drive = np.array([inputs[0, 0], inputs[0, 1], 0, 0], np.float32) * np.float32(.65)
        incoming = np.zeros(4, np.float32)
        for receiver, sender, weight in edges:
            incoming[receiver] += np.float32(weight) * spikes[sender]
        next_voltage = np.float32(.85) * voltage + np.float32(.08) * incoming + drive
        spikes = (next_voltage >= 1).astype(np.float32)
        voltage = np.where(spikes != 0, 0, np.maximum(next_voltage, -5)).astype(np.float32)
        trace = np.float32(.85) * trace + np.float32(.15) * spikes
        ever |= spikes != 0
        brain.step(inputs, ticks=1)
        assert np.array_equal(brain.spikes[:4, 0], spikes), 'tiny spike oracle mismatch'
        assert np.allclose(brain.v[:4, 0], voltage, atol=1e-7, rtol=0), 'tiny voltage oracle mismatch'
        assert np.allclose(brain.trace[:4, 0], trace, atol=1e-7, rtol=0), 'tiny trace oracle mismatch'
        assert np.array_equal(brain.ever[:4, 0], ever), 'tiny recruitment mismatch'
        assert not np.any(brain.ever[4:]) and not np.any(brain.ever[:, 1])
        inhibitory_seen |= bool(brain.v[3, 0] < 0)
    assert inhibitory_seen and ever[2], 'fixture did not exercise signed propagation'
    brain.reset()
    assert not any(np.any(getattr(brain, name)) for name in STATE_FIELDS)
    return {'ticks': 16, 'signed_propagation': True, 'reset': True,
            'voltage_max_abs_tolerance': 1e-7,
            'scope': 'four-node scalar oracle embedded in CPU reference; not GPU or biological validation'}


def verify(out: Path):
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    tiny_gate = verify_tiny_semantics()
    brain = CPUBrain()
    rng = np.random.default_rng(20260930)
    prefix = rng.uniform(0, 1, (3, 2, 8)).astype(np.float32)
    suffix = rng.uniform(0, 1, (5, 2, 8)).astype(np.float32)
    for inputs in prefix:
        brain.step(inputs)
    assert np.any(brain.ever), 'fixture never recruited neurons'
    checkpoint = out / 'neural-state.npz'
    checkpoint_metadata = save_checkpoint(brain, checkpoint)
    saved_state = state(brain)
    original = [brain.step(inputs) for inputs in suffix]
    final_state = state(brain)
    brain.reset()
    restore_checkpoint(brain, checkpoint)
    assert equal_state(brain, saved_state), 'restored state differs'
    replayed = [brain.step(inputs) for inputs in suffix]
    assert original == replayed, 'neural response replay differs'
    assert equal_state(brain, final_state), 'replayed final state differs'

    # A second instance uses its own mutable state; a one-sided input stays local.
    peer = CPUBrain()
    untouched = state(brain)
    one_sided = np.zeros((2, 8), np.float32)
    one_sided[0] = .8
    response = peer.step(one_sided)
    assert not np.any(peer.ever[:, 1]), 'lane leakage'
    assert not np.any(np.asarray(response['features'])[1]), 'feature leakage'
    assert equal_state(brain, untouched), 'instance leakage'

    # A changed effective graph is rejected without changing destination state.
    peer.graph.data[0] += np.float32(.5)
    before = state(peer)
    try:
        restore_checkpoint(peer, checkpoint)
    except ValueError:
        pass
    else:
        raise AssertionError('changed graph accepted')
    assert equal_state(peer, before), 'failed restore mutated state'

    # Corrupt a finite state value without changing its recorded checksum.
    with np.load(checkpoint, allow_pickle=False) as archive:
        corrupt = {name: archive[name].copy() for name in archive.files}
    corrupt['v'][0, 0] = -.125 if corrupt['v'][0, 0] != -.125 else -.25
    damaged = out / 'corrupt-state.npz'
    np.savez_compressed(damaged, **corrupt)
    try:
        restore_checkpoint(brain, damaged)
    except ValueError:
        pass
    else:
        raise AssertionError('corrupt checkpoint accepted')
    assert equal_state(brain, final_state), 'corruption rejection mutated state'
    damaged.unlink()

    events = [{'index': i, 'inputs': inputs.tolist(), 'response': response}
              for i, (inputs, response) in enumerate(zip(suffix, original))]
    (out / 'events.jsonl').write_text(''.join(json.dumps(e, allow_nan=False) + '\n' for e in events))
    (out / 'config.json').write_text(json.dumps({'seed': 20260930, 'prefix': prefix.tolist(),
        'suffix': suffix.tolist(), 'ticks_per_exchange': 4}, indent=2) + '\n')
    manifest = {'scope': 'CPU neural state only; no body, GPU, readout learning or biological validation',
                'identity': identity(brain), 'checkpoint': checkpoint_metadata,
                'python': platform.python_version(), 'numpy': np.__version__,
                'source_hashes': {str(p.relative_to(ROOT)): digest(p) for p in
                    [Path(__file__), Path(__file__).with_name('neural_checkpoint.py')]}}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    report = {'status': 'PASS', 'tiny_semantics': tiny_gate, 'scope': manifest['scope'], 'neurons': brain.graph.shape[0],
              'edges': brain.graph.nnz, 'replayed_exchanges': len(suffix),
              'exact_response_and_state_replay': True, 'independent_instances_and_lanes': True,
              'changed_graph_rejected_without_mutation': True,
              'corrupt_state_rejected_without_mutation': True,
              'wall_seconds': time.perf_counter() - started,
              'checkpoint_sha256': digest(checkpoint)}
    (out / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True, help='New run directory; existing paths are refused')
    verify(parser.parse_args().out)
