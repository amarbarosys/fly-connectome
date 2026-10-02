"""Versioned, identity-bound checkpoints for the CPU neural reference only.

Does not checkpoint MuJoCo, the resident GPU worker, readouts or motivation.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from agent.neural_connectome import CPUBrain, digest

STATE_FIELDS = ('v', 'spikes', 'trace', 'ever')
SCHEMA = 'fly-world.cpu-neural-checkpoint.v1'


def _array_hash(value):
    array = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(array.dtype).encode())
    h.update(json.dumps(array.shape).encode())
    h.update(memoryview(array).cast('B'))
    return h.hexdigest()


def identity(brain: CPUBrain):
    """Includes effective weights, so disconnected controls have distinct IDs."""
    arrays = {'row_ptr': brain.graph.indptr, 'col_idx': brain.graph.indices,
              'weights': brain.graph.data, 'ports': brain.ports,
              'descending': brain.descending}
    return {'schema': SCHEMA, 'shape': list(brain.graph.shape),
            'arrays': {name: _array_hash(value) for name, value in arrays.items()},
            'implementation_sha256': digest(Path(__file__).with_name('neural_connectome.py'))}


def _validate(brain, state):
    expected_shape = brain.v.shape
    for name in STATE_FIELDS:
        value = state[name]
        expected_dtype = np.dtype(bool if name == 'ever' else np.float32)
        if value.shape != expected_shape or value.dtype != expected_dtype:
            raise ValueError(f'invalid checkpoint shape/dtype: {name}')
        if not np.isfinite(value).all():
            raise ValueError(f'nonfinite checkpoint state: {name}')
    if np.any((state['spikes'] != 0) & (state['spikes'] != 1)):
        raise ValueError('checkpoint spikes must be binary')
    if np.any((state['trace'] < 0) | (state['trace'] > 1)):
        raise ValueError('checkpoint trace outside [0,1]')
    if np.any(state['v'] < -5) or np.any(state['v'] >= 1):
        raise ValueError('checkpoint voltage outside post-reset range')
    if np.any((state['spikes'] != 0) & ~state['ever']):
        raise ValueError('checkpoint spikes inconsistent with recruitment')


def save_checkpoint(brain: CPUBrain, path: Path):
    path = Path(path)
    state = {name: getattr(brain, name).copy() for name in STATE_FIELDS}
    _validate(brain, state)
    metadata = identity(brain)
    metadata['state_hashes'] = {name: _array_hash(value) for name, value in state.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    # Publish a complete checkpoint atomically, including on interrupted writes.
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            np.savez_compressed(stream, metadata=np.array(json.dumps(metadata)), **state)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return metadata


def restore_checkpoint(brain: CPUBrain, path: Path):
    # Validate the entire archive before mutating any destination array.
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {*STATE_FIELDS, 'metadata'}:
            raise ValueError('checkpoint fields mismatch')
        metadata = json.loads(str(archive['metadata'].item()))
        if not isinstance(metadata, dict):
            raise ValueError('checkpoint metadata must be an object')
        expected = identity(brain)
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError('checkpoint graph/ports/implementation identity mismatch')
        state = {name: archive[name].copy() for name in STATE_FIELDS}
    _validate(brain, state)
    if metadata.get('state_hashes') != {name: _array_hash(value) for name, value in state.items()}:
        raise ValueError('checkpoint state checksum mismatch')
    for name, value in state.items():
        setattr(brain, name, value)
    return metadata
