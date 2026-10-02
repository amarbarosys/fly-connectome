"""Strict JSON-line boundary for the resident Mojo neural worker."""
from __future__ import annotations
import json
import math
import sys
import ctypes
import numpy as np
from functools import lru_cache
from pathlib import Path
import os


@lru_cache(maxsize=10)
def state_identity(flies):
    from agent.neural_connectome import ROOT, ASSETS, digest
    return {'schema': 'fly-world.gpu-neural-state.v1', 'flies': flies,
            'disconnected': os.environ.get('FLY_NEURAL_DISCONNECT', '0') == '1',
            'files': {str(p.relative_to(ROOT)): digest(p) for p in
                      [ROOT/'data/row_ptr.bin', ROOT/'data/col_idx.bin', ROOT/'data/weights.bin',
                       ASSETS/'ports.bin', ASSETS/'descending.bin', ASSETS/'spatial-vision.npz',
                       ROOT/'agent/neural_worker.mojo', ROOT/'agent/neural_io.py']}}


def validate_state(state, flies):
    if set(state) != {'v', 'spikes', 'trace', 'ever'}:
        raise ValueError('neural state fields mismatch')
    for name, value in state.items():
        if value.shape != (139255, flies) or value.dtype != np.dtype(bool if name == 'ever' else np.float32):
            raise ValueError('neural state shape/dtype mismatch: ' + name)
        if not np.isfinite(value).all(): raise ValueError('nonfinite neural state')
    if np.any((state['spikes'] != 0) & (state['spikes'] != 1)): raise ValueError('nonbinary spikes')
    if np.any((state['trace'] < 0) | (state['trace'] > 1)): raise ValueError('trace outside [0,1]')
    if np.any((state['v'] < -5) | (state['v'] >= 1)): raise ValueError('invalid post-reset voltage')
    if np.any((state['spikes'] != 0) & ~state['ever']): raise ValueError('inconsistent recruitment')


def state_path(path):
    from agent.neural_connectome import ROOT
    value = Path(path).resolve()
    value.relative_to((ROOT/'.work/neural-flies/worker-state').resolve())
    return value


def receive(flies=2):
    line = sys.stdin.readline()
    if not line:
        return {"op": "quit"}
    request = json.loads(line)
    if request.get("op") not in {"step", "reset", "quit", "export_state", "import_state"}:
        raise ValueError("unknown neural operation")
    if request['op'] in {'export_state', 'import_state'}:
        request['path'] = str(state_path(request['path']))
        if request['op'] == 'import_state':
            with np.load(request['path'], allow_pickle=False) as archive:
                if set(archive.files) != {'v','spikes','trace','ever','identity'}:
                    raise ValueError('worker state archive fields mismatch')
                if json.loads(str(archive['identity'].item())) != state_identity(flies):
                    raise ValueError('worker state identity mismatch')
                request['_state'] = {name: archive[name].copy() for name in ('v','spikes','trace','ever')}
            validate_state(request['_state'], flies)
    if request["op"] == "step":
        if 'visual' in request:
            from agent.spatial_vision import mapping, validate_visual
            values=request['visual']
            if not isinstance(values,list) or len(values)!=flies*len(mapping()['nodes']):raise ValueError('visual input dimensions mismatch')
            request['_visual']=validate_visual(np.asarray(values).reshape(flies,-1),flies)
        values = request.get("inputs")
        if not isinstance(values, list) or len(values) != flies*8:
            raise ValueError(f"exactly {flies} independent eight-channel inputs required")
        if any(not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 <= x <= 1 for x in values):
            raise ValueError("sensory inputs must be finite and in [0,1]")
        direct=request.get('direct',[])
        if not isinstance(direct,list) or len(direct)>256 or any(type(i) is not int or not 0<=i<139255 for i in direct):
            raise ValueError('direct stimulation requires at most 256 valid neuron indices')
        request['direct']=direct
        after=request.get('after',0)
        if type(after) is not int or not 0<=after<=139255:raise ValueError('invalid inactive scan cursor')
        request['after']=after
    return request


def ready(features, flies=2, spatial=False):
    from agent.spatial_vision import mapping
    spatial_map=mapping()
    print(json.dumps({"ready": True, "neurons": 139255, "edges": 10572269,
                      "flies": flies, "features_per_fly": features, "state_transfer": True,
                      "spatial_drive":bool(spatial),
                      "spatial_mapping_sha256":spatial_map['sha256'] if spatial else None,
                      "spatial_neurons":len(spatial_map['nodes']) if spatial else 0}), flush=True)


def apply_visual(request, direct_address, flies):
    if '_visual' not in request:return
    from agent.spatial_vision import mapping
    values=np.frombuffer((ctypes.c_float*(flies*139255)).from_address(direct_address),dtype=np.float32).reshape(flies,139255)
    values[:,mapping()['nodes']]=request['_visual']


def reply(features, counts, inactive):
    print(json.dumps({"features": features, "counts": counts, "inactive": inactive}, allow_nan=False), flush=True)


def reply_buffers(feature_address, spike_address, ever_address, flies, features, neurons, after):
    """Read synchronously-owned Mojo host buffers without per-value Python calls."""
    def view(address, size):
        return np.frombuffer((ctypes.c_float*size).from_address(address), dtype=np.float32)
    f = view(feature_address, flies*features).reshape(flies, features)
    spikes = view(spike_address, flies*neurons).reshape(flies, neurons)
    ever = view(ever_address, flies*neurons).reshape(flies, neurons)
    counts = np.column_stack((spikes.sum(axis=1, dtype=np.float64), ever.sum(axis=1, dtype=np.float64)))
    inactive = (np.flatnonzero(ever[0, after:] == 0)+after)[:256]
    reply(f.tolist(), counts.tolist(), inactive.tolist())


def reset_reply():
    print('{"reset": true}', flush=True)


def state_buffers(request, voltage_address, spike_address, trace_address, ever_address, flies):
    views = {name: np.frombuffer((ctypes.c_float*(flies*139255)).from_address(address), dtype=np.float32).reshape(flies,139255)
             for name, address in zip(('v','spikes','trace','ever'),
                                       (voltage_address,spike_address,trace_address,ever_address))}
    if request['op'] == 'import_state':
        for name, view in views.items(): view[:] = request['_state'][name].T
    else:
        state = {name: view.T.copy().astype(bool if name == 'ever' else np.float32)
                 for name, view in views.items()}
        validate_state(state, flies)
        with state_path(request['path']).open('xb') as stream:
            np.savez_compressed(stream, identity=np.array(json.dumps(state_identity(flies))), **state)


def state_reply():
    print('{"state_transfer": true}', flush=True)
