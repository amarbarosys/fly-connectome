"""Identity-bound, single-file continuation snapshots for the single-fly runner."""
from collections import deque
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from agent.neural_connectome import ROOT, ASSETS, Worker, digest
from agent.neural_io import validate_state
from fly_arena.checkpoint import code_digest, load_checkpoint, save_checkpoint

FIELDS = ('v', 'spikes', 'trace', 'ever')


def tree_hash(value):
    if isinstance(value, np.ndarray):
        h = hashlib.sha256()
        h.update(str(value.dtype).encode())
        h.update(json.dumps(value.shape).encode())
        h.update(np.ascontiguousarray(value).tobytes())
        return h.hexdigest()
    if isinstance(value, dict):
        value = {key: tree_hash(item) for key, item in sorted(value.items())}
    elif isinstance(value, (list, tuple)):
        value = [tree_hash(item) for item in value]
    elif isinstance(value, np.generic):
        value = value.item()
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def compatibility(arena, brain, backend, profile):
    import mujoco
    sources = ('live_checkpoint.py', 'neural_single.py', 'neural_connectome.py',
               'neural_worker.mojo', 'neural_io.py', 'neural_duel.py', 'desire.py', 'vision.py','spatial_vision.py')
    result = {'schema':'fly-world.single-live-checkpoint.v1', 'backend':backend, 'profile':profile,
              'neural_flies':brain.flies, 'mujoco':mujoco.__version__, 'numpy':np.__version__,
              'model_sha256':hashlib.sha256(arena.sim.mj_model.__getstate__()).hexdigest(),
              'arena_code_sha256':code_digest(),
              'sources':{name:digest(ROOT/'agent'/name) for name in sources},
              'assets':{str(path.relative_to(ROOT)):digest(path) for path in
                        [ROOT/'data/row_ptr.bin',ROOT/'data/col_idx.bin',ROOT/'data/weights.bin',
                         ASSETS/'ports.npz',ASSETS/'ports.bin',ASSETS/'descending.bin',ASSETS/'spatial-vision.npz']}}
    if isinstance(brain, Worker):
        result['worker_sha256'] = digest(Path(brain.proc.args[0]))
        result['disconnected'] = brain.disconnected
    return result


def capture(arena, brain, environment, organism, model, decision, desire, vision, initial_body, continuation):
    desire_state = copy.deepcopy(vars(desire))
    desire_state['progress'] = list(desire_state['progress'])
    neural = brain.export_state() if isinstance(brain, Worker) else {name:getattr(brain,name).copy() for name in FIELDS}
    validate_state(neural, brain.flies)
    return {'arena':arena.get_state(), 'neural':neural, 'environment':environment.get_state(),
            'organism':organism.get_state(), 'readout':{name:getattr(model,name).copy() for name in ('mean','scale','weights')},
            'decision':copy.deepcopy(vars(decision)), 'desire':desire_state,
            'vision_previous':None if vision.previous is None else vision.previous.copy(),
            'vision_registration':vision.registration(),
            'initial_body':copy.deepcopy(initial_body), 'continuation':copy.deepcopy(continuation)}


def save(path, state, identity):
    if Path(path).exists(): raise ValueError('checkpoint already exists')
    save_checkpoint(path, {'payload':state,'payload_sha256':tree_hash(state)}, identity)
    return {'checkpoint':str(Path(path).resolve()),'checkpoint_sha256':digest(path),
            'payload_sha256':tree_hash(state)}


def load(path, identity):
    envelope, _ = load_checkpoint(path, identity)
    if set(envelope) != {'payload','payload_sha256'}: raise ValueError('invalid checkpoint envelope')
    if tree_hash(envelope['payload']) != envelope['payload_sha256']: raise ValueError('checkpoint payload checksum mismatch')
    state = envelope['payload']
    if set(state) != {'arena','neural','environment','organism','readout','decision','desire',
                      'vision_previous','vision_registration','initial_body','continuation'}: raise ValueError('checkpoint fields mismatch')
    validate_state(state['neural'], identity['neural_flies'])
    image = state['vision_previous']
    if image is not None and (image.ndim != 3 or image.shape[0] != 2 or image.dtype != np.float32 or not np.isfinite(image).all()):
        raise ValueError('invalid visual history')
    registration=state['vision_registration']
    map_key=str((ASSETS/'spatial-vision.npz').relative_to(ROOT))
    if not isinstance(registration,dict) or registration.get('spatial') is not True or registration.get('mapping_sha256')!=identity['assets'][map_key]:
        raise ValueError('visual registration identity mismatch')
    for name in ('horizontal_flip','vertical_flip'):
        flags=registration.get(name)
        if not isinstance(flags,list) or len(flags)!=2 or any(type(flag) is not bool for flag in flags):
            raise ValueError('invalid image axis registration')
    for name, shape in (('mean',(2606,)),('scale',(2606,)),('weights',(2607,5))):
        value = state['readout'][name]
        if value.shape != shape or not np.isfinite(value).all(): raise ValueError('invalid decoder ' + name)
    if np.any(state['readout']['scale'] <= 0): raise ValueError('invalid decoder scale')
    continuation = state['continuation']
    if set(continuation) != {'t','exchange','pulse_end','pulse','sweep_start','sweep','reward_updates','resets',
                              'last','action','direct_mode','attempted','direct_batches','blind'}:
        raise ValueError('invalid continuation fields')
    for key in ('t','pulse_end','sweep_start'):
        if not isinstance(continuation[key],(int,float)) or not np.isfinite(continuation[key]) or continuation[key]<0:
            raise ValueError('invalid continuation clock')
    for key in ('exchange','reward_updates','resets','direct_batches'):
        if type(continuation[key]) is not int or continuation[key]<0: raise ValueError('invalid continuation counter')
    for key in ('sweep','direct_mode','blind'):
        if type(continuation[key]) is not bool: raise ValueError('invalid continuation flag')
    pulse = continuation['pulse']
    if pulse.shape != (8,) or pulse.dtype != np.float32 or not np.isfinite(pulse).all() or np.any((pulse<0)|(pulse>1)):
        raise ValueError('invalid pending sensory pulse')
    attempted = continuation['attempted']
    if not isinstance(attempted,list) or len(set(attempted))!=len(attempted) or any(type(i) is not int or not 0<=i<139255 for i in attempted):
        raise ValueError('invalid direct-target history')
    last = continuation['last']
    if last is not None:
        features = np.asarray(last['features'])
        counts = np.asarray(last['counts'])
        if features.shape != (identity['neural_flies'],2606) or counts.shape != (identity['neural_flies'],2) or not np.isfinite(features).all() or not np.isfinite(counts).all():
            raise ValueError('invalid prior neural result')
    return state


def restore(state, arena, brain, environment, organism, model, decision, desire, vision):
    # Startup only: any restore error aborts this run before READY or exchanges.
    arena.set_state(state['arena'])
    environment.set_state(state['environment'])
    organism.set_state(state['organism'])
    for name, value in state['readout'].items(): setattr(model,name,value.copy())
    decision.__dict__.clear(); decision.__dict__.update(copy.deepcopy(state['decision']))
    desire.__dict__.clear(); desire.__dict__.update(copy.deepcopy(state['desire']))
    desire.progress = deque(desire.progress)
    vision.previous = None if state['vision_previous'] is None else state['vision_previous'].copy()
    registration=state['vision_registration']
    if registration['mapping_sha256']!=vision.mapping['sha256'] or not registration['spatial']:
        raise ValueError('visual registration identity mismatch')
    vision.horizontal_flip=tuple(registration['horizontal_flip']);vision.vertical_flip=tuple(registration['vertical_flip'])
    vision.camera_axes=registration['camera_axes']
    if isinstance(brain, Worker): brain.import_state(state['neural'])
    else:
        for name,value in state['neural'].items(): setattr(brain,name,value.copy())
    return copy.deepcopy(state['initial_body']), copy.deepcopy(state['continuation'])
