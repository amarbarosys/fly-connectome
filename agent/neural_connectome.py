"""Provenance-bound sensory ports, full-graph CPU oracle and learned readout.

The sensory assignments and body readout are experimental interfaces. Anatomical
class/side labels locate ports; they do not establish natural motor semantics.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import sysconfig
import time

import numpy as np
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / '.work/neural-flies/assets'
ANNOTATIONS = ROOT / 'data/annotations/flyconnectome-flywire_annotations-a83b277/supplemental_files/Supplemental_file1_neuron_annotations.tsv'
N = 139255
E = 10572269
CHANNELS = ('odor_left', 'odor_right', 'taste', 'touch', 'looming_left',
            'looming_right', 'rival_left', 'rival_right')
OUTPUTS = ('walk_left', 'walk_right', 'feed', 'fence', 'retreat')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def graph_arrays(directory=ROOT / 'data'):
    row = np.fromfile(directory / 'row_ptr.bin', '<i4')
    col = np.fromfile(directory / 'col_idx.bin', '<i4')
    weights = np.fromfile(directory / 'weights.bin', '<f4')
    if (row.shape != (N + 1,) or col.shape != (E,) or weights.shape != (E,)
        or row[0] != 0 or row[-1] != E or np.any(np.diff(row) < 0)
        or np.any((col < 0) | (col >= N)) or not np.isfinite(weights).all()):
        raise ValueError('CSR graph validation failed')
    return row, col, weights


def normalize_weights_inplace(indptr, weights):
    """Normalize CSR weights by absolute postsynaptic row sum without edge-sized scratch arrays."""
    for row in range(len(indptr) - 1):
        start, end = int(indptr[row]), int(indptr[row + 1])
        if end > start:
            total = np.float32(np.abs(weights[start:end]).sum(dtype=np.float64))
            if total:
                weights[start:end] /= total


def prepare():
    ASSETS.mkdir(parents=True, exist_ok=True)
    roots = np.load(ROOT / 'data/proofread_root_ids_783.npy')
    if roots.shape != (N,) or len(np.unique(roots)) != N:
        raise ValueError('invalid root-ID roster')
    index = {int(root): i for i, root in enumerate(roots)}
    ports = np.zeros((N, len(CHANNELS)), dtype=np.uint8)
    descending = []
    rows = list(csv.DictReader(ANNOTATIONS.open(), delimiter='\t'))
    for row in rows:
        node = index.get(int(row['root_id']))
        if node is None:
            continue
        side = row['side']
        cls, super_cls = row['cell_class'], row['super_class']
        if super_cls == 'descending':
            descending.append(node)
        if super_cls in ('sensory', 'sensory_ascending'):
            if cls == 'olfactory' and side in ('left', 'right'):
                ports[node, 0 if side == 'left' else 1] = 1
            elif cls == 'gustatory':
                ports[node, 2] = 1
            elif cls == 'mechanosensory':
                ports[node, 3] = 1
        # Lamina injection bypasses uncertain photoreceptor sign/relay dynamics.
        # Looming and rival-bearing encodings are engineered visual proxies.
        if row['cell_type'] in ('L1', 'L2', 'L3', 'LC4', 'LPLC2') and side in ('left', 'right'):
            offset = 0 if side == 'left' else 1
            channel = 4 + offset if row['cell_type'] in ('LC4', 'LPLC2') else 6 + offset
            ports[node, channel] = 1
    descending = np.unique(descending).astype('<i4')
    if len(descending) != 1303 or np.any(ports.sum(axis=0) == 0):
        raise ValueError('missing or unexpected anatomical port population')
    ports.tofile(ASSETS / 'ports.bin')
    descending.tofile(ASSETS / 'descending.bin')
    np.savez(ASSETS / 'ports.npz', ports=ports, descending=descending)
    row, col, weights = graph_arrays()
    paths = [ROOT/'data'/name for name in ('row_ptr.bin', 'col_idx.bin', 'weights.bin', 'proofread_root_ids_783.npy')]
    paths += [ANNOTATIONS, ASSETS/'ports.bin', ASSETS/'descending.bin']
    meta = {'neurons': N, 'edges': E, 'flies': 2, 'channels': CHANNELS,
            'outputs': OUTPUTS, 'descending_neurons': len(descending),
            'port_sizes': dict(zip(CHANNELS, ports.sum(axis=0).tolist())),
            'mapping_status': 'experimental_class_and_side_encoding_not_validated_sensory_or_motor_functions',
            'dynamics': {'decay_per_tick': .85, 'coupling': .08, 'threshold': 1., 'voltage_floor': -5., 'trace_decay': .85,
                         'ticks_per_body_exchange': 4, 'physical_exchange_s': .01,
                         'physical_neural_timestep_calibration': 'unvalidated'},
            'graph_orientation': 'row=postsynaptic, column=presynaptic',
            'weight_semantics': 'existing signed weights from build_csr.py; not measured effective synaptic strength',
            'input_hashes': {str(p.relative_to(ROOT)): digest(p) for p in paths}}
    (ASSETS/'manifest.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(json.dumps(meta['port_sizes']))
    return meta


class CPUBrain:
    """Independent two-fly reference; whole graph remains present."""
    def __init__(self, *, disconnected=False, flies=2, adaptation_da=0.95, adaptation_ka=0.0,
                 gain_mode='raw', gain=1.0):
        if type(flies) is not int or not 1 <= flies <= 64:
            raise ValueError('flies must be an integer in [1,64]')
        if not 0 <= adaptation_da <= 1 or adaptation_ka < 0:
            raise ValueError('invalid spike-frequency adaptation parameters')
        if gain_mode not in ('raw', 'global', 'normalized') or gain < 0:
            raise ValueError('invalid recurrent gain')
        self.gain_mode, self.gain = gain_mode, np.float32(gain)
        self.adaptation_da, self.adaptation_ka = np.float32(adaptation_da), np.float32(adaptation_ka)
        self.flies = flies
        arrays = np.load(ASSETS/'ports.npz')
        self.ports = arrays['ports'].astype(np.float32)
        self.descending = arrays['descending']
        row, col, weights = graph_arrays()
        if disconnected:
            weights[:] = 0
        self.graph = csr_matrix((weights, col, row), shape=(N, N))
        self.raw_weights = None if gain_mode == 'normalized' else self.graph.data
        self.normalized_weights = None
        if gain_mode == 'normalized':
            normalize_weights_inplace(self.graph.indptr, self.graph.data)
            self.normalized_weights = self.graph.data
        self.sparse = None  # optional (nodes, k): at most k of those nodes spike per tick
        self.reset()

    def load_columns(self, directory):
        """Swap in another arm's column indices in place; row pointers and weights are shared."""
        col = self.graph.indices
        assert col.dtype == np.int32 and col.flags.c_contiguous
        with open(Path(directory) / 'col_idx.bin', 'rb') as stream:
            if stream.readinto(memoryview(col).cast('B')) != col.nbytes or stream.read(1):
                raise ValueError(f'{directory}: col_idx.bin size mismatch')
        if col.min() < 0 or col.max() >= N:
            raise ValueError(f'{directory}: column index out of range')
        self.graph.has_sorted_indices = False

    def reset(self):
        self.v = np.zeros((N, self.flies), np.float32)
        self.spikes = np.zeros_like(self.v)
        self.trace = np.zeros_like(self.v)
        self.a = np.zeros_like(self.v)
        self.ever = np.zeros((N, self.flies), bool)

    def step(self, inputs, ticks=4, direct=(), after=0, visual=None, extra_drive=None,
             measure_activity=False):
        inputs = np.asarray(inputs, np.float32).reshape(self.flies, 8)
        if not np.isfinite(inputs).all() or np.any((inputs < 0) | (inputs > 1)):
            raise ValueError('invalid sensory input')
        # Neural drive is bounded; graph dynamics reproduce the existing model.
        drive = (self.ports @ inputs.T) * np.float32(.65)
        if visual is not None:
            from agent.spatial_vision import mapping, validate_visual
            values=validate_visual(visual,self.flies)
            drive[mapping()['nodes']]+=np.float32(.65)*values.T
        direct=np.asarray(direct,dtype=np.int64)
        if len(direct)>256 or np.any((direct<0)|(direct>=N)):raise ValueError('invalid direct targets')
        if len(direct):drive[direct[~self.ever[direct,0]],0]+=np.float32(.65)
        if extra_drive is not None:
            extra_drive=np.asarray(extra_drive,np.float32).reshape(N,self.flies)
            if not np.isfinite(extra_drive).all() or np.any((extra_drive<0)|(extra_drive>1)):raise ValueError('invalid extra drive')
            drive=drive+np.float32(.65)*extra_drive
        tick_activity = []
        for _ in range(ticks):
            if self.gain_mode == 'normalized':
                recurrent = self.gain * (self.graph @ self.spikes)
            elif self.gain_mode == 'global':
                recurrent = self.gain * (self.graph @ self.spikes)
            else:
                recurrent = self.graph @ self.spikes
            v = np.float32(.85)*self.v + np.float32(.08)*recurrent + drive
            self.spikes = (v - self.a >= 1).astype(np.float32)
            if self.sparse is not None:
                nodes, k = self.sparse
                # engineered stand-in for APL feedback inhibition: keep the k most depolarised
                sub = np.where(self.spikes[nodes] != 0, v[nodes], -np.inf)
                cut = np.argpartition(-sub, k, axis=0)[k:]
                block = self.spikes[nodes]
                np.put_along_axis(block, cut, 0, axis=0)
                self.spikes[nodes] = block
            self.v = np.where(self.spikes != 0, 0, np.maximum(v, -5)).astype(np.float32)
            self.a = self.adaptation_da*self.a + self.adaptation_ka*self.spikes
            self.trace = np.float32(.85)*self.trace + np.float32(.15)*self.spikes
            self.ever |= self.spikes != 0
            if measure_activity:
                tick_activity.append((self.spikes != 0).mean(axis=0))
        features = np.concatenate((self.v[self.descending].T, self.trace[self.descending].T), axis=1)
        result = {'features': features.tolist(), 'counts': np.column_stack((self.spikes.sum(axis=0), self.ever.sum(axis=0))).tolist(), 'inactive':(np.flatnonzero(~self.ever[after:,0])+after)[:256].tolist()}
        if measure_activity:
            result['tick_active_fraction'] = np.mean(tick_activity, axis=0).tolist()
        return result


class Worker:
    """Resident neural process with bounded request waits and explicit failure."""
    def __init__(self, binary=None, *, disconnected=False, flies=2):
        if type(flies) is not int or not 1 <= flies <= 10:
            raise ValueError('flies must be an integer in [1,10]')
        self.flies = flies
        binary = binary or os.environ.get('FLY_NEURAL_WORKER_BINARY') or ROOT/'.work/neural-flies/neural_worker'
        self.disconnected = disconnected
        env = dict(os.environ, FLY_NEURAL_ASSETS=str(ASSETS),
                   MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT='2',
                   PYTHONHOME=sys.base_prefix,
                   MOJO_PYTHON_LIBRARY=str(Path(sysconfig.get_config_var('LIBDIR'))/'libpython3.12.so'),
                   PYTHONPATH=os.pathsep.join((str(ROOT), sysconfig.get_path('purelib'))))
        env['FLY_NEURAL_DISCONNECT'] = '1' if disconnected else '0'
        self.proc = subprocess.Popen([str(binary)], cwd=ROOT, env=env, text=True,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(ROOT/'.work/neural-flies/worker.stderr', 'a'), bufsize=1)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.proc.stdout, selectors.EVENT_READ)
        try:
            self.ready = self.receive(timeout=90)
            if self.ready.get('neurons') != N or self.ready.get('flies') != flies:
                raise RuntimeError('neural worker readiness mismatch')
        except BaseException:
            self.close()
            raise

    def receive(self, timeout=30):
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            if not self.selector.select(max(0., deadline-time.monotonic())):
                break
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(f'neural worker exited: {self.proc.poll()}')
            if line.startswith('{'):
                return json.loads(line)
        raise TimeoutError('neural worker response deadline exceeded')

    def request(self, payload):
        self.proc.stdin.write(json.dumps(payload, allow_nan=False)+'\n')
        self.proc.stdin.flush()
        return self.receive()

    def reset(self):
        return self.request({'op': 'reset'})

    def export_state(self):
        return self._transfer_state()

    def import_state(self, state):
        self._transfer_state(state)

    def _transfer_state(self, state=None):
        from agent.neural_io import state_identity, validate_state
        import tempfile
        if not self.ready.get('state_transfer'): raise RuntimeError('worker needs rebuild for state transfer')
        directory = ROOT/'.work/neural-flies/worker-state'
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=directory) as temporary:
            path = Path(temporary)/'state.npz'
            if state is not None:
                validate_state(state, self.flies)
                # Worker environment may differ from its Python client's environment.
                expected = dict(state_identity(self.flies), disconnected=self.disconnected)
                with path.open('xb') as stream:
                    np.savez_compressed(stream, identity=np.array(json.dumps(expected)), **state)
            reply = self.request({'op':'import_state' if state is not None else 'export_state','path':str(path)})
            if reply != {'state_transfer':True}: raise RuntimeError('state transfer acknowledgement mismatch')
            if state is None:
                with np.load(path, allow_pickle=False) as archive:
                    result = {name:archive[name].copy() for name in ('v','spikes','trace','ever')}
                validate_state(result, self.flies)
                return result

    def step(self, inputs, ticks=4, direct=(), after=0, visual=None):
        if ticks != 4:
            raise ValueError('GPU exchange is exactly four neural ticks')
        payload={'op': 'step', 'inputs': np.asarray(inputs).reshape(-1).tolist(),'direct':list(direct),'after':after}
        if visual is not None:
            from agent.spatial_vision import mapping, validate_visual
            if not self.ready.get('spatial_drive') or self.ready.get('spatial_mapping_sha256')!=mapping()['sha256']:raise RuntimeError('worker spatial mapping mismatch; rebuild/restart required')
            payload['visual']=validate_visual(visual,self.flies).reshape(-1).tolist()
        return self.request(payload)

    def close(self):
        if self.proc.poll() is None:
            try:
                self.proc.stdin.write('{"op":"quit"}\n'); self.proc.stdin.flush()
                self.proc.wait(timeout=5)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                self.proc.terminate()
                try: self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired: self.proc.kill(); self.proc.wait()
        self.selector.close()


def teacher(inputs):
    """Declared imitation teacher; local senses only, not food coordinates."""
    x = np.asarray(inputs)
    odor = x[:, 0]+x[:, 1]
    rival = x[:, 6]+x[:, 7]
    turn = .6*(x[:, 1]-x[:, 0])+.4*(x[:, 7]-x[:, 6])
    retreat = np.maximum(x[:, 4], x[:, 5])
    speed = .35+.35*odor+.15*rival-.9*retreat
    return np.column_stack((np.clip(speed+turn, -1.2, 1.2),
                            np.clip(speed-turn, -1.2, 1.2), x[:, 2],
                            x[:, 3]*np.minimum(rival, 1), retreat))


class Readout:
    def __init__(self, path=ASSETS/'readout.npz'):
        values = np.load(path)
        self.mean, self.scale, self.weights = values['mean'], values['scale'], values['weights']
        if self.mean.shape!=(2606,) or self.scale.shape!=(2606,) or self.weights.shape!=(2607,5):
            raise ValueError('invalid motor readout dimensions')
        if not all(np.isfinite(x).all() for x in (self.mean,self.scale,self.weights)) or np.any(self.scale<=0):
            raise ValueError('non-finite or invalid motor readout')

    def predict(self, features):
        f = np.asarray(features, dtype=np.float64)
        output = np.column_stack(((f-self.mean)/self.scale, np.ones(len(f)))) @ self.weights
        # Fail closed for no neural signal, including the disconnected graph.
        output[np.max(np.abs(f), axis=1) < 1e-8] = 0
        output[:, :2] = np.clip(output[:, :2], -1.2, 1.2)
        output[:, 2:] = np.clip(output[:, 2:], 0, 1)
        if not np.isfinite(output).all():
            raise RuntimeError('non-finite motor readout')
        return output


    def reinforce(self, features, action, feedback):
        """Bounded decoder plasticity; does not change anatomical graph weights."""
        if not np.isfinite(feedback) or not -1 <= feedback <= 1:
            raise ValueError('feedback must be finite and in [-1,1]')
        f = np.asarray(features, np.float64)
        if f.shape != self.mean.shape or np.max(np.abs(f)) < 1e-8:
            return 0.
        z = np.r_[(f-self.mean)/self.scale, 1.]
        before = self.predict(f[None])[0]
        target = before.copy()
        channels = {'FEED': [2], 'FENCE': [3], 'WALK': [0,1]}.get(action, [])
        for channel in channels:
            if channel < 2:
                target[channel] = np.clip(before[channel]*(1+.2*feedback), -1.2,1.2)
            else:
                target[channel] = 1. if feedback>0 else 0.
        delta = .03*abs(feedback)*np.outer(z,target-before)/(1.+z@z)
        self.weights += delta
        if not np.isfinite(self.weights).all():
            raise RuntimeError('non-finite decoder plasticity')
        return float(np.linalg.norm(delta))

    def save(self, path):
        np.savez(path, mean=self.mean, scale=self.scale, weights=self.weights)


def collect(brain, inputs):
    features = []
    for pair in inputs:
        brain.reset()
        for _ in range(8):
            result = brain.step(pair)
        features.append(result['features'])
    return np.asarray(features).reshape(-1, 2606)


def train(backend='cpu', pairs=80):
    rng = np.random.default_rng(20260930)
    train_inputs = rng.uniform(.0, .8, (pairs, 2, 8)).astype(np.float32)
    test_inputs = rng.uniform(.0, .8, (24, 2, 8)).astype(np.float32)
    brain = Worker() if backend == 'gpu' else CPUBrain()
    try:
        x = collect(brain, train_inputs)
        y = teacher(train_inputs.reshape(-1, 8))
        mean, scale = x.mean(axis=0), np.maximum(x.std(axis=0), .025)
        z = np.column_stack(((x-mean)/scale, np.ones(len(x))))
        # Dual ridge is small even when every descending voltage/trace is used.
        regularization = 2.
        weights = z.T @ np.linalg.solve(z@z.T+regularization*np.eye(len(z)), y)
        np.savez(ASSETS/'readout.npz', mean=mean, scale=scale, weights=weights)
        held_out = collect(brain, test_inputs)
        predictor = Readout()
        predicted = predictor.predict(held_out)
        targets = teacher(test_inputs.reshape(-1, 8))
        shuffled = predictor.predict(held_out[:, rng.permutation(held_out.shape[1])])
        null = predictor.predict(np.zeros_like(held_out))
        baseline = np.repeat(y.mean(axis=0)[None], len(targets), axis=0)
        report = {'backend': backend, 'training_pairs': pairs, 'held_out_pairs': 24,
                  'learning': 'supervised_ridge_imitation_of_declared_local_sensory_teacher',
                  'connectome_weights_trained': False, 'neural_readout_features': 'all 1303 descending voltages and traces',
                  'rmse': float(np.sqrt(np.mean((predicted-targets)**2))),
                  'constant_rmse': float(np.sqrt(np.mean((baseline-targets)**2))),
                  'shuffled_feature_rmse': float(np.sqrt(np.mean((shuffled-targets)**2))),
                  'null_signal_rmse': float(np.sqrt(np.mean((null-targets)**2))),
                  'readout_sha256': digest(ASSETS/'readout.npz'),
                  'source_hashes': {str(p.relative_to(ROOT)):digest(p) for p in (Path(__file__), ROOT/'agent/neural_worker.mojo')},
                  'input_manifest_sha256':digest(ASSETS/'manifest.json'),
                  'neural_worker_sha256':digest(ROOT/'.work/neural-flies/neural_worker'),
                  'status': 'MODEL_INTERFACE_EXPERIMENT_NOT_BIOLOGICAL_LEARNING'}
        (ASSETS/'training.json').write_text(json.dumps(report, indent=2)+'\n')
        np.savez(ASSETS/'held-out.npz', inputs=test_inputs, features=held_out, targets=targets, predicted=predicted)
        print(json.dumps(report, indent=2), flush=True)
        return report
    finally:
        if isinstance(brain, Worker): brain.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'train'))
    parser.add_argument('--backend', choices=('cpu', 'gpu'), default='cpu')
    parser.add_argument('--pairs', type=int, default=80)
    args = parser.parse_args()
    prepare() if args.command == 'prepare' else train(args.backend, args.pairs)
