"""Shiu et al. 2024 whole-brain LIF (Nature): CPU oracle (ShiuBrain) and GPU engine (ShiuGPU).

Equations and parameters from the authors' Brian2 code (eonsystemspbc/fly-brain,
code/run_brian2_cuda.py), integrated exactly ('linear' method) at dt = 0.1 ms:

    dv/dt = (v_0 - v + g) / t_mbr      (unless refractory)
    dg/dt = -g / tau                   (unless refractory)
    spike: v > v_th  ->  v = v_0, g = 0, refractory t_rfc
    on presynaptic spike, after t_dly:  g += w_syn * signed synapse count
    stimulus: Poisson events add w_syn * f_poi to v; stimulated cells have no refractory period
    input arriving at a refractory cell is dropped: with it, sugar-GRN spike counts match the
    published Brian2/GeNN/PyTorch runs; accumulating it instead gave +35% network spikes.

State is u = v - v_0 (mV), indexed column-major (column * n + neuron). Per step t, the order
both engines use: integrate the moving cells; threshold and reset (spikers get u = g = 0);
deliver the spikes of step t - delay to cells that are free and did not spike at t; Poisson
events on stimulated cells that did not spike at t. Delivery is push-based (out-edges of the
cells that spiked), and Poisson draws use a counter hash of (seed, t, entry) so the CPU and
GPU engines see the same events.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import numpy as np

V_TH, T_MBR, TAU, T_RFC, T_DLY, W_SYN, F_POI = 7.0, 20e-3, 5e-3, 2.2e-3, 1.8e-3, 0.275, 250
ROOT = Path(__file__).resolve().parents[1]
GRAPHS = ROOT / '.work/agent/shiu/graphs'
# The MAX device context pre-reserves a memory pool, by default most of the card: 22.6 GiB VRAM
# for a 4 GB job, so a second worker ran out of memory. 5% reserves 1.2 GiB and allocates the
# rest on demand: 4.8 GiB peak, sniff time unchanged (1,774 vs 1,776 ms).
WORKER_ENV = {**os.environ, 'MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT':
              os.environ.get('MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT', '5')}


def uniform(seed, t, entry):
    """Counter hash (murmur3 fmix32) of (seed, step, entry) to [0, 1), float32; GPU twin in shiu_worker."""
    with np.errstate(over='ignore'):  # uint32 arithmetic wraps by design
        h = (np.uint32(seed) * np.uint32(0x9E3779B9) + np.uint32(t) * np.uint32(0x85EBCA6B)
             + entry.astype(np.uint32) * np.uint32(0xC2B2AE35))
        h ^= h >> np.uint32(16); h *= np.uint32(0x85EBCA6B)
        h ^= h >> np.uint32(13); h *= np.uint32(0xC2B2AE35)
        h ^= h >> np.uint32(16)
    return (h >> np.uint32(8)).astype(np.float32) * np.float32(2.0 ** -24)


def constants(dt=1e-4):
    """float32 update factors shared by both engines: u' = u*em + g*c, g' = g*eg."""
    em, eg = np.float32(np.exp(-dt / T_MBR)), np.float32(np.exp(-dt / TAU))
    return em, eg, np.float32(TAU / (TAU - T_MBR) * (float(eg) - float(em)))


def graph_dir(pre, post, synapses, n):
    """Out-edge (pre-major) CSR in mV, written once per content hash; graph-null arms are dirs.
    int32 indices, no sort when pre is already sorted, arrays hashed in place: the 11-arm
    combined graph (166M edges) peaked at 13 GiB host RSS with int64, argsort and a byte copy."""
    pre, post = np.asarray(pre, np.int32), np.asarray(post, np.int32)
    weights = np.asarray(synapses, np.float32) * np.float32(W_SYN)
    if (np.diff(pre) < 0).any():
        order = np.argsort(pre, kind='stable')
        pre, post, weights = pre[order], post[order], weights[order]
    row = np.zeros(n + 1, np.int32)
    np.cumsum(np.bincount(pre, minlength=n), out=row[1:])
    digest = hashlib.sha256()
    for array in (row, post, weights):
        digest.update(memoryview(np.ascontiguousarray(array)).cast('B'))
    path = GRAPHS / f'{n}-{len(post)}-{digest.hexdigest()[:16]}'
    if not (path / 'weights.bin').is_file():
        path.mkdir(parents=True, exist_ok=True)
        for name, array in (('row_ptr', row), ('col_idx', post), ('weights', weights)):
            array.tofile(path / f'{name}.bin')
    return path


class ShiuBrain:
    def __init__(self, pre, post, synapses, n, flies=1, dt=1e-4, seed=0):
        """pre/post/synapses: one row per connection, synapses signed (excitatory x count)."""
        self.graph = graph_dir(pre, post, synapses, n)
        self.n, self.flies, self.dt, self.seed = n, flies, dt, seed
        self.row = np.fromfile(self.graph / 'row_ptr.bin', np.int32)
        self.col = np.fromfile(self.graph / 'col_idx.bin', np.int32)
        self.w = np.fromfile(self.graph / 'weights.bin', np.float32)
        self.em, self.eg, self.c = constants(dt)
        self.delay, self.refractory = round(T_DLY / dt), round(T_RFC / dt)
        self.stim = {}  # entry (column * n + neuron) -> Poisson probability per step
        self.reset()

    def reset(self):
        size = self.n * self.flies
        self.u, self.g = np.zeros(size, np.float32), np.zeros(size, np.float32)
        self.last = np.full(size, -10**9, np.int64)
        self.ring = [np.zeros(0, np.int64)] * self.delay
        self.t, self.counts = 0, np.zeros(size, np.int32)

    def stimulate(self, cells, rate, fly=None):
        """Poisson input at `rate` Hz on `cells` (all flies unless one is named)."""
        for column in range(self.flies) if fly is None else [fly]:
            for cell in np.asarray(cells, np.int64):
                self.stim[int(column * self.n + cell)] = np.float32(rate * self.dt)

    def _stim_arrays(self):
        entries = np.fromiter(self.stim, np.int64, len(self.stim))
        return entries, np.fromiter(self.stim.values(), np.float32, len(self.stim))

    def step(self, steps=1):
        entries, p = self._stim_arrays()
        refractory = np.full(self.n * self.flies, self.refractory, np.int64)
        refractory[entries] = 0
        for _ in range(steps):
            t = self.t
            moving = t - 1 - self.last >= refractory
            self.u = np.where(moving, self.u * self.em + self.g * self.c, self.u)
            self.g = np.where(moving, self.g * self.eg, self.g)
            spike = np.flatnonzero((t - self.last >= refractory) & (self.u > V_TH))
            self.u[spike], self.g[spike], self.last[spike] = 0, 0, t
            self.counts[spike] += 1
            arriving = self.ring[t % self.delay]
            if arriving.size:
                column, pre = np.divmod(arriving, self.n)
                start, end = self.row[pre].astype(np.int64), self.row[pre + 1].astype(np.int64)
                edges = np.repeat(end - (end - start).cumsum(), end - start) + np.arange(int((end - start).sum()))
                target = np.repeat(column, end - start) * self.n + self.col[edges]
                keep = (t - self.last[target] >= refractory[target]) & (self.last[target] != t)
                self.g += np.bincount(target[keep], self.w[edges][keep], minlength=self.g.size).astype(np.float32)
            self.ring[t % self.delay] = spike
            hit = entries[(uniform(self.seed, t, entries) < p) & (self.last[entries] != t)]
            self.u[hit] += np.float32(W_SYN * F_POI)
            self.t += 1

    def run(self, windows):
        """Counts (n, flies) at each step count in `windows` (ascending), from the current state."""
        out = []
        for stop in windows:
            self.step(stop - self.t)
            out.append(self.counts.reshape(self.flies, self.n).T.copy())
        return out

    @property
    def spikes(self):
        return self.counts.reshape(self.flies, self.n).T

    @property
    def ever(self):
        return self.spikes > 0


class ShiuGPU(ShiuBrain):
    """Same model on the GPU (agent/shiu_worker.mojo); run() is one process per call from t = 0."""
    def step(self, steps=1):
        raise NotImplementedError('ShiuGPU runs whole windows from t = 0: use run()')

    def run(self, windows):
        work = ROOT / '.work/agent/shiu/run'
        work.mkdir(parents=True, exist_ok=True)
        entries, p = self._stim_arrays()
        entries.astype(np.int32).tofile(work / 'stim_index.bin')
        p.tofile(work / 'stim_p.bin')
        np.asarray(windows, np.int32).tofile(work / 'windows.bin')
        np.asarray([self.em, self.eg, self.c, V_TH, W_SYN * F_POI], np.float32).tofile(work / 'consts.bin')
        subprocess.run([str(worker_binary(self.n, len(self.col), self.flies)), str(self.graph) + '/', str(work) + '/',
                        str(self.seed), str(self.delay), str(self.refractory)], check=True, env=WORKER_ENV)
        counts = np.fromfile(work / 'counts.bin', np.int32).reshape(len(windows), self.flies, self.n)
        self.t, self.counts = int(windows[-1]), counts[-1].reshape(-1).copy()
        return [c.T.copy() for c in counts]


def worker_binary(n, edges, flies):
    """shiu_worker built for this graph size and column count; rebuilt when the source is newer."""
    source = ROOT / 'agent/shiu_worker.mojo'
    binary = ROOT / f'.work/agent/shiu/build/shiu_worker_{n}_{edges}_c{flies}'
    if not binary.is_file() or binary.stat().st_mtime < source.stat().st_mtime:
        binary.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(Path.home() / '.local/bin/mojo'), 'build', f'-DSHIU_N={n}', f'-DSHIU_E={edges}',
                        f'-DSHIU_C={flies}', str(source), '-o', str(binary)], check=True, cwd=ROOT)
    return binary


class ShiuLive(ShiuGPU):
    """Resident ShiuGPU: graph uploaded once; each sniff(p) resets the brain, runs to the last
    window with Poisson probabilities p (per stimulate() entry, in insertion order) and returns
    counts (windows, len(readout)) for the readout entries only. Sniff k uses seed + k."""
    def start(self, windows, readout):
        self.work = ROOT / f'.work/agent/shiu/live-{id(self)}'
        self.work.mkdir(parents=True, exist_ok=True)
        entries, p = self._stim_arrays()
        entries.astype(np.int32).tofile(self.work / 'stim_index.bin')
        p.tofile(self.work / 'stim_p.bin')
        self.windows, self.readout = np.asarray(windows, np.int32), np.asarray(readout, np.int32)
        self.windows.tofile(self.work / 'windows.bin')
        self.readout.tofile(self.work / 'readout.bin')
        np.asarray([self.em, self.eg, self.c, V_TH, W_SYN * F_POI], np.float32).tofile(self.work / 'consts.bin')
        self.proc = subprocess.Popen([str(worker_binary(self.n, len(self.col), self.flies)), str(self.graph) + '/',
                                      str(self.work) + '/', str(self.seed), str(self.delay), str(self.refractory), 'resident'],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1, env=WORKER_ENV)
        return self

    def sniff(self, p):
        np.asarray(p, np.float32).tofile(self.work / 'stim_p.bin')
        self.proc.stdin.write('sniff\n')
        self.proc.stdin.flush()
        if self.proc.stdout.readline().strip() != 'ok':
            raise RuntimeError(f'FAIL shiu_worker resident exited {self.proc.wait()}')
        return np.fromfile(self.work / 'gather.bin', np.int32).reshape(len(self.windows), len(self.readout))

    def close(self):
        self.proc.stdin.write('q\n')
        self.proc.stdin.close()
        self.proc.wait()
