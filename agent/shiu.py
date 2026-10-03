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
KMAX = 8192  # plastic KCs per column, shiu_worker's SHIU_K
KEY = 'MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT'


def worker_env():
    """The MAX device context pre-reserves a pool, by default most of the card (22.6 GiB for a 4 GB job, so
    a second worker ran out of memory). Workers allocate once, so no pool: pool 0 holds exactly the
    worker's own buffers, at the same sniff time (391 ms vs 393 for a sized pool, 96 columns). An outer
    setting wins."""
    return {**os.environ, KEY: os.environ.get(KEY, '0')}


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
        self.drive = {}  # entry -> (amount added to u per event, t0, t1 steps); absent = Poisson kick, always on
        self.key = {}  # entry -> entry whose Poisson draws it uses (common random numbers)
        self.grp, self.scale, self.plastic_groups = None, None, []
        self.modp = None
        self.reset()

    def edge_groups(self, grp, scale):
        """Weight variants as columns: edge e (CSR order of self.col) delivers w[e] x scale[column, grp[e]].
        grp uint8 per edge, scale float32 (flies, G); unset = every weight unchanged."""
        self.grp, self.scale = np.asarray(grp, np.uint8), np.asarray(scale, np.float32).reshape(self.flies, -1)
        assert len(self.grp) == len(self.col) and int(self.grp.max(initial=0)) < self.scale.shape[1] <= 256

    def modulate(self, plane, tau, k, b=(0.0, 0.0, 0.0)):
        """Slow neuromodulation: edges of group g with plane[g] >= 0 (a dopamine, octopamine or serotonin pre)
        add to the target's level m[plane] instead of its conductance; m decays with tau[plane] seconds and the
        target integrates g x c x max(0, 1 + sum k m) + c x sum b m (b: slow current per unit level). plane: int per edge group (len G, -1 = fast); <= 3 planes."""
        assert self.grp is not None and len(plane) == self.scale.shape[1] and max(plane) < 3
        self.modp = np.asarray(plane, np.int32)
        P = int(self.modp.max()) + 1
        self.mdecay = np.float32(np.exp(-self.dt / np.asarray(tau, np.float64)))[:P].astype(np.float32)
        self.mk = np.asarray(k, np.float32)[:P]
        self.mb = np.asarray(b, np.float32)[:P]
        self.reset()

    def modulators(self, nt, tau=(0.5, 1.0, 2.0), k=(0.0, 0.0, 0.0), b=(0.0, 0.0, 0.0)):
        """Route every edge whose pre neuron's predicted transmitter (FlyWire top_nt, experiment.cells().nt) is
        dopamine, octopamine or serotonin into planes 0, 1, 2 (three new edge groups, scale 1, appended after the
        existing ones), then modulate(). tau in seconds per plane, k the gain per unit level."""
        pre = np.repeat(np.arange(self.n), np.diff(self.row))
        q = np.select([nt == 'dopamine', nt == 'octopamine', nt == 'serotonin'], [0, 1, 2], -1)[pre]
        grp = np.zeros(len(self.col), np.uint8) if self.grp is None else self.grp
        scale = np.ones((self.flies, 1), np.float32) if self.scale is None else self.scale
        G = scale.shape[1]
        self.edge_groups(np.where(q >= 0, G + q, grp), np.hstack([scale, np.ones((self.flies, 3), np.float32)]))
        self.modulate(np.r_[np.full(G, -1), 0, 1, 2], tau, k, b)
        return q

    def plastic(self, group, kcs, punish=None):
        """Learnable edges: every edge of edge group `group` (pre always one of `kcs`) is scaled by its
        column's factor for that KC, initially 1, changed only by reward(); factors survive reset().
        `punish` names a second group (KC -> approach MBONs) with its own factors, the second plane:
        factor is (flies, P x KCs), plane p in columns p x KCs onward."""
        self.plastic_groups, self.kcs = [int(group)] + ([] if punish is None else [int(punish)]), np.asarray(kcs, np.int64)
        self.rank = np.zeros(self.n, np.int64)
        self.rank[self.kcs] = np.arange(len(self.kcs))
        self.factor = np.ones((self.flies, len(self.plastic_groups) * len(self.kcs)), np.float32)

    def reward(self, dopamine, punishment=None):
        """Scale the factor of every KC that spiked since the last reset by 1 - dopamine[column] (reward
        plane) and by 1 - punishment[column] (punish plane)."""
        spiked, k = self.spikes[self.kcs].T > 0, len(self.kcs)
        for plane, d in enumerate([dopamine] + ([] if punishment is None else [punishment])):
            f = self.factor[:, plane * k:(plane + 1) * k]
            f[spiked] *= np.repeat(1 - np.asarray(d, np.float32), spiked.sum(1))

    def _write_groups(self, work):
        for name in ('grp.bin', 'scale.bin', 'plastic.bin', 'punish.bin', 'factor.bin', 'modp.bin', 'modk.bin'):
            (work / name).unlink(missing_ok=True)
        if self.modp is not None:
            self.modp.tofile(work / 'modp.bin')
            mk = np.zeros(9, np.float32)
            P = len(self.mk)
            mk[:P], mk[3:3 + P], mk[6:6 + P] = self.mdecay, self.mk, self.mb
            mk.tofile(work / 'modk.bin')
        if self.grp is not None:
            self.grp.tofile(work / 'grp.bin')
            self.scale.tofile(work / 'scale.bin')
        if self.plastic_groups:
            np.concatenate([self.plastic_groups[:1], self.kcs]).astype(np.int32).tofile(work / 'plastic.bin')
            if len(self.plastic_groups) > 1:
                np.int32(self.plastic_groups[1:]).tofile(work / 'punish.bin')
            P, k = len(self.plastic_groups), len(self.kcs)
            padded = np.ones((P, self.flies, KMAX), np.float32)
            padded[:, :, :k] = self.factor.reshape(self.flies, P, k).transpose(1, 0, 2)
            padded.tofile(work / 'factor.bin')

    def reset(self):
        size = self.n * self.flies
        self.u, self.g = np.zeros(size, np.float32), np.zeros(size, np.float32)
        self.last = np.full(size, -10**9, np.int64)
        self.ring = [np.zeros(0, np.int64)] * self.delay
        self.t, self.counts = 0, np.zeros(size, np.int32)
        self.m = np.zeros((0 if self.modp is None else len(self.mk), size), np.float32)

    def stimulate(self, cells, rate, fly=None, key=None, amount=None, window=None, p=None):
        """Poisson input at `rate` Hz on `cells` (all flies unless one is named); with `key`, the
        named fly's events are drawn as column `key`'s would be, so columns can share inputs.
        amount: u added per event (default the Poisson kick W_SYN x F_POI, which also drops the cell's
        refractory period); window: (t0, t1) steps it is on; p: probability per step instead of rate x dt
        (p = 1 with a small amount is a constant injected current, see inject)."""
        for column in range(self.flies) if fly is None else [fly]:
            for cell in np.asarray(cells, np.int64):
                e = int(column * self.n + cell)
                self.stim[e] = np.float32(rate * self.dt if p is None else p)
                self.key[e] = int((column if key is None else key) * self.n + cell)
                if amount is not None or window is not None:
                    self.drive[e] = (np.float32(W_SYN * F_POI if amount is None else amount), *(window or (0, 2**31 - 1)))
                else:
                    self.drive.pop(e, None)

    def inject(self, cells, current, fly=None, window=None):
        """Constant current on `cells`: u += current (mV per ms) x dt every step it is not refractory."""
        self.stimulate(cells, 0, fly=fly, amount=current * self.dt * 1e3, window=window, p=1.0)

    def _write_drive(self, work, entries):
        """stim_amount.bin + stim_window.bin when any entry is not a plain always-on Poisson kick, else neither."""
        for name in ('stim_amount.bin', 'stim_window.bin'):
            (work / name).unlink(missing_ok=True)
        if self.drive:
            amount, window = self._drive_arrays(entries)
            amount.tofile(work / 'stim_amount.bin')
            window.tofile(work / 'stim_window.bin')

    def _drive_arrays(self, entries):
        """(amount, t0, t1) per entry, in entry order: the worker's stim_amount.bin / stim_window.bin."""
        d = [self.drive.get(int(e), (np.float32(W_SYN * F_POI), 0, 2**31 - 1)) for e in entries]
        return np.array([x[0] for x in d], np.float32), np.array([x[1:] for x in d], np.int32).reshape(-1, 2)

    def _stim_arrays(self):
        entries = np.fromiter(self.stim, np.int64, len(self.stim))
        return entries, np.fromiter(self.stim.values(), np.float32, len(self.stim))

    def _stim_keys(self, entries):
        return np.fromiter((self.key[int(e)] for e in entries), np.int64, len(entries))

    def step(self, steps=1):
        entries, p = self._stim_arrays()
        keys = self._stim_keys(entries)
        amount, window = self._drive_arrays(entries)
        refractory = np.full(self.n * self.flies, self.refractory, np.int64)
        refractory[entries[amount == np.float32(W_SYN * F_POI)]] = 0
        for _ in range(steps):
            t = self.t
            moving = t - 1 - self.last >= refractory
            if len(self.m):  # the worker's order: gain from the levels before this step's decay
                acc, drive = np.zeros(self.u.size, np.float32), np.zeros(self.u.size, np.float32)
                for q in range(len(self.m)):
                    acc += self.mk[q] * self.m[q]
                    drive += self.mb[q] * self.m[q]
                    self.m[q] *= self.mdecay[q]
                gain = np.maximum(np.float32(1) + acc, np.float32(0))
                self.u = np.where(moving, self.u * self.em + self.g * self.c * gain + drive * self.c, self.u)
            else:
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
                w = self.w[edges]
                if self.grp is not None:
                    w = w * self.scale[np.repeat(column, end - start), self.grp[edges]]
                for plane, group in enumerate(self.plastic_groups):
                    learn = self.grp[edges] == group
                    w[learn] *= self.factor[np.repeat(column, end - start)[learn], plane * len(self.kcs) + self.rank[np.repeat(pre, end - start)[learn]]]
                if len(self.m):  # modulatory edges feed the slow levels (every arrival, refractory or not)
                    q = self.modp[self.grp[edges]]
                    for plane in range(len(self.m)):
                        self.m[plane] += np.bincount(target[q == plane], w[q == plane], minlength=self.g.size).astype(np.float32)
                    keep &= q < 0
                self.g += np.bincount(target[keep], w[keep], minlength=self.g.size).astype(np.float32)
            self.ring[t % self.delay] = spike
            on = (uniform(self.seed, t, keys) < p) & (self.last[entries] != t) & (window[:, 0] <= t) & (t < window[:, 1])
            np.add.at(self.u, entries[on], amount[on])
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
        counts = self._launch(windows).reshape(len(windows), self.flies, self.n)
        self.t, self.counts = int(windows[-1]), counts[-1].reshape(-1).copy()
        return [c.T.copy() for c in counts]

    def run_groups(self, windows, cgrp, readout=()):
        """Spikes summed per cell group on the device: (windows, G, flies) int32 for cgrp (group per
        neuron, -1 none), and per-cell counts (windows, len(readout)) for readout entries
        (column * n + neuron); the full N x C counts never leave the device."""
        cgrp = np.asarray(cgrp, np.int32)
        groups = int(cgrp.max()) + 1
        sums, got = self._launch(windows, cgrp, np.asarray(readout, np.int32))
        return sums.reshape(len(windows), groups, self.flies), got.reshape(len(windows), -1)

    def _launch(self, windows, cgrp=None, readout=None):
        work = ROOT / '.work/agent/shiu/run'
        work.mkdir(parents=True, exist_ok=True)
        for name in ('cgrp.bin', 'readout.bin', 'sums.bin', 'gather.bin', 'counts.bin'):
            (work / name).unlink(missing_ok=True)
        if cgrp is not None:
            cgrp.tofile(work / 'cgrp.bin')
            if len(readout):
                readout.tofile(work / 'readout.bin')
        entries, p = self._stim_arrays()
        entries.astype(np.int32).tofile(work / 'stim_index.bin')
        self._stim_keys(entries).astype(np.int32).tofile(work / 'stim_key.bin')
        p.tofile(work / 'stim_p.bin')
        self._write_drive(work, entries)
        np.asarray(windows, np.int32).tofile(work / 'windows.bin')
        self._write_groups(work)
        np.asarray([self.em, self.eg, self.c, V_TH, W_SYN * F_POI], np.float32).tofile(work / 'consts.bin')
        subprocess.run([str(worker_binary(self.n, len(self.col), self.flies)), str(self.graph) + '/', str(work) + '/',
                        str(self.seed), str(self.delay), str(self.refractory)], check=True, env=worker_env())
        if cgrp is None:
            return np.fromfile(work / 'counts.bin', np.int32)
        got = np.fromfile(work / 'gather.bin', np.int32) if len(readout) else np.zeros(0, np.int32)
        return np.fromfile(work / 'sums.bin', np.int32), got


def worker_binary(n, edges, flies, source=ROOT / 'agent/shiu_worker.mojo'):
    """shiu_worker built for this graph size and column count; rebuilt when the source is newer.
    Another source (a checkpoint's own kernel) builds beside itself."""
    out = ROOT / '.work/agent/shiu/build' if source == ROOT / 'agent/shiu_worker.mojo' else source.parent / 'build'
    binary = out / f'shiu_worker_{n}_{edges}_c{flies}'
    if not binary.is_file() or binary.stat().st_mtime < source.stat().st_mtime:
        binary.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(Path.home() / '.local/bin/mojo'), 'build', f'-DSHIU_N={n}', f'-DSHIU_E={edges}',
                        f'-DSHIU_C={flies}', str(source), '-o', str(binary)], check=True, cwd=ROOT)
    return binary


class ShiuLive(ShiuGPU):
    """Resident ShiuGPU: graph uploaded once; each sniff(p) resets the brain, runs to the last
    window with Poisson probabilities p (per stimulate() entry, in insertion order) and returns
    counts (windows, len(readout)) for the readout entries only. Sniff k uses seed + k. record=path writes
    every neuron's counts at every window there; read it back with recording()."""
    def start(self, windows, readout, record=None):
        self.work = ROOT / f'.work/agent/shiu/live-{id(self)}'
        self.work.mkdir(parents=True, exist_ok=True)
        entries, p = self._stim_arrays()
        entries.astype(np.int32).tofile(self.work / 'stim_index.bin')
        self._stim_keys(entries).astype(np.int32).tofile(self.work / 'stim_key.bin')
        p.tofile(self.work / 'stim_p.bin')
        self._write_drive(self.work, entries)
        self.windows, self.readout = np.asarray(windows, np.int32), np.asarray(readout, np.int32)
        self.windows.tofile(self.work / 'windows.bin')
        self.readout.tofile(self.work / 'readout.bin')
        self._write_groups(self.work)
        np.asarray([self.em, self.eg, self.c, V_TH, W_SYN * F_POI], np.float32).tofile(self.work / 'consts.bin')
        self.proc = subprocess.Popen([str(worker_binary(self.n, len(self.col), self.flies, *self.source)), str(self.graph) + '/',
                                      str(self.work) + '/', str(self.seed), str(self.delay), str(self.refractory), 'resident'],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
                                     env=worker_env() | ({'FLY_RECORD': str(record)} if record else {}))
        self.record = record
        return self

    def recording(self):
        """The full brain so far: int32 (sniffs, windows, n, flies) spike counts since each sniff began."""
        r = np.fromfile(self.record, np.int32)
        return r.reshape(-1, len(self.windows), self.flies, self.n).transpose(0, 1, 3, 2)

    source = ()  # (path,) builds the worker from that kernel source instead of agent/

    def factors(self):
        """The learned per-KC factors (flies, KCs), read back from the device."""
        self._command('save')
        P, k = len(self.plastic_groups), len(self.kcs)
        self.factor = np.fromfile(self.work / 'factor.bin', np.float32).reshape(P, self.flies, KMAX)[:, :, :k].transpose(1, 0, 2).reshape(self.flies, P * k).copy()
        return self.factor

    def _ok(self):
        """The worker answers 'ok'; anything else (its error goes to stdout) is raised with its text."""
        line = self.proc.stdout.readline().strip()
        if line != 'ok':
            raise RuntimeError(f'FAIL shiu_worker resident exited {self.proc.wait()}: {line} {self.proc.stdout.read().strip()}'.strip())

    def _command(self, line):
        self.proc.stdin.write(line + '\n')
        self.proc.stdin.flush()
        self._ok()

    def reward(self, dopamine, punishment=None):
        """Device twin of ShiuBrain.reward for the last sniff; the factors stay on the device."""
        np.concatenate([dopamine] + ([] if punishment is None else [punishment])).astype(np.float32).tofile(self.work / 'dopamine.bin')
        self.proc.stdin.write('reward\n')
        self.proc.stdin.flush()
        self._ok()

    def push(self):
        """Send changed settings to the running worker; it reads them at the next sniff: the scale row (same
        groups) and the drive amounts / windows / probabilities of the entries given at start."""
        if self.scale is not None:
            self.scale.tofile(self.work / 'scale.bin')
        entries, p = self._stim_arrays()
        p.tofile(self.work / 'stim_p.bin')
        self._write_drive(self.work, entries)
        (self.work / 'pushed').touch()  # the worker rereads only after a push
        return p

    def sniff(self, p):
        np.asarray(p, np.float32).tofile(self.work / 'stim_p.bin')
        self.proc.stdin.write('sniff\n')
        self.proc.stdin.flush()
        self._ok()
        return np.fromfile(self.work / 'gather.bin', np.int32).reshape(len(self.windows), len(self.readout))

    def close(self):
        self.proc.stdin.write('q\n')
        self.proc.stdin.close()
        self.proc.wait()
