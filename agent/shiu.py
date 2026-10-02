"""Shiu et al. 2024 whole-brain LIF (Nature), CPU reference.

Equations and parameters from the authors' Brian2 code (eonsystemspbc/fly-brain,
code/run_brian2_cuda.py), integrated exactly ('linear' method) at dt = 0.1 ms:

    dv/dt = (v_0 - v + g) / t_mbr      (unless refractory)
    dg/dt = -g / tau                   (unless refractory)
    spike: v > v_th  ->  v = v_0, g = 0, refractory t_rfc
    on presynaptic spike, after t_dly:  g += w_syn * signed synapse count
    stimulus: Poisson events add w_syn * f_poi to v; stimulated cells have no refractory period
    input arriving at a refractory cell is dropped: with it, sugar-GRN spike counts match the
    published Brian2/GeNN/PyTorch runs; accumulating it instead gave +35% network spikes.

State is held as u = v - v_0 (mV). Per step, in Brian's schedule order: integrate the
non-refractory cells, threshold, deliver spikes from t_dly ago and the Poisson events,
then reset. Delivery is push-based (only presynaptic cells that spiked), so a step costs
the out-edges of the active cells, not the whole graph.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

V_TH, T_MBR, TAU, T_RFC, T_DLY, W_SYN, F_POI = 7.0, 20e-3, 5e-3, 2.2e-3, 1.8e-3, 0.275, 250


class ShiuBrain:
    def __init__(self, pre, post, synapses, n, flies=1, dt=1e-4, seed=0):
        """pre/post/synapses: one row per connection, synapses signed (excitatory x count)."""
        self.n, self.flies, self.dt = n, flies, dt
        self.out = csr_matrix((np.asarray(synapses, np.float32) * np.float32(W_SYN),
                               (np.asarray(pre), np.asarray(post))), shape=(n, n))
        self.em, self.eg = np.float32(np.exp(-dt / T_MBR)), np.float32(np.exp(-dt / TAU))
        self.k = np.float32(TAU / (TAU - T_MBR))
        self.delay, self.refractory = round(T_DLY / dt), round(T_RFC / dt)
        self.rng = np.random.default_rng(seed)
        self.rate = np.zeros((n, flies), np.float32)  # Poisson stimulus rate per cell, Hz
        self.reset()

    def reset(self):
        n, c = self.n, self.flies
        self.u, self.g = np.zeros((n, c), np.float32), np.zeros((n, c), np.float32)
        self.last = np.full((n, c), -10**9, np.int64)
        self.ring = np.zeros((self.delay, n, c), bool)
        self.t, self.spikes, self.ever = 0, np.zeros((n, c), np.int64), np.zeros((n, c), bool)

    def stimulate(self, cells, rate, fly=None):
        """Poisson input at `rate` Hz on `cells` (all flies unless one is named)."""
        self.rate[cells, slice(None) if fly is None else fly] = rate

    def step(self, steps=1):
        stim = self.rate > 0
        for _ in range(steps):
            # Brian updates not_refractory in the threshold slot, so the integrator sees last step's flag
            moving = (self.t - 1 - self.last >= self.refractory) | stim
            free = (self.t - self.last >= self.refractory) | stim
            u0, g0 = self.u, self.g
            self.u = np.where(moving, u0 * self.em + g0 * self.k * (self.eg - self.em), u0)
            self.g = np.where(moving, g0 * self.eg, g0)
            spike = free & (self.u > V_TH)
            arriving = self.ring[self.t % self.delay]
            for fly in range(self.flies):
                active = np.flatnonzero(arriving[:, fly])
                if active.size:
                    start, end = self.out.indptr[active], self.out.indptr[active + 1]
                    edges = np.repeat(end - (end - start).cumsum(), end - start) + np.arange((end - start).sum())
                    arrived = np.bincount(self.out.indices[edges], self.out.data[edges], minlength=self.n)
                    self.g[:, fly] += arrived.astype(np.float32) * free[:, fly]  # refractory cells drop input
            events = stim & (self.rng.random(stim.shape) < self.rate * self.dt)
            self.u[events] += np.float32(W_SYN * F_POI)
            self.u[spike], self.g[spike], self.last[spike] = 0, 0, self.t
            self.ring[self.t % self.delay] = spike
            self.spikes += spike
            self.ever |= spike
            self.t += 1
        return self.spikes
