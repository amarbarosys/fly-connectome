"""C. elegans graded-potential brain (Wicks 1996, Kunert 2014) on the Cook 2019 wiring.

    C dV/dt = -Gc (V - Ec) - Gg sum_j gap_ij (V_i - V_j) - Gs sum_j chem_ij s_j (V_i - E_j) + I
    ds/dt   = ar sigmoid(beta (V - Vth)) (1 - s) - ad s

E_j = -48 mV when the presynaptic neuron is one of the 26 GABAergic neurons (McIntire 1993), else 0 mV.
Units: mV, pS, pF, seconds. Neurons = cells with a chemical output row in Cook 2019 (muscles dropped).

Hot path is one step for every arm at once: arms are stacked (K, N, N) float32 and each step is
two batched matmuls into preallocated buffers. Gap junctions are stiff (fastest mode 76k/s), so leak +
gap are backward Euler through a precomputed inverse M = (I + dt (Gc - L))^-1, chemical synapses
explicit: [chem; chem * E] @ s, then V = M @ (V + dt (Gc Ec + I + syn)). Stable at dt = 0.1 ms.
Dense beats CSR here: 300 x 300 float32 = 360 KB per matrix, cache resident.
"""
from __future__ import annotations

import numpy as np

from agent.shiu import ROOT

C, GC, EC, GG, GS, AR, AD, BETA, VTH, E_INH = 1.0, 10.0, -35.0, 100.0, 100.0, 1.0, 5.0, 0.125, -35.0, -48.0
DATA = ROOT / 'data/worm'


GABA = {f'DD{i:02d}' for i in range(1, 7)} | {f'VD{i:02d}' for i in range(1, 14)} | {
    'RMED', 'RMEV', 'RMEL', 'RMER', 'RIS', 'AVL', 'DVB'}


def load():
    """Neuron names, chem and gap edge lists (pre, post, count) over neuron indices, reversal per neuron."""
    import sys
    sys.path.insert(0, str(ROOT / 'tools'))
    from build_worm_csr import read_csr
    nodes = (DATA / 'worm_nodes.txt').read_text().split()
    csr = {kind: read_csr(DATA / f'worm_{kind}_csr.bin')[1:] for kind in ('chemical', 'gap')}
    keep = np.zeros(len(nodes), bool)
    keep[csr['chemical'][1]] = True
    index = np.cumsum(keep) - 1
    names = [n for n, k in zip(nodes, keep) if k]
    edges = {}
    for kind, (row, col, w) in csr.items():
        post = np.repeat(np.arange(len(nodes)), np.diff(row))
        m = keep[col] & keep[post] & (col != post)
        edges[kind] = index[col[m]], index[post[m]], w[m]
    return names, edges['chemical'], edges['gap'], np.where([n in GABA for n in names], E_INH, 0.0)


def dense(n, pre, post, w):
    a = np.zeros((n, n), np.float32)
    np.add.at(a, (post, pre), w)
    return a


class WormBrain:
    def __init__(self, chems, gaps, rev, batch, dt=1e-4):
        """chems, gaps: per arm (N, N) post x pre counts; gaps symmetrised here. batch: columns per arm."""
        k, n = len(chems), chems[0].shape[0]
        eye = np.eye(n)
        self.m = np.stack([np.linalg.inv(eye * (1 + dt * GC / C) - dt / C * GG * (lap(g))) for g in gaps]).astype(np.float32)
        w = np.stack(chems) * np.float32(GS)
        self.syn = np.concatenate([w, w * rev.astype(np.float32)], 1)  # (K, 2N, N)
        self.v = np.full((k, n, batch), EC, np.float32)
        self.s = np.zeros((k, n, batch), np.float32)
        self.x, self.ss = np.empty_like(self.v), np.empty((k, 2 * n, batch), np.float32)
        self.dt, self.n = np.float32(dt), n

    def step(self, current):
        """current: (K or 1, N, B) fA (pS x mV), so dV/dt comes out in mV/s."""
        n, v, s, x, dt = self.n, self.v, self.s, self.x, self.dt
        np.matmul(self.syn, s, out=self.ss)
        s += dt * (AR / (1 + np.exp(-BETA * (v - VTH))) * (1 - s) - AD * s)
        np.multiply(v, self.ss[:, :n], out=x)
        np.subtract(self.ss[:, n:], x, out=x)
        x += current
        x += GC * EC
        x *= dt / C
        x += v
        np.matmul(self.m, x, out=v)

    def run(self, current, seconds):
        for _ in range(round(seconds / self.dt)):
            self.step(current)
        return self.v


def lap(g):
    """Symmetrised gap counts as a graph Laplacian with sign: L @ V = sum_j g_ij (V_j - V_i)."""
    g = (g + g.T) / 2
    return g - np.diag(g.sum(1))
