"""Dopamine-gated plasticity on Kenyon cell -> MBON synapses of the FlyWire graph.

Port of the rule in ASHR12/FlySweeper `flysweeper/plasticity.py` (MIT licence), adapted to the
CPUBrain here: the graph is stored row=postsynaptic, column=presynaptic, and eligibility is
accumulated once per body exchange from the brain's spike traces rather than per LIF tick.

    e      <- e * exp(-dt/elig_tau) + trace[KC] * trace[MBON]       (per KC->MBON edge)
    reward (sign>0) or punishment (sign<0):
    w      <- clip(w + eta * sign * e * |w0|, w_min_ratio*|w0|, w_max_ratio*|w0|)

Only KC->MBON edges with positive weight change; the rest of the connectome stays frozen. Every
dopamine event is logged so a "the fly learned" claim can be checked against weight changes.
Not ported: FlySweeper also injects a voltage kick into PAM/PPL1 cells per event. CPUBrain has
no general drive hook, so this module changes weights only; the cell sets are kept for that use.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from agent.neural_connectome import ANNOTATIONS, N, ROOT


@dataclass
class PlasticityParams:
    eta: float = 0.05
    elig_tau: float = 1.0          # seconds
    exchange_s: float = 0.01       # physical seconds per body exchange (manifest)
    w_min_ratio: float = 0.1
    w_max_ratio: float = 3.0


def mushroom_body_cells(annotations=ANNOTATIONS):
    """Node indices of KC, MBON, PAM and PPL1 cells, by exact int root-ID lookup."""
    roots = np.load(ROOT / 'data/proofread_root_ids_783.npy')
    index = {int(root): i for i, root in enumerate(roots)}
    cells = {'KC': [], 'MBON': [], 'PAM': [], 'PPL1': []}
    for row in csv.DictReader(Path(annotations).open(), delimiter='\t'):
        node = index.get(int(row['root_id']))
        if node is None:
            continue
        cell_type = row['cell_type'] or ''
        if row['cell_class'] == 'Kenyon_Cell':
            cells['KC'].append(node)
        elif row['cell_class'] == 'MBON':
            cells['MBON'].append(node)
        elif cell_type.startswith('PAM'):
            cells['PAM'].append(node)
        elif cell_type.startswith('PPL1'):
            cells['PPL1'].append(node)
    return {name: np.unique(np.asarray(nodes, dtype=np.int64)) for name, nodes in cells.items()}


class MBPlasticity:
    def __init__(self, brain, params: PlasticityParams | None = None, fly: int = 0):
        self.p = params or PlasticityParams()
        self.brain, self.fly = brain, fly
        self.cells = mushroom_body_cells()
        g = brain.graph
        is_kc = np.zeros(N, bool); is_kc[self.cells['KC']] = True
        is_mbon = np.zeros(N, bool); is_mbon[self.cells['MBON']] = True
        post = np.repeat(np.arange(N, dtype=np.int64), np.diff(g.indptr))
        pre = g.indices.astype(np.int64)
        block = is_mbon[post] & is_kc[pre]
        self.excluded_nonpositive = int((block & (g.data <= 0)).sum())
        self.pos = np.flatnonzero(block & (g.data > 0))
        self.post, self.pre = post[self.pos], pre[self.pos]
        self.w0 = g.data[self.pos].astype(np.float32)
        self.w_frozen = self.w0.copy()
        self.w_learned = self.w0.copy()
        self.attached = True
        self.e = np.zeros(len(self.pos), np.float32)
        self.decay = np.float32(np.exp(-self.p.exchange_s / self.p.elig_tau))
        self.log: list[dict] = []
        self.total_abs_delta = 0.0
        self.exchanges = 0

    def observe(self):
        """Call once after each brain.step: accumulate KC x MBON coactivity on each edge."""
        trace = self.brain.trace[:, self.fly]
        self.e *= self.decay
        self.e += trace[self.pre] * trace[self.post]
        self.exchanges += 1

    def dopamine(self, sign: float, note: str = ''):
        data = self.brain.graph.data
        w = data[self.pos]
        new = np.clip(w + self.p.eta * sign * self.e * self.w0,
                      self.p.w_min_ratio * self.w0, self.p.w_max_ratio * self.w0)
        delta = new - w
        data[self.pos] = new
        abs_delta = float(np.abs(delta).sum())
        self.total_abs_delta += abs_delta
        self.log.append({'exchange': self.exchanges, 'sign': float(sign), 'note': note,
                         'edges_changed': int((delta != 0).sum()), 'abs_delta': abs_delta,
                         'mean_ratio': float((new / self.w0).mean())})

    def detach(self):
        """Restore frozen KC->MBON weights; keep the learned ones aside."""
        if self.attached:
            self.w_learned = self.brain.graph.data[self.pos].copy()
            self.brain.graph.data[self.pos] = self.w_frozen
            self.attached = False

    def attach(self):
        if not self.attached:
            self.brain.graph.data[self.pos] = self.w_learned
            self.attached = True

    def summary(self):
        w = self.brain.graph.data[self.pos] if self.attached else self.w_learned
        ratio = w / self.w0
        return {'cells': {k: int(len(v)) for k, v in self.cells.items()},
                'kc_mbon_edges': int(len(self.pos)),
                'kc_mbon_edges_excluded_nonpositive': self.excluded_nonpositive,
                'events': len(self.log), 'total_abs_delta': self.total_abs_delta,
                'edges_changed': int((w != self.w_frozen).sum()), 'attached': self.attached,
                'mean_weight_ratio': float(ratio.mean()), 'min_weight_ratio': float(ratio.min()),
                'max_weight_ratio': float(ratio.max()),
                'edges_at_floor': int((ratio <= self.p.w_min_ratio + 1e-6).sum()),
                'edges_at_ceiling': int((ratio >= self.p.w_max_ratio - 1e-6).sum()),
                'params': asdict(self.p)}

    def write_log(self, path):
        """Weight-change log: one row per dopamine event plus the end-state summary."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'events': self.log, 'summary': self.summary()}, indent=2) + '\n')
