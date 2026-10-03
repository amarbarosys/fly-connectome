"""One spec for Shiu experiments: topologies x weight variants x stimuli x repeats, read by cell group.

Weight variants of a topology are columns of one graph (edge groups, agent/verify_groups.py), so the
graph is uploaded once per topology instead of once per variant; several topologies (shuffle arms)
sit side by side as today. Column (v * S + s) * R + r runs variant v, stimulus s, repeat r. Groups
are summed on the device; groups named in `per_cell` also return their cells' own counts.

    Experiment(topologies=[(pre, post, w)], n=n,
               edges={'alln': lambda pre, post, w: (cells.klass[pre] == 'ALLN') & (w < 0)},
               variants=[{}, {'alln': 12}], stimuli={'odour': (orn_ids, 200.)}, repeats=8,
               windows=(300, 2000), groups={'kc': kc_ids, 'dn': dn_ids}, per_cell=('kc',)).run()

A variant may carry 'hz', which replaces every stimulus rate for its columns.
"""
import csv
from dataclasses import dataclass, field

import numpy as np

from agent.neural_connectome import ANNOTATIONS, ROOT
from agent.shiu import ShiuGPU
from agent.verify_shiu import _reference_graph


def cells():
    """FlyWire annotation columns in graph order (class = cell_class, else super_class), parsed once into .work."""
    path = ROOT / '.work/agent/shiu/cells-flywire.npz'
    if not path.is_file():
        index = _reference_graph()[3]
        table = {k: np.full(len(index), '', dtype=object) for k in ('klass', 'type', 'side', 'super')}
        for r in csv.DictReader(open(ANNOTATIONS), delimiter='\t'):
            i = index.get(int(r['root_id']))
            if i is not None:
                for k, col in (('klass', 'cell_class'), ('type', 'cell_type'), ('side', 'side'), ('super', 'super_class')):
                    table[k][i] = r[col]
        table['klass'] = np.where(table['klass'] == '', table['super'], table['klass'])
        np.savez(path, **{k: v.astype(str) for k, v in table.items()})
    z = np.load(path)
    return type('Cells', (), {k: z[k] for k in z.files})


@dataclass
class Experiment:
    topologies: list                      # [(pre, post, signed synapses)]
    n: int
    stimuli: dict                         # name -> (cell ids, Hz)
    windows: tuple
    groups: dict                          # name -> cell ids
    edges: dict = field(default_factory=dict)       # name -> mask(pre, post, w) over edges
    variants: list = field(default_factory=lambda: [{}])
    repeats: int = 1
    per_cell: tuple = ()
    seed: int = 61

    def run(self):
        self.topologies = [g if not (np.diff(g[0]) < 0).any() else [np.asarray(x)[np.argsort(g[0], kind='stable')] for x in g]
                           for g in self.topologies]  # CSR order: edge groups must follow it
        T, V, S, R, W = len(self.topologies), len(self.variants), len(self.stimuli), self.repeats, len(self.windows)
        cols, names = V * S * R, list(self.edges)
        grp = np.concatenate([self._edge_groups(*g) for g in self.topologies])
        scale = np.ones((V, len(names) + 1), np.float32)
        for v, variant in enumerate(self.variants):
            for k, name in enumerate(names):
                scale[v, k + 1] = variant.get(name, 1)
        pre, post, w = (np.concatenate([np.asarray(g[i]) + (t * self.n if i < 2 else 0) for t, g in enumerate(self.topologies)])
                        for i in range(3))
        brain = ShiuGPU(pre, post, w, n=T * self.n, flies=cols, seed=self.seed)
        for v, variant in enumerate(self.variants):
            for s, (ids, hz) in enumerate(self.stimuli.values()):
                for r in range(R):
                    for t in range(T):
                        brain.stimulate(np.asarray(ids) + t * self.n, variant.get('hz', hz), fly=(v * S + s) * R + r)
        brain.edge_groups(grp, np.repeat(scale, S * R, axis=0))
        cgrp = np.full(T * self.n, -1, np.int32)
        for t in range(T):
            for g, ids in enumerate(self.groups.values()):
                cgrp[np.asarray(ids) + t * self.n] = t * len(self.groups) + g
        each = [np.asarray(self.groups[name]) for name in self.per_cell]
        flat = np.concatenate([np.zeros(0, np.int64), *each])
        readout = (np.arange(cols)[:, None, None] * T * self.n + np.arange(T)[None, :, None] * self.n + flat[None, None]).reshape(-1)
        sums, got = brain.run_groups(list(self.windows), cgrp, readout)
        out = {'sums': {name: sums[:, g::len(self.groups)].reshape(W, T, V, S, R) for g, name in enumerate(self.groups)}}
        got = got.reshape(W, cols, T, len(flat)).transpose(0, 3, 2, 1).reshape(W, len(flat), T, V, S, R)
        bounds = np.cumsum([0] + [len(e) for e in each])
        out['cells'] = {name: got[:, bounds[k]:bounds[k + 1]] for k, name in enumerate(self.per_cell)}
        return out

    def _edge_groups(self, pre, post, w):
        grp = np.zeros(len(pre), np.uint8)
        for k, mask in enumerate(self.edges.values()):
            grp[mask(pre, post, w)] = k + 1  # later rules win where they overlap
        return grp
