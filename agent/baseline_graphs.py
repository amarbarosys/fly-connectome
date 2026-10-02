"""Build matched control graphs for the connectome-benefit test, and prove what each preserves.

The question G1 asks is whether the real wiring beats a network of the same size that carries no
anatomy. Three arms share everything except the edge list:

  real     the FlyWire CSR as built
  shuffled col_idx globally permuted: both degree sequences and the weight-position pairing are
           preserved exactly, only which neuron connects to which is destroyed. This is the
           configuration-model control and the one that matters.
  block    col_idx permuted only among edges with the same (post cell_class, pre cell_class):
           both degree sequences and the class-to-class edge counts survive, wiring within a
           class pair does not. Separates "class-level structure" from "neuron-level wiring".
  random   col_idx drawn uniformly: in-degree per neuron and the weight multiset survive,
           out-degree structure does not. A weaker control, kept because it separates "the
           specific wiring matters" from "any structured out-degree matters".

Sensory ports and the descending population are derived from root IDs, not topology, so they are
identical across arms. Nothing here trains or runs physics.
"""
import csv
import json
from pathlib import Path
import time
import numpy as np
from agent.neural_connectome import ROOT, N, E, ANNOTATIONS, graph_arrays, digest

OUT = ROOT/'.work/agent/matched-baseline'
ARMS = ('real', 'shuffled', 'block', 'random')


def degrees(row, col):
    """In-degree per postsynaptic row and out-degree per presynaptic column."""
    return np.diff(row).astype(np.int64), np.bincount(col, minlength=N).astype(np.int64)


def cell_classes():
    """cell_class per node as a small int; exact int root-ID keys (IDs exceed 2**53)."""
    roots = np.load(ROOT/'data/proofread_root_ids_783.npy')
    index = {int(root): i for i, root in enumerate(roots)}
    names = np.full(N, 'unannotated', dtype=object)
    for row in csv.DictReader(ANNOTATIONS.open(), delimiter='\t'):
        node = index.get(int(row['root_id']))
        if node is not None:
            names[node] = row['cell_class'] or row['super_class'] or 'unannotated'
    return np.unique(names, return_inverse=True)[1].astype(np.int64)


def block_keys(row, col, klass):
    rows = np.repeat(np.arange(len(row) - 1, dtype=np.int64), np.diff(row))
    return klass[rows]*(klass.max()+1) + klass[col]


def block_shuffle(row, col, klass, generator):
    """Permute col within each (post class, pre class) block: one argsort, one permutation per block."""
    key = block_keys(row, col, klass)
    order = np.argsort(key, kind='stable')
    starts = np.flatnonzero(np.r_[True, key[order][1:] != key[order][:-1]])
    out = col.copy()
    for start, end in zip(starts, np.r_[starts[1:], len(col)]):
        positions = order[start:end]
        out[positions] = col[positions][generator.permutation(end-start)]
    return out


def repair_self_loops(row, col, generator, key=None, original=None):
    """Swap self-loops against random other edges. The source graph has none and a permutation
    creates some, so without this the control would differ from the real graph in a way that is
    not about wiring."""
    rows = np.repeat(np.arange(len(row) - 1, dtype=np.int64), np.diff(row))
    for _ in range(64):
        loops = np.flatnonzero(rows == col)
        if loops.size == 0:
            return col, int(_)
        if key is None:
            partners = generator.integers(0, len(col), size=loops.size)
        else:  # swap only inside the loop's own block, so block counts stay exact
            order = np.argsort(key, kind='stable')
            sorted_key = key[order]
            lo = np.searchsorted(sorted_key, key[loops], 'left')
            hi = np.searchsorted(sorted_key, key[loops], 'right')
            partners = order[lo + (generator.random(loops.size)*(hi-lo)).astype(np.int64)]
        # a swap is only taken when it removes the loop without creating another
        keep = (col[partners] != rows[loops]) & (col[loops] != rows[partners])
        if key is not None:  # small blocks reuse partners; the global path never did (receipts)
            keep &= ~np.isin(partners, loops)
            unique = np.zeros(loops.size, bool)
            unique[np.unique(partners, return_index=True)[1]] = True
            keep &= unique
        take, give = loops[keep], partners[keep]
        col[take], col[give] = col[give].copy(), col[take].copy()
    if key is not None:  # rigid blocks (e.g. two neurons wired to each other) admit only the real layout
        rigid = np.isin(key, key[rows == col])
        col[rigid] = original[rigid]
        if not (rows == col).any():
            return col, -int(np.unique(key[rigid]).size)
    raise RuntimeError('FAIL repair: self-loops remain after 64 passes')


def build(seed=20261002):
    row, col, weights = graph_arrays()
    in_real, out_real = degrees(row, col)
    klass = cell_classes()
    real_key = block_keys(row, col, klass)
    real_blocks = np.unique(real_key, return_counts=True)
    report = {'status': 'PASS', 'neurons': int(N), 'edges': int(E), 'seed': seed, 'arms': {}}
    for arm in ARMS:
        generator = np.random.default_rng(seed)
        if arm == 'real':
            arm_col = col.copy()
            passes = 0
        elif arm == 'shuffled':
            arm_col = generator.permutation(col)
            arm_col, passes = repair_self_loops(row, arm_col, generator)
        elif arm == 'block':
            arm_col = block_shuffle(row, col, klass, generator)
            arm_col, passes = repair_self_loops(row, arm_col, generator, key=real_key, original=col)
        else:
            arm_col = generator.integers(0, N, size=E, dtype=np.int32)
            arm_col, passes = repair_self_loops(row, arm_col, generator)
        in_arm, out_arm = degrees(row, arm_col)
        rows = np.repeat(np.arange(N, dtype=np.int64), np.diff(row))
        pairs = rows*N+arm_col.astype(np.int64)
        same = int((arm_col == col).sum())
        # the real arm is seed-independent; each control gets a per-seed directory, because one
        # realisation cannot separate "the anatomy helps" from "that particular rewiring was poor"
        directory = OUT/arm if arm == 'real' else OUT/f'{arm}-s{seed}'
        directory.mkdir(parents=True, exist_ok=True)
        row.tofile(directory/'row_ptr.bin')
        arm_col.astype('<i4').tofile(directory/'col_idx.bin')
        weights.astype('<f4').tofile(directory/'weights.bin')
        np.save(directory/'proofread_root_ids_783.npy',
                np.load(ROOT/'data/proofread_root_ids_783.npy'))
        report['arms'][arm] = {
            'in_degree_preserved': bool(np.array_equal(in_arm, in_real)),
            'out_degree_preserved': bool(np.array_equal(out_arm, out_real)),
            'weight_multiset_preserved': True,
            'class_block_counts_preserved': bool(all(np.array_equal(a, b) for a, b in zip(
                np.unique(block_keys(row, arm_col, klass), return_counts=True), real_blocks))),
            'self_loops': int((rows == arm_col).sum()),
            'repair_passes': passes,
            'distinct_edges': int(np.unique(pairs).size),
            'edges_unchanged_from_real': same,
            'edges_unchanged_fraction': same/E,
            'directory': str(directory.relative_to(ROOT))}
    checks = report['arms']
    assert checks['shuffled']['in_degree_preserved'] and checks['shuffled']['out_degree_preserved']
    assert checks['random']['in_degree_preserved']
    assert checks['block']['in_degree_preserved'] and checks['block']['out_degree_preserved']
    assert checks['block']['class_block_counts_preserved']
    assert all(arm['self_loops'] == 0 for arm in checks.values())
    assert checks['real']['edges_unchanged_fraction'] == 1.
    assert checks['shuffled']['edges_unchanged_fraction'] < .01
    report['scope'] = ('graph construction only: proves what each arm preserves, measures no '
                       'behaviour and trains nothing')
    report['fairness'] = ('row_ptr, weights, roster, sensory ports and the descending population '
                          'are identical across arms; only col_idx differs')
    report['sources_sha256'] = {str(p.relative_to(ROOT)): digest(p) for p in
                                (ROOT/'data/col_idx.bin', ROOT/'data/row_ptr.bin', Path(__file__))}
    report['recorded_unix'] = time.time()
    (OUT/'graphs.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    result = build()
    print(json.dumps({'status': result['status'],
                      'arms': {name: {k: v for k, v in arm.items() if k != 'directory'}
                               for name, arm in result['arms'].items()}}, indent=2))
