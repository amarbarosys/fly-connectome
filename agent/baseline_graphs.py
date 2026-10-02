"""Build matched control graphs for the connectome-benefit test, and prove what each preserves.

The question G1 asks is whether the real wiring beats a network of the same size that carries no
anatomy. Three arms share everything except the edge list:

  real     the FlyWire CSR as built
  shuffled col_idx globally permuted: both degree sequences and the weight-position pairing are
           preserved exactly, only which neuron connects to which is destroyed. This is the
           configuration-model control and the one that matters.
  random   col_idx drawn uniformly: in-degree per neuron and the weight multiset survive,
           out-degree structure does not. A weaker control, kept because it separates "the
           specific wiring matters" from "any structured out-degree matters".

Sensory ports and the descending population are derived from root IDs, not topology, so they are
identical across arms. Nothing here trains or runs physics.
"""
import json
from pathlib import Path
import time
import numpy as np
from agent.neural_connectome import ROOT, N, E, graph_arrays, digest

OUT = ROOT/'.work/agent/matched-baseline'
ARMS = ('real', 'shuffled', 'random')


def degrees(row, col):
    """In-degree per postsynaptic row and out-degree per presynaptic column."""
    return np.diff(row).astype(np.int64), np.bincount(col, minlength=N).astype(np.int64)


def repair_self_loops(row, col, generator):
    """Swap self-loops against random other edges. The source graph has none and a permutation
    creates some, so without this the control would differ from the real graph in a way that is
    not about wiring."""
    rows = np.repeat(np.arange(N, dtype=np.int64), np.diff(row))
    for _ in range(64):
        loops = np.flatnonzero(rows == col)
        if loops.size == 0:
            return col, int(_)
        partners = generator.integers(0, E, size=loops.size)
        # a swap is only taken when it removes the loop without creating another
        keep = (col[partners] != rows[loops]) & (col[loops] != rows[partners])
        take, give = loops[keep], partners[keep]
        col[take], col[give] = col[give].copy(), col[take].copy()
    raise RuntimeError('FAIL repair: self-loops remain after 64 passes')


def build(seed=20261002):
    row, col, weights = graph_arrays()
    in_real, out_real = degrees(row, col)
    report = {'status': 'PASS', 'neurons': int(N), 'edges': int(E), 'seed': seed, 'arms': {}}
    for arm in ARMS:
        generator = np.random.default_rng(seed)
        if arm == 'real':
            arm_col = col.copy()
            passes = 0
        elif arm == 'shuffled':
            arm_col = generator.permutation(col)
            arm_col, passes = repair_self_loops(row, arm_col, generator)
        else:
            arm_col = generator.integers(0, N, size=E, dtype=np.int32)
            arm_col, passes = repair_self_loops(row, arm_col, generator)
        in_arm, out_arm = degrees(row, arm_col)
        rows = np.repeat(np.arange(N, dtype=np.int64), np.diff(row))
        pairs = rows*N+arm_col.astype(np.int64)
        same = int((arm_col == col).sum())
        directory = OUT/arm
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
            'self_loops': int((rows == arm_col).sum()),
            'repair_passes': passes,
            'distinct_edges': int(np.unique(pairs).size),
            'edges_unchanged_from_real': same,
            'edges_unchanged_fraction': same/E,
            'directory': str(directory.relative_to(ROOT))}
    checks = report['arms']
    assert checks['shuffled']['in_degree_preserved'] and checks['shuffled']['out_degree_preserved']
    assert checks['random']['in_degree_preserved']
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
