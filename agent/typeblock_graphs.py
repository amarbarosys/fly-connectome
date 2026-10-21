"""Type-level block controls: the 'genome' at FlyWire cell_type resolution.

The block arm of baseline_graphs.py shuffles within (post cell_class, pre cell_class) and keeps a
50-class wiring table; it lost the learning valence gap (real 0.73, block ~0). This arm keeps the
8,847-type table instead: col_idx permuted only among edges with the same (post cell_type, pre
cell_type), so both degree sequences, the weight multiset and every type-to-type edge count survive;
which neuron within a type connects to which does not. A blank cell_type falls back to cell_class,
then super_class. Writes .work/agent/matched-baseline/typeblock-s<seed>/ and typeblock.json only;
graphs.json and the other arms are not touched.
"""
import csv
import json
import numpy as np
from agent.neural_connectome import ROOT, N, E, ANNOTATIONS, graph_arrays
from agent.baseline_graphs import OUT, degrees, block_keys, block_shuffle, repair_edges

SEEDS = (20261002, 7, 1009, 424242, 31337)


def cell_types():
    roots = np.load(ROOT/'data/proofread_root_ids_783.npy')
    index = {int(root): i for i, root in enumerate(roots)}
    names = np.full(N, 'unannotated', dtype=object)
    for row in csv.DictReader(ANNOTATIONS.open(), delimiter='\t'):
        node = index.get(int(row['root_id']))
        if node is not None:
            names[node] = row['cell_type'] or row['cell_class'] or row['super_class'] or 'unannotated'
    return np.unique(names, return_inverse=True)[1].astype(np.int64)


def build():
    row, col, weights = graph_arrays()
    in_real, out_real = degrees(row, col)
    types = cell_types()
    real_key = block_keys(row, col, types)
    real_blocks = np.unique(real_key, return_counts=True)
    report = {'status': 'PASS', 'neurons': int(N), 'edges': int(E), 'types': int(types.max() + 1),
              'blocks': int(real_blocks[0].size), 'arms': {}}
    for seed in SEEDS:
        generator = np.random.default_rng(seed)
        arm_col = block_shuffle(row, col, types, generator)
        arm_col, passes = repair_edges(row, arm_col, generator, key=real_key, original=col)
        in_arm, out_arm = degrees(row, arm_col)
        rows = np.repeat(np.arange(N, dtype=np.int64), np.diff(row))
        same = int((arm_col == col).sum())
        directory = OUT/f'typeblock-s{seed}'
        directory.mkdir(parents=True, exist_ok=True)
        row.tofile(directory/'row_ptr.bin')
        arm_col.astype('<i4').tofile(directory/'col_idx.bin')
        weights.astype('<f4').tofile(directory/'weights.bin')
        np.save(directory/'proofread_root_ids_783.npy', np.load(ROOT/'data/proofread_root_ids_783.npy'))
        arm = {'in_degree_preserved': bool(np.array_equal(in_arm, in_real)),
               'out_degree_preserved': bool(np.array_equal(out_arm, out_real)),
               'type_block_counts_preserved': bool(all(np.array_equal(a, b) for a, b in zip(
                   np.unique(block_keys(row, arm_col, types), return_counts=True), real_blocks))),
               'self_loops': int((rows == arm_col).sum()),
               'repair_passes': passes,
               'edges_unchanged_fraction': same/E,
               'directory': str(directory.relative_to(ROOT))}
        assert arm['in_degree_preserved'] and arm['out_degree_preserved'] and arm['type_block_counts_preserved']
        assert arm['self_loops'] == 0
        report['arms'][f'typeblock-s{seed}'] = arm
    (OUT/'typeblock.json').write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    result = build()
    print(json.dumps({k: v for k, v in result.items() if k != 'arms'}))
    for name, arm in result['arms'].items():
        print(name, f"unchanged {arm['edges_unchanged_fraction']:.4f}", f"repair {arm['repair_passes']}")
