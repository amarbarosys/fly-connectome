"""Independently recompute sampled skeleton morphology and audit the L2 link receipt."""
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from agent.neural_connectome import ROOT, digest

SOURCE = ROOT/'data/sk_lod1_783_healed_ds2.parquet'


def cable_by_dict(node_id, parent_id, xyz):
    """Cable length through an explicit node_id -> row dict: no sorting, no searchsorted."""
    where = {int(node): row for row, node in enumerate(node_id)}
    total = 0.
    for row, parent in enumerate(parent_id):
        source = where.get(int(parent))
        if parent < 0 or source is None:
            continue
        step = xyz[row]-xyz[source]
        total += float(np.sqrt(float(step@step)))
    return total


def verify(root, samples=12):
    link = json.loads((root/'link.json').read_text())
    data = np.load(root/'morphology.npz')
    # int64, never uint64: np.isin would promote a mixed comparison to float64 and root IDs
    # exceed 2**53, which silently matched 7 extra neurons before this cast was added
    roster = np.load(ROOT/'data/proofread_root_ids_783.npy').astype(np.int64)
    assert link['status'] == 'PASS' and link['linked_fraction'] == 1.
    assert np.array_equal(data['root_id'].astype(np.int64), roster) and bool(data['linked'].all())
    assert digest(ROOT/'agent/link_skeletons.py') == link['sources_sha256']['agent/link_skeletons.py']
    source = pq.ParquetFile(SOURCE)
    neuron = source.read(columns=['neuron'])['neuron'].to_numpy(zero_copy_only=False)
    starts = np.concatenate(([0], np.flatnonzero(np.diff(neuron) != 0)+1))
    sizes = np.concatenate((starts[1:], [neuron.size]))-starts
    assert starts.size == np.unique(neuron[starts]).size, 'a neuron is not contiguous'
    rostered = np.isin(neuron[starts], roster)
    accounted = int(sizes[rostered].sum()); extra = int(sizes[~rostered].sum())
    assert accounted == int(link['node_count_total']) and accounted+extra == neuron.size
    assert int((~rostered).sum()) == link['skeleton_without_roster']
    assert int(rostered.sum()) == link['linked_neurons'] == roster.size
    order = np.argsort(sizes)
    picks = np.unique(np.concatenate(([order[0], order[-1]],
        order[np.linspace(0, order.size-1, max(samples-2, 1)).astype(int)])))
    index = {int(value): row for row, value in enumerate(roster)}
    columns = source.read(columns=['node_id', 'parent_id', 'x', 'y', 'z'])
    checks = []
    for pick in picks:
        name = int(neuron[starts[pick]]); row = index.get(name)
        if row is None:
            continue
        block = columns.slice(int(starts[pick]), int(sizes[pick]))
        node_id = block['node_id'].to_numpy(zero_copy_only=False).astype(np.int64)
        parent_id = block['parent_id'].to_numpy(zero_copy_only=False).astype(np.int64)
        xyz = np.stack([block[axis].to_numpy(zero_copy_only=False).astype(np.float64)
                        for axis in 'xyz'], axis=1)
        measured = cable_by_dict(node_id, parent_id, xyz)
        stored = float(data['cable_length_nm'][row])
        error = abs(measured-stored)/max(measured, 1.)
        assert error < 1e-9, f'cable mismatch for {name}: {measured} vs {stored}'
        assert int(data['node_count'][row]) == int(sizes[pick])
        assert np.isfinite([data[field][row] for field in
            ('radius_mean_nm', 'extent_x_nm', 'centroid_x_nm')]).all()
        checks.append({'root_id': name, 'nodes': int(sizes[pick]),
            'independent_nm': measured, 'stored_nm': stored, 'relative_error': error})
    report = {'status': 'PASS', 'checked_neurons': len(checks),
        'max_relative_cable_error': max(check['relative_error'] for check in checks),
        'roster_aligned': True, 'all_roster_linked': True, 'contiguous_neurons': True,
        'nodes_rostered': accounted, 'nodes_unrostered': extra, 'nodes_total': int(neuron.size),
        'method': 'dict-based parent resolution against the linker argsort/searchsorted path',
        'cases': checks,
        'scope': 'recomputes cable length and node count for sampled neurons on the LOD1 ds2 '
                 'downsampling; does not revalidate extent/centroid/radius values or convert units'}
    (root/'independent-verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'cases'}, indent=2))
    return report


if __name__ == '__main__':
    verify(ROOT/'.work/agent/skeleton-link')
