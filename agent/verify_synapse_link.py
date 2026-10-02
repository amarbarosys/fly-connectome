"""Independently recompute sampled L1 endpoint aggregates and check L2 bounds."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pyarrow.ipc as ipc

from agent.neural_connectome import ROOT, digest

SOURCE = ROOT / 'data/flywire_synapses_783.feather'
OUT = ROOT / '.work/agent/synapse-link'
MORPH = ROOT / '.work/agent/skeleton-link/morphology.npz'
SAMPLE_SEED = 20261002
SAMPLE_SIZE = 16
AXES = 'xyz'
MARGIN_NM = 5000.


def _new_accumulator():
    return {'pre': {'points': [], 'neuropil': {}}, 'post': {'points': [], 'neuropil': {}}}


def verify():
    link = json.loads((OUT / 'link.json').read_text())
    data = np.load(OUT / 'synapses.npz')
    morphology = np.load(MORPH)
    roster = np.load(ROOT / 'data/proofread_root_ids_783.npy').astype(np.int64, copy=False)
    assert link['status'] == 'PASS'
    assert np.array_equal(data['root_id'].astype(np.int64), roster)
    assert link['source_rows_read'] == link['source_metadata_rows']
    assert int(data['pre_count'].sum()) == link['pre_roster_synapse_endpoints']
    assert int(data['post_count'].sum()) == link['post_roster_synapse_endpoints']

    rng = np.random.default_rng(SAMPLE_SEED)
    sampled = np.sort(rng.choice(roster, size=SAMPLE_SIZE, replace=False))
    sampled_set = set(map(int, sampled))
    accum = {int(root): _new_accumulator() for root in sampled}
    reader = ipc.open_file(SOURCE)
    metadata = json.loads(reader.schema.metadata[b'pandas'])
    r = next(row for row in metadata['index_columns'] if row.get('kind') == 'range')
    metadata_rows = int((r['stop'] - r['start']) / r['step'])
    rows_read = 0
    columns = ['pre_pt_root_id', 'post_pt_root_id', 'neuropil',
               *[f'{side}_pt_position_{axis}' for side in ('pre', 'post') for axis in AXES]]
    for batch_no in range(reader.num_record_batches):
        batch = reader.get_batch(batch_no).select(columns)
        rows_read += batch.num_rows
        ids = {
            'pre': batch['pre_pt_root_id'].to_numpy(zero_copy_only=False).astype(np.int64, copy=False),
            'post': batch['post_pt_root_id'].to_numpy(zero_copy_only=False).astype(np.int64, copy=False),
        }
        points = {
            side: np.column_stack([batch[f'{side}_pt_position_{axis}'].to_numpy(zero_copy_only=False)
                                   for axis in AXES]).astype(np.int64, copy=False)
            for side in ('pre', 'post')
        }
        neuropil = batch['neuropil'].to_pylist()
        for side in ('pre', 'post'):
            mask = np.isin(ids[side], sampled, assume_unique=False)
            if not mask.any():
                continue
            roots, inverse = np.unique(ids[side][mask], return_inverse=True)
            selected_points = points[side][mask]
            selected_neuropil = [neuropil[i] for i in np.flatnonzero(mask)]
            for group, root in enumerate(roots):
                root = int(root)
                if root not in sampled_set:
                    continue
                group_mask = inverse == group
                entry = accum[root][side]
                entry['points'].append(selected_points[group_mask])
                for label in set(label for label, keep in zip(selected_neuropil, group_mask) if keep):
                    amount = sum(1 for label_value, keep in zip(selected_neuropil, group_mask)
                                 if keep and label_value == label)
                    entry['neuropil'][label] = entry['neuropil'].get(label, 0) + amount
    assert rows_read == metadata_rows == link['source_rows_read']

    row_for_root = {int(root): row for row, root in enumerate(roster)}
    morph_for_root = {int(root): row for row, root in enumerate(morphology['root_id'])}
    stored_distributions = {'pre': {}, 'post': {}}
    for side in ('pre', 'post'):
        prefix = f'{side}_neuropil_'
        for neuron_index, neuropil_id, amount in zip(
                data[prefix+'neuron_index'], data[prefix+'index'], data[prefix+'count']):
            stored_distributions[side].setdefault(int(neuron_index), {})[
                str(data['neuropil_names'][int(neuropil_id)])] = int(amount)
    checks = []
    for root in sampled:
        root = int(root)
        row, morph_row = row_for_root[root], morph_for_root[root]
        item = {'root_id': root, 'pre': {}, 'post': {}}
        for side in ('pre', 'post'):
            endpoint = accum[root][side]
            pts = np.concatenate(endpoint['points'], axis=0) if endpoint['points'] else np.empty((0, 3), np.int64)
            stored_count = int(data[f'{side}_count'][row])
            assert pts.shape[0] == stored_count, f'{side} count mismatch for {root}'
            stored_raw = np.asarray(data[f'{side}_centroid_raw'][row], np.float64)
            if len(pts):
                centroid = pts.astype(np.float64).mean(axis=0)
                extent = np.ptp(pts, axis=0)
                assert np.allclose(centroid, stored_raw, rtol=0, atol=1e-8), f'{side} centroid mismatch for {root}'
                assert np.array_equal(extent, data[f'{side}_extent_raw'][row]), f'{side} extent mismatch for {root}'
                bounds_lo = np.array([morphology[f'min_{axis}_nm'][morph_row] for axis in AXES])
                bounds_hi = np.array([morphology[f'max_{axis}_nm'][morph_row] for axis in AXES])
                centroid_in_box = bool(np.all((centroid >= bounds_lo - MARGIN_NM) &
                                              (centroid <= bounds_hi + MARGIN_NM)))
                assert centroid_in_box, f"{side} centroid outside sampled neuron's arbor box: {root}"
            else:
                centroid_in_box = None
            distribution = stored_distributions[side].get(row, {})
            assert distribution == endpoint['neuropil'], f'{side} neuropil distribution mismatch for {root}'
            item[side] = {'count': stored_count, 'independent_count': int(len(pts)),
                'centroid_raw': stored_raw.tolist(), 'centroid_inside_morphology_box_or_margin': centroid_in_box,
                'neuropil_categories': len(distribution)}
        checks.append(item)

    report = {'status': 'PASS', 'checked_neurons': len(checks), 'source_rows_read': rows_read,
        'source_metadata_rows': metadata_rows, 'sample_seed': SAMPLE_SEED,
        'all_sampled_counts_centroids_extents_neuropils_match': True,
        'sampled_synapse_centroids_inside_arbor_box_or_margin': True,
        'margin_nm': MARGIN_NM,
        'method': 'independent roster sample and Python per-root endpoint accumulator; exact int64 IDs; separate full Feather scan from link pass',
        'cases': checks,
        'sources_sha256': {'agent/link_synapses.py': digest(ROOT / 'agent/link_synapses.py'),
            'agent/verify_synapse_link.py': digest(Path(__file__)),
            'agent/verify_synapse_units.py': digest(ROOT / 'agent/verify_synapse_units.py')},
        'recorded_unix': time.time()}
    (OUT / 'independent-verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'cases'}, indent=2))
    return report


if __name__ == '__main__':
    result = verify()
    raise SystemExit(0 if result['status'] == 'PASS' else 1)
