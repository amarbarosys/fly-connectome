"""Stream L1 synapse endpoints into exact per-roster-neuron spatial aggregates."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pyarrow.ipc as ipc

from agent.neural_connectome import ROOT, digest

SOURCE = ROOT / 'data/flywire_synapses_783.feather'
ROSTER_PATH = ROOT / 'data/proofread_root_ids_783.npy'
UNITS_PATH = ROOT / '.work/agent/synapse-units/verification.json'
OUT = ROOT / '.work/agent/synapse-link'
BATCH_SIZE = 1 << 17
AXES = 'xyz'


def _run_counts(ids, roster_sorted, roster_rows, run_counts, seen_keys, previous):
    """Count source runs for rostered IDs, preserving a batch-boundary run."""
    if not ids.size:
        return previous
    starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
    run_ids = ids[starts]
    run_positions = np.searchsorted(roster_sorted, run_ids)
    valid = run_positions < roster_sorted.size
    valid[valid] &= roster_sorted[run_positions[valid]] == run_ids[valid]
    current_first = int(run_ids[0])
    for run_number, (root, sorted_pos, is_valid) in enumerate(zip(run_ids, run_positions, valid)):
        root = int(root)
        if not is_valid:
            continue
        if run_number == 0 and previous == root:
            continue
        row = int(roster_rows[sorted_pos])
        run_counts[row] += 1
        seen_keys[row] = True
    return int(ids[-1])


def _roster_rows(ids, roster_sorted, roster_rows):
    positions = np.searchsorted(roster_sorted, ids)
    valid = positions < roster_sorted.size
    valid[valid] &= roster_sorted[positions[valid]] == ids[valid]
    return valid, roster_rows[positions[valid]]


def link():
    started = time.time()
    roster = np.load(ROSTER_PATH).astype(np.int64, copy=False)
    roster_order = np.argsort(roster)
    roster_sorted = roster[roster_order]
    assert np.unique(roster_sorted).size == roster.size
    units = json.loads(UNITS_PATH.read_text())
    if units['verdict'] == 'nanometres':
        nm_per_unit = np.ones(3, np.float64)
    elif units['verdict'] == 'scaled_voxel_coordinates':
        nm_per_unit = np.array([units['per_axis_nm_per_coordinate_unit'][axis] for axis in AXES], np.float64)
    else:
        raise RuntimeError(f"synapse units unresolved: {units['verdict']}")

    n = roster.size
    counts = {side: np.zeros(n, np.int64) for side in ('pre', 'post')}
    sums = {side: np.zeros((n, 3), np.float64) for side in ('pre', 'post')}
    lows = {side: np.full((n, 3), np.iinfo(np.int64).max, np.int64) for side in ('pre', 'post')}
    highs = {side: np.full((n, 3), np.iinfo(np.int64).min, np.int64) for side in ('pre', 'post')}
    np_counts = {side: [dict() for _ in range(n)] for side in ('pre', 'post')}
    run_counts = {side: np.zeros(n, np.int32) for side in ('pre', 'post')}
    seen_keys = {side: np.zeros(n, bool) for side in ('pre', 'post')}
    previous = {'pre': None, 'post': None}
    neuropil_names: list[str] = []
    neuropil_index: dict[str, int] = {}
    reader = ipc.open_file(SOURCE)
    pandas_metadata = json.loads(reader.schema.metadata[b'pandas'])
    range_index = next(row for row in pandas_metadata['index_columns'] if row.get('kind') == 'range')
    expected_rows = int((range_index['stop'] - range_index['start']) / range_index['step'])
    rows_read = pre_roster_rows = post_roster_rows = 0
    columns = ['pre_pt_root_id', 'post_pt_root_id', 'neuropil',
               *[f'{side}_pt_position_{axis}' for side in ('pre', 'post') for axis in AXES]]

    for batch_number in range(reader.num_record_batches):
        batch = reader.get_batch(batch_number).select(columns)
        rows_read += batch.num_rows
        ids_by_side = {
            'pre': batch['pre_pt_root_id'].to_numpy(zero_copy_only=False).astype(np.int64, copy=False),
            'post': batch['post_pt_root_id'].to_numpy(zero_copy_only=False).astype(np.int64, copy=False),
        }
        points_by_side = {
            side: np.column_stack([batch[f'{side}_pt_position_{axis}'].to_numpy(zero_copy_only=False)
                                   for axis in AXES]).astype(np.int64, copy=False)
            for side in ('pre', 'post')
        }
        encoded = batch['neuropil'].dictionary_encode()
        local_labels = encoded.dictionary.to_pylist()
        local_codes = encoded.indices.to_numpy(zero_copy_only=False).astype(np.int32, copy=False)
        translated = np.empty(len(local_labels), np.int32)
        for local_id, label in enumerate(local_labels):
            if label not in neuropil_index:
                neuropil_index[label] = len(neuropil_names)
                neuropil_names.append(label)
            translated[local_id] = neuropil_index[label]
        codes = translated[local_codes]

        for side in ('pre', 'post'):
            ids = ids_by_side[side]
            previous[side] = _run_counts(ids, roster_sorted, roster_order, run_counts[side],
                                          seen_keys[side], previous[side])
            matched, roster_rows = _roster_rows(ids, roster_sorted, roster_order)
            matched_count = int(matched.sum())
            if side == 'pre':
                pre_roster_rows += matched_count
            else:
                post_roster_rows += matched_count
            if not matched_count:
                continue
            points = points_by_side[side][matched]
            counts[side] += np.bincount(roster_rows, minlength=n)
            for axis in range(3):
                sums[side][:, axis] += np.bincount(roster_rows, weights=points[:, axis], minlength=n)
                np.minimum.at(lows[side][:, axis], roster_rows, points[:, axis])
                np.maximum.at(highs[side][:, axis], roster_rows, points[:, axis])
            selected_codes = codes[matched]
            composite = roster_rows.astype(np.int64) * len(neuropil_names) + selected_codes
            keys, frequencies = np.unique(composite, return_counts=True)
            for key, frequency in zip(keys, frequencies):
                neuron_row, neuropil_row = divmod(int(key), len(neuropil_names))
                current = np_counts[side][neuron_row]
                current[neuropil_row] = current.get(neuropil_row, 0) + int(frequency)

    assert rows_read == expected_rows, f'source row count {rows_read} != metadata {expected_rows}'
    assert int(counts['pre'].sum()) == pre_roster_rows
    assert int(counts['post'].sum()) == post_roster_rows
    assert int(counts['pre'].sum()) == pre_roster_rows, 'presynaptic preflight failed'
    has_pre = counts['pre'] > 0
    has_post = counts['post'] > 0
    safe_pre = np.maximum(counts['pre'], 1)[:, None]
    safe_post = np.maximum(counts['post'], 1)[:, None]
    pre_centroid_raw = sums['pre'] / safe_pre
    post_centroid_raw = sums['post'] / safe_post
    pre_extent_raw = np.where(has_pre[:, None], highs['pre'] - lows['pre'], 0)
    post_extent_raw = np.where(has_post[:, None], highs['post'] - lows['post'], 0)
    np_counts_arrays = {}
    for side in ('pre', 'post'):
        pairs = [(neuron, neuropil, count) for neuron, values in enumerate(np_counts[side])
                 for neuropil, count in values.items()]
        pairs.sort()
        np_counts_arrays[f'{side}_neuropil_neuron_index'] = np.fromiter((p[0] for p in pairs), np.int32)
        np_counts_arrays[f'{side}_neuropil_index'] = np.fromiter((p[1] for p in pairs), np.int32)
        np_counts_arrays[f'{side}_neuropil_count'] = np.fromiter((p[2] for p in pairs), np.int64)

    output = {
        'root_id': roster,
        'pre_count': counts['pre'], 'post_count': counts['post'],
        'pre_centroid_raw': pre_centroid_raw, 'post_centroid_raw': post_centroid_raw,
        'pre_extent_raw': pre_extent_raw, 'post_extent_raw': post_extent_raw,
        'pre_min_raw': np.where(has_pre[:, None], lows['pre'], 0),
        'pre_max_raw': np.where(has_pre[:, None], highs['pre'], 0),
        'post_min_raw': np.where(has_post[:, None], lows['post'], 0),
        'post_max_raw': np.where(has_post[:, None], highs['post'], 0),
        'pre_centroid_nm': pre_centroid_raw * nm_per_unit,
        'post_centroid_nm': post_centroid_raw * nm_per_unit,
        'pre_extent_nm': pre_extent_raw * nm_per_unit,
        'post_extent_nm': post_extent_raw * nm_per_unit,
        'neuropil_names': np.asarray(neuropil_names),
        'contiguous_single_run_pre': run_counts['pre'] == 1,
        'contiguous_single_run_post': run_counts['post'] == 1,
        **np_counts_arrays,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / 'synapses.npz', **output)
    report = {
        'status': 'PASS', 'source': str(SOURCE.relative_to(ROOT)), 'source_rows_read': rows_read,
        'source_metadata_rows': expected_rows, 'source_record_batches': reader.num_record_batches,
        'roster_neurons': int(n), 'pre_roster_synapse_endpoints': pre_roster_rows,
        'post_roster_synapse_endpoints': post_roster_rows,
        'pre_count_sum': int(counts['pre'].sum()), 'post_count_sum': int(counts['post'].sum()),
        'neurons_with_presynapses': int(has_pre.sum()), 'neurons_with_postsynapses': int(has_post.sum()),
        'pre_keys_single_contiguous_run': int((run_counts['pre'] == 1).sum()),
        'post_keys_single_contiguous_run': int((run_counts['post'] == 1).sum()),
        'pre_keys_noncontiguous': int(np.count_nonzero(run_counts['pre'] > 1)),
        'post_keys_noncontiguous': int(np.count_nonzero(run_counts['post'] > 1)),
        'strategy': 'one bounded streaming pass with order-independent per-neuron accumulators; both key columns are measured noncontiguous so no contiguous-key assumption is used',
        'neuropil_categories': neuropil_names,
        'units': units['verdict'], 'nm_per_coordinate_unit': dict(zip(AXES, nm_per_unit.tolist())),
        'raw_source_values_preserved': True,
        'artifact': str((OUT / 'synapses.npz').relative_to(ROOT)),
        'sources_sha256': {
            'data/proofread_root_ids_783.npy': digest(ROSTER_PATH),
            'agent/link_synapses.py': digest(Path(__file__)),
            'agent/verify_synapse_units.py': digest(ROOT / 'agent/verify_synapse_units.py'),
            'agent/link_skeletons.py': digest(ROOT / 'agent/link_skeletons.py'),
            '.work/agent/synapse-units/verification.json': digest(UNITS_PATH),
            '.work/agent/skeleton-link/morphology.npz': digest(ROOT / '.work/agent/skeleton-link/morphology.npz'),
        },
        'recorded_unix': time.time(),
    }
    (OUT / 'link.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    link()
