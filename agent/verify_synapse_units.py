"""Test L1 synapse endpoint coordinates against exact L2 arbor bounding boxes."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pyarrow.ipc as ipc

from agent.neural_connectome import ROOT

SYNAPSES = ROOT / 'data/flywire_synapses_783.feather'
MORPHOLOGY = ROOT / '.work/agent/skeleton-link/morphology.npz'
OUT = ROOT / '.work/agent/synapse-units/verification.json'
AXES = 'xyz'
FACTORS = (1., 2., 4., 8., 10., 16., 20., 32., 40., 64., 80., 100.)
MARGIN_NM = 5000.
SAMPLE_SIZE = 512
SAMPLE_SEED = 783


def verify():
    morphology = np.load(MORPHOLOGY)
    roster = morphology['root_id'].astype(np.int64, copy=False)
    assert morphology['linked'].all()
    lower = np.column_stack([morphology[f'min_{axis}_nm'] for axis in AXES])
    upper = np.column_stack([morphology[f'max_{axis}_nm'] for axis in AXES])
    assert np.isfinite(lower).all() and np.isfinite(upper).all()
    assert np.all(upper >= lower)

    rng = np.random.default_rng(SAMPLE_SEED)
    chosen_rows = np.sort(rng.choice(roster.size, size=min(SAMPLE_SIZE, roster.size), replace=False))
    sampled_roots = roster[chosen_rows]
    sampled_index = {int(root): i for i, root in enumerate(sampled_roots)}
    endpoint_parts: list[list[np.ndarray]] = [[] for _ in sampled_roots]
    reader = ipc.open_file(SYNAPSES)
    pandas_metadata = json.loads(reader.schema.metadata[b'pandas'])
    index_metadata = next(row for row in pandas_metadata['index_columns'] if row.get('kind') == 'range')
    metadata_rows = int((index_metadata['stop'] - index_metadata['start']) / index_metadata['step'])
    row_count = 0
    columns = ['pre_pt_root_id', 'post_pt_root_id',
               *[f'{side}_pt_position_{axis}' for side in ('pre', 'post') for axis in AXES]]
    for batch_index in range(reader.num_record_batches):
        batch = reader.get_batch(batch_index).select(columns)
        row_count += batch.num_rows
        pre = batch['pre_pt_root_id'].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        post = batch['post_pt_root_id'].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        pre_xyz = np.column_stack([batch[f'pre_pt_position_{axis}'].to_numpy(zero_copy_only=False)
                                   for axis in AXES]).astype(np.float64, copy=False)
        post_xyz = np.column_stack([batch[f'post_pt_position_{axis}'].to_numpy(zero_copy_only=False)
                                    for axis in AXES]).astype(np.float64, copy=False)
        for ids, xyz in ((pre, pre_xyz), (post, post_xyz)):
            mask = np.isin(ids, sampled_roots, assume_unique=False)
            if not mask.any():
                continue
            selected_ids = ids[mask]
            selected_xyz = xyz[mask]
            roots, inverse = np.unique(selected_ids, return_inverse=True)
            for group, root in enumerate(roots):
                row = sampled_index[int(root)]
                endpoint_parts[row].append(selected_xyz[inverse == group])

    assert row_count == metadata_rows, f'row count {row_count} differs from source metadata {metadata_rows}'

    endpoints = [np.concatenate(parts, axis=0) if parts else np.empty((0, 3), np.float64)
                 for parts in endpoint_parts]
    present_rows = np.array([i for i, points in enumerate(endpoints) if len(points)], dtype=np.int64)
    assert present_rows.size >= SAMPLE_SIZE * .9, f'too few sampled neurons have L1 endpoints: {present_rows.size}'
    points_by_neuron = [endpoints[i] for i in present_rows]
    lo_by_neuron, hi_by_neuron = lower[chosen_rows[present_rows]], upper[chosen_rows[present_rows]]
    total_points = sum(len(points) for points in points_by_neuron)

    axis_best = []
    axis_scores = {}
    for axis_index, axis in enumerate(AXES):
        scores = {}
        for factor in FACTORS:
            inside = 0
            for points, lo, hi in zip(points_by_neuron, lo_by_neuron, hi_by_neuron):
                values = points[:, axis_index] * factor
                inside += int(np.count_nonzero((values >= lo[axis_index] - MARGIN_NM) &
                                               (values <= hi[axis_index] + MARGIN_NM)))
            scores[str(factor)] = inside / total_points
        best = max(FACTORS, key=lambda factor: (scores[str(factor)], -factor))
        axis_best.append(best)
        axis_scores[axis] = scores

    in_box = near_box = 0
    strict_inside = 0
    for points, lo, hi in zip(points_by_neuron, lo_by_neuron, hi_by_neuron):
        scaled = points * np.asarray(axis_best)[None, :]
        strict = np.all((scaled >= lo[None, :]) & (scaled <= hi[None, :]), axis=1)
        near = np.all((scaled >= (lo - MARGIN_NM)[None, :]) &
                      (scaled <= (hi + MARGIN_NM)[None, :]), axis=1)
        strict_inside += int(strict.sum())
        near_box += int(near.sum())

    per_axis_near = {axis: axis_scores[axis][str(factor)] for axis, factor in zip(AXES, axis_best)}
    near_fraction = near_box / total_points
    if all(factor == 1. for factor in axis_best) and near_fraction >= .95:
        verdict = 'nanometres'
    elif near_fraction >= .95:
        verdict = 'scaled_voxel_coordinates'
    else:
        verdict = 'no_single_scale_factor_supported'

    report = {
        'status': 'PASS' if verdict != 'no_single_scale_factor_supported' else 'FAIL',
        'verdict': verdict,
        'sampled_neurons': int(present_rows.size),
        'requested_sampled_neurons': int(len(sampled_roots)),
        'sampled_synapse_endpoints': int(total_points),
        'source_rows_read': int(row_count),
        'source_record_batches': int(reader.num_record_batches),
        'source_metadata_rows': metadata_rows,
        'source_units': 'raw integer endpoint coordinates from FlyWire v783 feather',
        'skeleton_units': 'source nanometres, exact per-neuron min/max bounds from LOD1 ds2 morphology',
        'per_axis_nm_per_coordinate_unit': {axis: factor for axis, factor in zip(AXES, axis_best)},
        'per_axis_best_near_fraction': per_axis_near,
        'scale_search_factors': list(FACTORS),
        'margin_nm': MARGIN_NM,
        'strict_3d_inside_fraction': strict_inside / total_points,
        'inside_or_within_margin_fraction': near_fraction,
        'method': 'deterministic uniform roster sample; stream L1 pre/post endpoint keys; exact int64 root joins; score each axis against its neuron-specific L2 bounding box',
        'sample_seed': SAMPLE_SEED,
        'recorded_unix': time.time(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    result = verify()
    raise SystemExit(0 if result['status'] == 'PASS' else 1)
