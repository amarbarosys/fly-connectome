"""Bounded, provenance-labelled visual replay and calcium comparison.

The cue renderer and spike-to-fluorescence filter are engineering hypotheses.
NWB ROI indices and FlyWire root IDs remain separate namespaces.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from agent.experiment_ledger import consume_final

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / '.work/agent/calcium-validation'
NWB = ROOT / 'data/sub-13_ses-20201201_behavior_ophys.nwb'
REFERENCE = ROOT / '.work/agent/five-layer-assets/calcium-behavior-reference.npz'
ANNOTATIONS = ROOT / 'data/annotations/flyconnectome-flywire_annotations-a83b277/supplemental_files/Supplemental_file1_neuron_annotations.tsv'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def cue_relative(body_yaw, cue_world):
    return wrap(cue_world - body_yaw)


def trial_values(out, name):
    """Read six source trial values, retaining HDF5 descriptions."""
    target = out / (name + '.bin')
    result = subprocess.run(
        ['/usr/bin/h5dump', '-d', '/intervals/trials/' + name,
         '-b', 'LE', '-o', str(target), str(NWB)],
        check=True, capture_output=True, text=True)
    (out / (name + '-source.ddl')).write_text(result.stdout)
    values = np.fromfile(target, dtype='<f8')
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError('unexpected NWB trial table')
    return values


def prepare(out, frames=48, start_offset=1., source_matched=False):
    if (out/'protocol.npz').exists():raise ValueError('protocol already exists; choose a fresh output directory')
    out.mkdir(parents=True, exist_ok=True)
    if frames < 24:
        raise ValueError('at least 24 frames per trial required')
    if not np.isfinite(start_offset) or start_offset < 1:
        raise ValueError('window offset must be finite and at least one second')
    a = np.load(REFERENCE)
    t, heading = a['heading_timestamps'], a['heading_radians']
    if not np.array_equal(t, a['calcium_timestamps']) or not np.all(np.diff(t) > 0):
        raise ValueError('reference clocks do not match')
    starts, stops, contrasts = [trial_values(out, n) for n in
                               ('start_time', 'stop_time', 'cue_contrast')]
    metadata = subprocess.run(['/usr/bin/h5dump', '-d', '/intervals/trials/start_time',
        '-d', '/intervals/trials/stop_time', '-d', '/intervals/trials/cue_contrast',
        str(NWB)], check=True, capture_output=True, text=True)
    (out / 'trial-source-metadata.ddl').write_text(metadata.stdout)
    selected, trials = [], []
    for trial, (start, stop) in enumerate(zip(starts, stops)):
        # A bounded contiguous window after one second, never a prefix across trials.
        indices = np.flatnonzero((t >= start + start_offset) & (t < stop))[:frames]
        if len(indices) != frames:
            raise ValueError('trial lacks requested samples')
        selected.extend(indices)
        trials.extend([trial] * frames)
    selected, trials = np.asarray(selected), np.asarray(trials)
    code = contrasts[trials].astype(int)
    if not set(code) <= {1, 2, 3}:
        raise ValueError('unknown source contrast category')
    proxy_contrast = np.choose(code - 1, [0., 1. / 15 if source_matched else .25, 1.])
    # Trial-level final holdout. Train/selection share the first three trials.
    within = np.tile(np.arange(frames), 6)
    train = (trials < 3) & (within >= 8) & (within < int(frames * .65))
    selection = (trials < 3) & (within >= int(frames * .65) + 4)
    final = (trials >= 3) & (within >= 8)
    if not train.any() or not selection.any() or not final.any():
        raise ValueError('empty data split')
    np.savez_compressed(out / 'protocol.npz', source_indices=selected,
        source_timestamps=t[selected], heading=heading[selected], trial=trials,
        contrast_code=code, proxy_contrast=proxy_contrast,
        calcium=a['calcium'][selected], roi_indices=a['roi_indices'],
        train=train, selection=selection, final=final,
        cue_heading_sign=-1 if source_matched else 1,
        neural_tick_s=.0025 if source_matched else 0.,
        source_stimulus=source_matched)
    report = {'status': 'PASS', 'reference_sha256': digest(REFERENCE),
        'source_nwb': str(NWB), 'frames_per_trial': frames,
        'window_start_offset_s': start_offset,
        'source_trial_contrast_codes': contrasts.tolist(),
        'frames': len(selected), 'train_frames': int(train.sum()),
        'selection_frames': int(selection.sum()), 'final_frames': int(final.sum()),
        'source_heading_zero': 'visual cue directly in front',
        'positive_heading_orientation': ('source fly HD counterclockwise; cue bearing = -HD'
            if source_matched else 'assumed counterclockwise; source convention unverified'),
        'stimulus': 'yoked cue-relative heading replay, not autonomous behavior reproduction',
        'contrast_mapping': {'1': 0., '2': 1. / 15 if source_matched else .25, '3': 1.},
        'contrast_status': ('source digital panel levels; optical luminance and spectrum not calibrated'
            if source_matched else 'source categories known; numerical contrast and cue geometry are proxies'),
        'cue_geometry': ('7.5 degree rectangular stripe, black background; grayscale panel-level proxy'
            if source_matched else 'Gaussian stripe on gray, engineering proxy'),
        'neural_clock': ('fixed 2.5 ms engineering tick; matches existing 4 updates/10 ms exchange, not biological calibration'
            if source_matched else 'four dimensionless updates per source frame'),
        'methods_url': 'https://www.nature.com/articles/s41593-024-01823-z',
        'splits': 'first 3 trial windows train/selection with 4-frame gap; last 3 windows final; first 8 frames excluded from scoring',
        'coverage': 'six short contiguous windows, not full 20-minute replay'}
    write_json(out / 'protocol.json', report)
    return report


def cue_images(relative_heading, contrast, height=32, width=64, source_stimulus=False):
    """Analytic grayscale retinal cue, not native rendered eye imagery."""
    azimuth = (np.linspace(-np.pi / 2, np.pi / 2, width, endpoint=False) + np.pi / (2 * width)
               if source_stimulus else np.linspace(-np.pi / 2, np.pi / 2, width))
    centers = np.asarray([np.pi / 2, -np.pi / 2]) if source_stimulus else np.asarray([np.pi / 4, -np.pi / 4])
    distance = wrap(azimuth[None, :] + centers[:, None] - relative_heading)
    if source_stimulus:
        stripe = (np.abs(distance) <= np.deg2rad(7.5) / 2).astype(float)
        # Relative digital intensity, not a blue-spectrum/photoreceptor model.
        luminance = float(contrast) * stripe
    else:
        stripe = np.exp(-.5 * (distance / .15) ** 2)
        luminance = .5 + .45 * float(contrast) * stripe
    pixels = np.broadcast_to(luminance[:, None, :, None], (2, height, width, 3))
    return np.rint(pixels * 255).astype(np.uint8)


def epg_population():
    roster = np.load(ROOT / 'data/proofread_root_ids_783.npy')
    index = {int(root): i for i, root in enumerate(roster)}
    rows = list(csv.DictReader(ANNOTATIONS.open(), delimiter='\t'))
    roots = sorted({int(r['root_id']) for r in rows
                    if r['cell_type'] == 'EPG' and int(r['root_id']) in index})
    if not roots:
        raise ValueError('no annotated EPG population in simulation roster')
    return np.asarray(roots, dtype=np.int64), np.asarray([index[r] for r in roots])


def record(out):
    """Full CPU connectome and physical body, no GPU context or viewer."""
    if (out/'simulation.npz').exists():raise ValueError('simulation already exists; use recorded receipt or a fresh run')
    import copy
    from agent.neural_connectome import CPUBrain
    from agent.vision import EyeVision
    from fly_arena.ethology_arena import EthologyArena

    protocol = np.load(out / 'protocol.npz')
    source_stimulus = bool(protocol['source_stimulus']) if 'source_stimulus' in protocol else False
    sign = int(protocol['cue_heading_sign']) if 'cue_heading_sign' in protocol else 1
    tick_s = float(protocol['neural_tick_s']) if 'neural_tick_s' in protocol else 0.
    roots, indices = epg_population()
    brain = CPUBrain(flies=1)
    arena = None
    rates, rows = [], []
    started = time.monotonic()
    current_trial = -1
    previous_source = None
    try:
        arena = EthologyArena(seed=20261001, food_patches=[])
        # No eye or body image rendering is needed for the explicit retinal proxy.
        arena.eye_renderer.close()
        arena.eye_renderer = None
        initial = copy.deepcopy(arena.get_state())
        for i, (trial, source_time, heading, contrast) in enumerate(zip(
                protocol['trial'], protocol['source_timestamps'],
                protocol['heading'], protocol['proxy_contrast'])):
            if int(trial) != current_trial:
                current_trial = int(trial)
                arena.set_state(copy.deepcopy(initial))
                brain.reset()
                vision = EyeVision()
                previous_source = float(source_time) - .01
                trial_origin = previous_source
            dt = float(source_time) - previous_source
            previous_source = float(source_time)
            # The source clock determines the physical interval; no wall pacing.
            arena.step_behavior('WALK', [.4, .6], milliseconds=dt * 1000)
            body_yaw = float(arena.heading)
            cue_world = float(wrap(body_yaw + sign * heading))
            relative = float(cue_relative(body_yaw, cue_world))
            if abs(float(wrap(relative - sign * heading))) > 1e-10:
                raise AssertionError('cue frame does not match source heading')
            visual, _, info = vision.step(cue_images(relative, contrast,
                width=48 if source_stimulus else 64, source_stimulus=source_stimulus))
            sensory = np.zeros((1, 8), np.float32)
            sensory[0, 6:8] = visual
            counts = np.zeros(len(indices), np.float64)
            # The legacy pilot used four updates/frame; the source-informed mode
            # preserves the existing nominal 400 Hz model clock across sample rates.
            ticks = int(round(dt / tick_s)) if tick_s else 4
            if ticks < 1 or (tick_s and abs(ticks * tick_s - dt) > tick_s / 2 + 1e-10):
                raise ValueError('source frame cannot be aligned to neural clock')
            for _ in range(ticks):
                brain.step(sensory, ticks=1)
                counts += brain.spikes[indices, 0]
            rates.append(counts / dt)
            rows.append({'sample': i, 'source_frame': int(protocol['source_indices'][i]),
                'trial': current_trial, 'source_time_s': float(source_time),
                'source_elapsed_s': float(source_time) - trial_origin,
                'physical_time_s': arena.physics_steps * arena.sim.timestep,
                'body_yaw_rad': body_yaw, 'cue_world_rad': cue_world,
                'cue_relative_rad': relative, 'contrast_code': int(protocol['contrast_code'][i]),
                'proxy_contrast': float(contrast), 'inputs': sensory[0].tolist(),
                'position_mm': arena.position.tolist(), 'upright': float(arena.upright),
                'source_fly_heading_rad': float(heading),
                'epg_spikes_all_updates': counts.tolist(),
                'neural_updates': ticks,
                'model_update_interval_s_assumption': tick_s or dt / 4,
                'retinal_image_sha256': info['image_sha256']})
            if (i + 1) % 48 == 0:
                print(json.dumps({'recorded_frames': i + 1,
                                  'wall_s': time.monotonic() - started}), flush=True)
        np.savez_compressed(out / 'simulation.npz', source_indices=protocol['source_indices'],
            trial=protocol['trial'], timestamps=protocol['source_timestamps'],
            epg_root_ids=roots, epg_roster_indices=indices, spike_rate=rates)
        with (out / 'telemetry.jsonl').open('w') as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
        report = {'status': 'PASS', 'backend': 'full-connectome CPU oracle + full MuJoCo CPU body',
            'frames': len(rows), 'epg_neurons': len(roots), 'wall_seconds': time.monotonic() - started,
            'max_clock_error_s': max(abs(r['source_elapsed_s'] - r['physical_time_s']) for r in rows),
            'min_upright': min(r['upright'] for r in rows),
            'visual_input': 'analytic retinal cue through existing engineered lamina ports',
            'physical_actions': 'prescribed gait; neural output does not control body in this validation replay',
            'model_clock': ('fixed .0025 s engineering tick, not biologically calibrated'
                if tick_s else '4 dimensionless updates/source frame; dt/4 interpretation is uncalibrated'),
            'source_stimulus': source_stimulus,
            'neural_updates': sum(r['neural_updates'] for r in rows),
            'graph_weights': 'unchanged', 'protocol_sha256': digest(out / 'protocol.npz'),
            'annotation_sha256': digest(ANNOTATIONS)}
        write_json(out / 'recording.json', report)
        return report
    finally:
        if arena is not None:
            if arena.eye_renderer is None:
                # The upstream close method expects a renderer. Avoid double-close.
                if arena.body_renderer is not None:
                    arena.body_renderer.close()
            else:
                arena.close()


def calcium_filter(rates, timestamps, trials, decay_s=.6, rise_s=.05):
    """Cascaded stable exponential filters, reset at each trial boundary."""
    rates = np.asarray(rates, float)
    timestamps, trials = np.asarray(timestamps), np.asarray(trials)
    if rates.ndim != 2 or len(rates) != len(timestamps) or len(trials) != len(rates):
        raise ValueError('filter shape mismatch')
    if not np.isfinite(rates).all() or np.any(rates < 0) or decay_s <= 0 or rise_s <= 0:
        raise ValueError('invalid filter inputs')
    output = np.empty_like(rates)
    c = np.zeros(rates.shape[1]); f = c.copy()
    for i, rate in enumerate(rates):
        if i == 0 or trials[i] != trials[i - 1]:
            c[:] = 0; f[:] = 0; dt = .01
        else:
            dt = float(timestamps[i] - timestamps[i - 1])
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('nonpositive filter interval')
        a, b = np.exp(-dt / decay_s), np.exp(-dt / rise_s)
        c = a * c + (1 - a) * rate
        f = b * f + (1 - b) * c
        output[i] = f
    return output


def fit_observer(features, target, train, ridge=10.):
    """Train-only normalization and linear population-to-ROI measurement map."""
    mu = features[train].mean(axis=0)
    sigma = np.maximum(features[train].std(axis=0), 1e-6)
    x = np.column_stack([np.ones(len(features)), (features - mu) / sigma])
    penalty = np.eye(x.shape[1]) * ridge; penalty[0, 0] = 0
    weights = np.linalg.solve(x[train].T @ x[train] + penalty, x[train].T @ target[train])
    return {'mean': mu, 'scale': sigma, 'weights': weights}


def predict(features, observer):
    x = np.column_stack([np.ones(len(features)), (features - observer['mean']) / observer['scale']])
    return x @ observer['weights']


def r2(prediction, target, train, test):
    baseline = target[train].mean(axis=0)
    denom = np.sum((target[test] - baseline) ** 2, axis=0)
    valid = denom > 1e-12
    if not valid.all():
        raise ValueError('constant or degenerate ROI target')
    return 1 - np.sum((target[test] - prediction[test]) ** 2, axis=0) / denom


def evaluate(out):
    if any((out/name).exists() for name in ('evaluation.json','evaluation-consumption.json','frozen-observer.npz')):
        raise ValueError('final evaluation already consumed; use recorded receipt or a fresh split')
    p, simulation = np.load(out / 'protocol.npz'), np.load(out / 'simulation.npz')
    if not np.array_equal(p['source_indices'], simulation['source_indices']):
        raise ValueError('neural/behavior sample alignment mismatch')
    target, train, selection, final = p['calcium'], p['train'], p['selection'], p['final']
    source_stimulus = bool(p['source_stimulus']) if 'source_stimulus' in p else False
    evaluated = []
    # One bounded parameter-group change: observer decay, not anatomical dynamics.
    for decay in (.6, 1.2):
        features = calcium_filter(simulation['spike_rate'], simulation['timestamps'], simulation['trial'], decay)
        observer = fit_observer(features, target, train)
        predictions = predict(features, observer)
        selection_r2 = r2(predictions, target, train, selection)
        evaluated.append((decay, features, observer, predictions, selection_r2))
    proposed = float(np.median(evaluated[1][4])) > float(np.median(evaluated[0][4]))
    chosen = evaluated[int(proposed)]
    decay, features, observer, predictions, selection_r2 = chosen
    # Freeze before examining the final trial set.
    np.savez(out / 'frozen-observer.npz', **observer, decay_s=decay, rise_s=.05,
             epg_root_ids=simulation['epg_root_ids'], roi_indices=p['roi_indices'])
    frozen_hash = digest(out / 'frozen-observer.npz')
    metadata=json.loads((out/'protocol.json').read_text())
    consumption=consume_final(out,'nwb:'+Path(metadata['source_nwb']).name,
        p['source_indices'][final].astype(int).tolist(),frozen_hash)
    write_json(out/'evaluation-consumption.json',consumption)
    final_r2 = r2(predictions, target, train, final)
    rng = np.random.default_rng(20261001)
    null_scores = []
    for _ in range(99):
        shifted = features.copy()
        for trial in np.unique(p['trial']):
            ids = np.flatnonzero(p['trial'] == trial)
            # Within-trial circular shifts >= quarter-window in either direction.
            shift = rng.integers(max(1, len(ids) // 4), max(2, len(ids) * 3 // 4))
            shifted[ids] = np.roll(features[ids], int(shift), axis=0)
        null_observer = fit_observer(shifted, target, train)
        null_scores.append(r2(predict(shifted, null_observer), target, train, final))
    null_scores = np.asarray(null_scores)
    population_p = float((1 + (np.median(null_scores, axis=1) >= np.median(final_r2)).sum()) / 100)
    heading = p['heading']
    heading_features = np.column_stack([np.sin(heading), np.cos(heading),
        np.sin(2 * heading), np.cos(2 * heading), p['proxy_contrast']])
    heading_observer = fit_observer(heading_features, target, train)
    heading_r2 = r2(predict(heading_features, heading_observer), target, train, final)
    # Offline measurement fit is never published into the active simulator.
    gate = bool(np.median(final_r2) > 0 and population_p <= .05 and
                np.median(final_r2) > np.median(heading_r2))
    accepted_path = out / 'accepted-observer.npz'
    if gate:
        # An accepted *measurement* model remains distinct from neural dynamics.
        accepted_path.write_bytes((out / 'frozen-observer.npz').read_bytes())
    elif accepted_path.exists():
        accepted_path.unlink()
    np.savez_compressed(out / 'comparison.npz', predictions=predictions,
        final_r2=final_r2, selection_r2=selection_r2, null_r2=null_scores,
        heading_final_r2=heading_r2, target=target, final_mask=final)
    with (out / 'comparison.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['nwb_roi_index', 'selection_r2', 'final_r2', 'heading_baseline_final_r2'])
        writer.writerows(zip(p['roi_indices'], selection_r2, final_r2, heading_r2))
    report = {'status': 'PASS', 'measurement_gate': 'PASS' if gate else 'FAIL',
        'baseline_decay_s': .6, 'candidate_decay_s': 1.2,
        'candidate_selected_on_selection_only': proposed, 'frozen_decay_s': decay,
        'selection_baseline_median_r2': float(np.median(evaluated[0][4])),
        'selection_candidate_median_r2': float(np.median(evaluated[1][4])),
        'final_median_r2': float(np.median(final_r2)),
        'final_positive_rois': int((final_r2 > 0).sum()),
        'heading_baseline_final_median_r2': float(np.median(heading_r2)),
        'population_time_shift_p': population_p, 'controls': 99,
        'frozen_observer_sha256': frozen_hash,
        'observation_model': 'two exponential filters (.05 s rise; .6 or 1.2 s decay) + train-only ridge map from 47 annotated EPG units to 42 ROIs; hypothesis, not calibrated GCaMP7f kinetics',
        'split': 'first 3 short trial windows train/selection; last 3 windows final, no final-driven fitting',
        'limits': ('one fly, reset short windows, coarse visual ports, grayscale digital-level proxy without optical calibration, imaging-rate stimulus holds, nominal engineering clock without biological calibration, unknown ROI-to-neuron mapping; shifted controls not biological replicates'
            if source_stimulus else 'one fly, short windows, coarse visual ports, proxy contrast/geometry, assumed heading sign and neural clock, unknown ROI-to-neuron mapping; shifted controls not biological replicates'),
        'active_simulator_changed': False,
        'accepted_measurement_model': str(accepted_path) if gate else None,
        'expression_parameter_update': 'DEFERRED: source X normalization and authoritative population crosswalk unverified',
        'experiment_id':consumption['experiment_id'],'final_consumption_ledger':consumption['ledger'],
        'data_scope':metadata.get('data_scope','recorded calcium; source in protocol.json'),
        'protocol_sha256': digest(out / 'protocol.npz'),
        'simulation_sha256': digest(out / 'simulation.npz')}
    write_json(out / 'evaluation.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'record', 'evaluate'])
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--frames-per-trial', type=int, default=48)
    parser.add_argument('--window-start-offset', type=float, default=1.)
    parser.add_argument('--source-matched', action='store_true',
                        help='source-informed stripe geometry/levels/sign; fixed engineering neural clock')
    args = parser.parse_args()
    report = prepare(args.output, args.frames_per_trial, args.window_start_offset, args.source_matched) if args.stage == 'prepare' else (
        record(args.output) if args.stage == 'record' else evaluate(args.output))
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
