"""Measurement semantics, reset, leakage and saved-run checks."""
from pathlib import Path
import json
import numpy as np

from agent.calcium_validation import (DEFAULT, calcium_filter, cue_images,
    cue_relative, digest, fit_observer, predict, r2, wrap)


def mechanics():
    # Direction remains continuous when physical yaw crosses +/- pi.
    yaw = np.asarray([0., np.pi - .01, -np.pi + .01])
    relative = np.asarray([0., .03, -.03])
    assert np.allclose(cue_relative(yaw, wrap(yaw + relative)), relative)
    assert np.array_equal(cue_images(.2, 0), cue_images(-2, 0))
    assert not np.array_equal(cue_images(.8, 1), cue_images(-.8, 1))
    assert not cue_images(1., 0., source_stimulus=True).any()
    bright = cue_images(0., 1., width=96, source_stimulus=True)
    dim = cue_images(0., 1. / 15, width=96, source_stimulus=True)
    assert bright.max() == 255 and dim.max() == 17
    assert np.array_equal(bright > 0, dim > 0)
    panorama = cue_images(0., 1., width=48, source_stimulus=True)
    assert np.count_nonzero(panorama[:, 0, :, 0]) == 2
    for angle in np.linspace(-np.pi, np.pi, 96, endpoint=False):
        assert np.count_nonzero(cue_images(angle, 1., width=48, source_stimulus=True)[:, 0, :, 0]) == 2
    # A single impulse decays; resetting a trial removes previous activity.
    t = np.arange(200) * .1
    rate = np.zeros((200, 2)); rate[0] = 100
    trials = np.repeat([0, 1], 100)
    filtered = calcium_filter(rate, t, trials)
    assert np.isfinite(filtered).all() and (filtered >= 0).all()
    assert filtered[20].max() < filtered[2].max()
    assert np.array_equal(filtered[100:], np.zeros((100, 2)))
    # A held-out target edit cannot change normalization or fitted coefficients.
    rng = np.random.default_rng(11)
    x = rng.normal(size=(90, 5)); y = x @ rng.normal(size=(5, 3)) + 2
    train = np.arange(90) < 50; test = ~train
    first = fit_observer(x, y, train)
    altered = y.copy(); altered[test] += 1000
    second = fit_observer(x, altered, train)
    for key in first:
        assert np.array_equal(first[key], second[key])
    assert np.median(r2(predict(x, first), y, train, test)) > .9
    # Invalid within-trial time cannot silently propagate.
    try:
        calcium_filter(np.ones((2, 1)), [0., 0.], [0, 0])
    except ValueError:
        pass
    else:
        raise AssertionError('duplicate time accepted')
    return {'mechanics': 'PASS', 'checks': ['cue wrap/direction and zero contrast',
        'impulse decay and trial reset', 'held-out target leakage isolation',
        'synthetic population mapping', 'invalid-clock rejection']}


def saved(out):
    p = np.load(out / 'protocol.npz'); s = np.load(out / 'simulation.npz')
    rows = [json.loads(line) for line in (out / 'telemetry.jsonl').read_text().splitlines()]
    assert len(rows) == len(p['source_indices']) == len(s['spike_rate'])
    assert np.array_equal(p['source_indices'], s['source_indices'])
    assert np.array_equal(p['source_timestamps'], s['timestamps'])
    assert set(p['trial'][p['train']]) == {0, 1, 2}
    assert set(p['trial'][p['final']]) == {3, 4, 5}
    assert not (p['train'] & p['selection']).any()
    assert not ((p['train'] | p['selection']) & p['final']).any()
    roster = np.load('data/proofread_root_ids_783.npy')
    assert np.array_equal(roster[s['epg_roster_indices']], s['epg_root_ids'])
    for i, row in enumerate(rows):
        assert row['source_frame'] == p['source_indices'][i]
        sign = int(p['cue_heading_sign']) if 'cue_heading_sign' in p else 1
        assert abs(wrap(row['cue_relative_rad'] - sign * p['heading'][i])) < 1e-10
        assert abs(cue_relative(row['body_yaw_rad'], row['cue_world_rad']) - row['cue_relative_rad']) < 1e-10
        assert abs(row['source_elapsed_s'] - row['physical_time_s']) < .001
        assert np.isfinite(row['position_mm']).all()
        assert np.all(np.asarray(row['inputs'])[:6] == 0)
        k = int(p['source_indices'][i])
        if 'replay_timestamps' in p:
            # Continuous replay: every frame follows the previous source sample.
            assert p['replay_heading'][k] == p['heading'][i] and p['replay_contrast'][k] == p['proxy_contrast'][i]
            dt = p['replay_timestamps'][k] - p['replay_timestamps'][k - 1] if k else .01
        elif i == 0 or p['trial'][i] != p['trial'][i - 1]:
            dt = .01
        else:
            dt = s['timestamps'][i] - s['timestamps'][i - 1]
        counts = row.get('epg_spikes_all_updates', row.get('epg_spikes_all_four_updates'))
        assert np.allclose(s['spike_rate'][i], np.asarray(counts) / dt)
        tick_s = float(p['neural_tick_s']) if 'neural_tick_s' in p else 0.
        if tick_s:
            assert row['neural_updates'] == round(dt / tick_s)
            assert abs(row['neural_updates'] * tick_s - dt) < 1e-8
    model = dict(np.load(out / 'frozen-observer.npz'))
    features = calcium_filter(s['spike_rate'], s['timestamps'], s['trial'], float(model['decay_s']), float(model['rise_s']))
    scores = r2(predict(features, model), p['calcium'], p['train'], p['final'])
    comparison = np.load(out / 'comparison.npz')
    assert np.array_equal(scores, comparison['final_r2'])
    evaluation = json.loads((out / 'evaluation.json').read_text())
    assert digest(out / 'frozen-observer.npz') == evaluation['frozen_observer_sha256']
    assert float(np.median(scores)) == evaluation['final_median_r2']
    return {'saved_run': 'PASS', 'frames': len(rows), 'root_id_alignment': 'exact',
        'clock_alignment': 'within 1 ms', 'splits': 'disjoint',
        'final_scores': 'recomputed from frozen observer',
        'measurement_gate': evaluation['measurement_gate']}


def neural_clock():
    """Sampling batches cannot change the state after the same 40 updates."""
    from agent.neural_connectome import CPUBrain
    brain = CPUBrain(flies=1)
    sensory = np.zeros((1, 8), np.float32); sensory[0, 6:] = .4
    for _ in range(40):
        brain.step(sensory, ticks=1)
    first = [x.copy() for x in (brain.v, brain.trace, brain.spikes, brain.ever)]
    brain.reset()
    for _ in range(10):
        brain.step(sensory, ticks=4)
    assert all(np.array_equal(x, y) for x, y in zip(first,
                   (brain.v, brain.trace, brain.spikes, brain.ever)))
    return {'neural_clock': 'PASS', 'check': '40 single updates equal ten four-update batches exactly'}


def continuity():
    """Brain state carries across windows: each window starts where the replay left it, not at zero."""
    import tempfile
    from agent.calcium_validation import ROOT, record
    t = np.arange(8) * .1; heading = np.linspace(-1., 1., 8); frames = np.asarray([2, 3, 6, 7])
    with tempfile.TemporaryDirectory(dir=ROOT / '.work') as tmp:
        np.savez(Path(tmp) / 'protocol.npz', replay_timestamps=t, replay_heading=heading,
            replay_contrast=np.ones(8), source_indices=frames, source_timestamps=t[frames],
            heading=heading[frames], proxy_contrast=np.ones(4), trial=np.asarray([0, 0, 1, 1]),
            contrast_code=np.full(4, 3), cue_heading_sign=-1, neural_tick_s=.0025, source_stimulus=True)
        states = record(Path(tmp))['window_initial_state']
    assert [w['trial'] for w in states] == [0, 1]
    assert all(w['state_l1'] > 0 for w in states), states
    return {'continuity': 'PASS', 'window_initial_state': states}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved', type=Path)
    parser.add_argument('--neural-clock', action='store_true')
    parser.add_argument('--continuity', action='store_true')
    args = parser.parse_args()
    report = mechanics()
    if args.neural_clock:
        report.update(neural_clock())
    if args.continuity:
        report.update(continuity())
    if args.saved:
        report.update(saved(args.saved))
        (args.saved / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
