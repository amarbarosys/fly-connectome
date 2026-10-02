"""Check the complete frozen body suite and actual-control receipts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(body, controls, replay, output):
    metrics = json.loads((body/'metrics.json').read_text())
    manifest = Path(metrics['suite_manifest'])
    frozen = json.loads(manifest.read_text())
    assert digest(manifest) == metrics['suite_manifest_sha256']
    assert frozen['suite']['seeds'] == list(range(20))
    assert frozen['suite']['floor_seconds'] == 60.
    assert frozen['suite']['wall_seconds'] == 5.
    for source, expected in frozen['source_sha256'].items():
        assert digest(source) == expected, source
    assert metrics['suite'] == 'full' and metrics['status'] == 'PASS'
    assert len(metrics['rows']) == 40
    expected = {(seed, scenario) for seed in range(20) for scenario in ('central_floor', 'east_wall')}
    assert {(row['seed'], row['scenario']) for row in metrics['rows']} == expected
    hashes = {}
    upright = []
    escapes = []
    exchanges = 0
    for seed, scenario in sorted(expected):
        path = body/f'{seed:02d}-{scenario}.json'
        row = json.loads(path.read_text())
        hashes[str(path)] = digest(path)
        trace = row['trajectory']
        assert len(trace) == (6000 if scenario == 'central_floor' else 500)
        assert all(point['finite'] for point in trace)
        assert all(np.isfinite(point['position_mm']).all() and np.isfinite(point['upright']) for point in trace)
        assert all(abs(point['time_s']-(i+1)*.01) < 1e-10 for i, point in enumerate(trace))
        assert row['reset_replay'] and row['terrain'] == frozen['terrain_receipt']
        fraction = sum(point['upright'] > .85 for point in trace)/len(trace)
        assert fraction == row['upright_fraction']
        exchanges += len(trace)
        if scenario == 'central_floor':
            assert fraction >= .95 and row['failure'] is None
            upright.append(fraction)
        else:
            assert row['wall_contact_exchanges'] > 0
            assert row['failure'] not in ('nonfinite_physics', 'wall_fixture_no_contact', 'reset_replay_mismatch')
            if row['wall_escape_seconds'] is not None:
                assert 0 < row['wall_escape_seconds'] <= 5.
                escapes.append(row['wall_escape_seconds'])
    assert len(escapes)/20 >= .9
    ui = json.loads((controls/'metrics.json').read_text())
    assert ui['status'] == 'PASS' and ui['post_reset_exchanges'] >= 10
    check = ui['reset_check']
    for key in ('physical_click', 'full_body_controller_state_matches_initial',
                'neural_was_nonzero', 'readout_weights_preserved', 'decision_idle'):
        assert check[key], key
    assert check.get('neural_zero_observables', check.get('neural_arrays_zero'))
    codes = [row['control_code'] for row in map(json.loads, (controls/'run/events.jsonl').read_text().splitlines())
             if 'control_code' in row]
    assert codes == [32, 78, 32, 999]
    assert (controls/'controls.png').is_file()
    assert 'Reset body / brain' in (controls/'controls-ocr.txt').read_text()
    neural = json.loads((replay/'metrics.json').read_text())
    assert neural['status'] == 'PASS' and neural['learning_updates_in_prefix'] > 0
    assert neural['exact_event_and_final_state_replay'] and neural['finite_physics']
    report = {'status': 'PASS', 'mode': 'engineering_agent',
              'scope': 'flat-body stability, bounded wall escape and actual body/brain UI reset; not full neural-loop or biological acceptance',
              'seeds': 20, 'floor_seconds_per_seed': 60, 'wall_seconds_per_seed': 5,
              'minimum_floor_upright_fraction': min(upright),
              'successful_wall_escapes': len(escapes), 'wall_escape_fraction': len(escapes)/20,
              'wall_escape_seconds_min': min(escapes), 'wall_escape_seconds_max': max(escapes),
              'finite_exchanges_checked': exchanges, 'actual_control_reset': 'PASS',
              'controls_backend': ui['backend'],
              'embodied_learning_replay': 'PASS',
              'body_receipt': str(body/'metrics.json'), 'controls_receipt': str(controls/'metrics.json'),
              'replay_receipt': str(replay/'metrics.json'), 'suite_manifest_sha256': digest(manifest),
              'trajectory_sha256': hashes, 'verifier_sha256': digest(__file__)}
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'trajectory_sha256'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--body', type=Path, required=True)
    parser.add_argument('--controls', type=Path, required=True)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    verify(args.body, args.controls, args.replay, args.output)
