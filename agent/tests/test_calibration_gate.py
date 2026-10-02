import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'agent' / 'sequential_memory_gate.py'
PYTHON = ROOT / '.venv' / 'bin' / 'python'


def _run_freeze(tmp_path, calibration):
    output = tmp_path / 'output'
    output.mkdir()
    if calibration is not None:
        (output / 'calibration.json').write_text(json.dumps(calibration))
    else:
        output.rmdir()
    food = tmp_path / 'food'
    food.mkdir()
    (food / 'metrics.json').write_text(json.dumps({'status': 'PASS'}))
    (food / 'trained.npz').write_bytes(b'synthetic-model')
    (food / 'protocol.json').write_text('{}\n')

    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT)
    env['PYTHONPATH'] = str(ROOT / 'agent/tests/fixtures') + os.pathsep + env['PYTHONPATH']
    result = subprocess.run(
        [str(PYTHON), str(SCRIPT), 'freeze', '--output', str(output), '--food-gate', str(food)],
        cwd=ROOT, env=env, text=True, capture_output=True,
    )
    return result, output


FEASIBLE = {
    'seed_values': [400],
    'upright_minimum': 0.99,
    'arm_results': [
        {'arm': 'straight', 'success_count': 6, 'total': 16, 'success_rate': 0.375,
         'gait_parameters_observed': [0.425, 0.425]},
        {'arm': 'shuffled', 'success_count': 0, 'total': 16, 'success_rate': 0.0},
    ],
    'correct_turn_diagnostic': {'success_count': 14, 'total': 16, 'success_rate': 0.875},
    'arm_definitions': {'shuffled': {'assignment': 'test', 'gaits': {
        'left': [0.2, 0.65], 'right': [0.65, 0.2]}}},
}


def test_feasible_criterion_freezes(tmp_path):
    result, output = _run_freeze(tmp_path, FEASIBLE)
    assert result.returncode == 0, result.stdout + result.stderr
    protocol = json.loads((output / 'protocol.json').read_text())
    assert protocol['criteria']['minimum_success_improvement'] == 0.25
    assert protocol['calibration_feasibility']['checked_against']['straight']['headroom_percentage_points'] == 50.0
    assert protocol['calibration_feasibility']['ceiling_success_percentage'] == 87.5


def test_infeasible_criterion_refuses_without_writing_protocol(tmp_path):
    calibration = json.loads(json.dumps(FEASIBLE))
    calibration['arm_results'][0]['success_count'] = 14
    calibration['arm_results'][0]['success_rate'] = 0.875
    calibration['correct_turn_diagnostic'] = {'success_count': 16, 'total': 16, 'success_rate': 1.0}
    result, output = _run_freeze(tmp_path, calibration)
    assert result.returncode != 0
    assert 'FAIL freeze: criterion unreachable, straight=87.5%, ceiling=100% leaves 12.5pp, need 25pp' in result.stdout
    assert not (output / 'protocol.json').exists()


def test_60_percent_baseline_with_70_percent_ceiling_refuses(tmp_path):
    calibration = json.loads(json.dumps(FEASIBLE))
    calibration['arm_results'][0].update(success_count=60, total=100, success_rate=0.60)
    calibration['correct_turn_diagnostic'] = {'success_count': 70, 'total': 100, 'success_rate': 0.70}
    result, output = _run_freeze(tmp_path, calibration)
    assert result.returncode != 0
    assert 'FAIL freeze: criterion unreachable, straight=60%, ceiling=70% leaves 10pp, need 25pp' in result.stdout
    assert not (output / 'protocol.json').exists()
    assert not (output / 'protocol.json').exists()


def test_missing_calibration_refuses_without_writing_protocol(tmp_path):
    result, output = _run_freeze(tmp_path, None)
    assert result.returncode != 0
    assert 'FAIL freeze: no calibration.json, run calibrate first' in result.stdout
    assert not (output / 'protocol.json').exists()
