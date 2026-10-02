"""Audit raw collision rewards, food retention and saved sequential models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
from agent.neural_connectome import Readout, digest


def verify(output):
    protocol = json.loads((output/'protocol.json').read_text())
    training = json.loads((output/'training.json').read_text())
    metrics = json.loads((output/'metrics.json').read_text())
    food = Path(protocol['food_gate'])
    assert digest(food/'metrics.json') == protocol['food_metrics_sha256']
    assert digest(food/'trained.npz') == protocol['food_model_sha256']
    assert metrics['evaluation_pid'] != training['training_pid']
    assert not metrics['learning_during_evaluation']
    assert training['rehearsal_bytes'] <= protocol['rehearsal_budget_bytes']
    for path, sha in protocol['source_sha256'].items():
        assert digest(Path(path)) == sha
    x = np.load(output/'restart-fixture.npz')['features']
    for name in ('trained', 'shuffled_reward'):
        assert digest(output/f'{name}.npz') == training['models'][name]['sha256']
        predicted = Readout(output/f'{name}.npz').predict(x)
        assert np.array_equal(predicted, np.asarray(training['models'][name]['predictions']))
    train_layouts = {json.dumps(s['hazard'], sort_keys=True) for s in protocol['training']}
    test_layouts = {json.dumps(s['hazard'], sort_keys=True) for s in protocol['evaluation']}
    assert train_layouts.isdisjoint(test_layouts)
    count = 0
    raw = {}
    all_upright = []
    for control in protocol['controls']:
        contacts, upright = [], []
        for scene in protocol['evaluation']:
            row = json.loads((output/f'eval-{scene["id"]}-{control}.json').read_text())
            assert row['scene'] == scene
            assert len(row['trajectory']) == 50 and len(row['eye_image_sha256']) == 8
            assert len(row['inputs']) == 8
            for index, event in enumerate(row['trajectory']):
                assert event['step'] == index
                assert np.isfinite([*event['position'], event['upright'], event['hazard_normal_force']]).all()
                assert event['hazard_normal_force'] >= 0.
                count += 1
            contact_exchanges = sum(event['hazard_normal_force'] > .01 for event in row['trajectory'])
            minimum = min(event['upright'] for event in row['trajectory'])
            assert contact_exchanges == row['contact_exchanges']
            assert row['reward'] == -float(contact_exchanges)
            assert row['minimum_upright'] == minimum
            contacts.append(contact_exchanges); upright.append(minimum)
        measured = {'success_fraction': float(np.mean(np.asarray(contacts) <= protocol['criteria']['maximum_contact_exchanges'])),
                    'mean_contact_exchanges': float(np.mean(contacts)), 'minimum_upright': min(upright)}
        assert measured == metrics['avoidance'][control]
        raw[control] = measured; all_upright.extend(upright)
    food_protocol = json.loads((food/'protocol.json').read_text())
    ingested = []
    for scene in food_protocol['evaluation']:
        row = json.loads((output/f'food-retention-{scene["id"]}.json').read_text())
        assert row['scene'] == scene and len(row['trajectory']) == 130
        contact_limit = 0.; previous = 0.
        for index, event in enumerate(row['trajectory']):
            assert event['step'] == index
            assert np.isfinite([*event['position'], event['upright'], event['ingested']]).all()
            duration = sum(event['mouth_contact_s'].values())
            assert 0 <= duration <= .01+1e-10
            contact_limit += 2.*duration
            assert previous <= event['ingested'] <= contact_limit+1e-8
            previous = event['ingested']; all_upright.append(event['upright']); count += 1
        assert abs(previous-row['ingested']) <= 1e-12
        ingested.append(previous)
    success = float(np.mean(np.asarray(ingested) >= food_protocol['criteria']['success_ingested_food_u']))
    assert success == metrics['food_success_after']
    c = protocol['criteria']; trained = raw['trained']['success_fraction']
    checks = {'avoidance_success': trained >= c['avoid_success_fraction'],
        'beats_each_control': all(trained-raw[name]['success_fraction'] >= c['minimum_success_improvement'] for name in protocol['controls'][1:]),
        'food_retention': success >= metrics['food_success_before']-c['maximum_food_success_regression'],
        'body_upright': min(all_upright) >= c['minimum_upright'], 'restart_exact': True, 'evaluation_models_unchanged': True}
    assert checks == metrics['checks']
    assert metrics['status'] == ('PASS' if all(checks.values()) else 'FAIL')
    receipt = {'status': 'PASS', 'behavioral_gate_status': metrics['status'],
        'finite_physical_exchanges': count, 'raw_avoidance_metrics': raw,
        'food_success_after': success, 'rehearsal_bytes': training['rehearsal_bytes'],
        'contact_rewards_and_food_intake_accounting': True, 'fresh_process': True,
        'models_unchanged': True, 'held_out_avoidance_layouts': len(test_layouts)}
    (output/'independent-verification.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    print(json.dumps(verify(parser.parse_args().output)))
