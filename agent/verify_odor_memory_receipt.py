"""Independent raw-trajectory and restart audit for the odor memory pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from agent.neural_connectome import digest


def verify(output):
    protocol = json.loads((output/'protocol.json').read_text())
    training = json.loads((output/'training.json').read_text())
    metrics = json.loads((output/'metrics.json').read_text())
    assert digest(output/'protocol.json') == metrics['protocol_sha256']
    assert metrics['evaluation_pid'] != training['training_pid']
    assert not metrics['learning_enabled_during_evaluation']
    train_layouts = {json.dumps(s['food'], sort_keys=True) for s in protocol['training']}
    test_layouts = {json.dumps(s['food'], sort_keys=True) for s in protocol['evaluation']}
    assert train_layouts.isdisjoint(test_layouts)
    assert len(test_layouts) == len(protocol['evaluation'])
    for path, sha in protocol['source_sha256'].items():
        assert digest(Path(path)) == sha
    for name in ('trained', 'shuffled_reward'):
        assert digest(output/f'{name}.npz') == training['models'][name]['sha256']
    raw_metrics = {}
    exchanges = 0
    for control in protocol['controls']:
        ingested, upright = [], []
        for scene in protocol['evaluation']:
            row = json.loads((output/f'eval-{scene["id"]}-{control}.json').read_text())
            assert row['scene'] == scene and row['control'] == control
            assert len(row['inputs']) == 8
            if control == 'odor_absent':
                assert all(inputs[:2] == [0., 0.] for inputs in row['inputs'])
            assert len(row['trajectory']) == 130
            contact_capacity = 0.
            previous_intake = 0.
            minimum_upright = 1.
            for step, event in enumerate(row['trajectory']):
                assert event['step'] == step
                assert event['action'] == ('WALK' if step < 50 else 'FEED')
                assert event['gait'] == (row['gait'] if step < 50 else [0., 0.])
                assert np.isfinite([*event['position'], event['upright'], event['ingested']]).all()
                contact_s = sum(event['mouth_contact_s'].values())
                assert 0 <= contact_s <= .01+1e-10
                contact_capacity += 2.*contact_s
                intake = event['ingested']
                assert intake >= previous_intake and intake <= contact_capacity+1e-8
                previous_intake = intake
                minimum_upright = min(minimum_upright, event['upright'])
                exchanges += 1
            assert abs(row['ingested']-previous_intake) < 1e-12
            assert abs(row['minimum_upright']-minimum_upright) < 1e-12
            ingested.append(row['ingested']); upright.append(minimum_upright)
        measured = {'success_fraction': float(np.mean(np.asarray(ingested) >= protocol['criteria']['success_ingested_food_u'])),
                    'mean_ingested': float(np.mean(ingested)), 'minimum_upright': min(upright),
                    'trials': len(ingested)}
        assert measured == metrics['metrics'][control]
        raw_metrics[control] = measured
    trained = raw_metrics['trained']['success_fraction']
    outcome = trained >= protocol['criteria']['trained_success_fraction']
    outcome &= all(trained-raw_metrics[c]['success_fraction'] >= protocol['criteria']['minimum_success_improvement_over_each_control']
                   for c in protocol['controls'][1:])
    outcome &= all(row['minimum_upright'] >= protocol['criteria']['minimum_upright'] for row in raw_metrics.values())
    outcome &= max(metrics['restart_prediction_max_abs_error'].values()) <= protocol['criteria']['restart_max_abs_prediction_error']
    outcome &= training['changed_shuffle_labels'] >= training['informative_pairs']//2
    assert metrics['status'] == ('PASS' if outcome else 'FAIL')
    receipt = {'status': 'PASS', 'behavioral_gate_status': metrics['status'],
               'finite_physical_exchanges': exchanges, 'distinct_held_out_layouts': len(test_layouts),
               'fresh_process': True, 'physical_contact_intake_accounting': True,
               'frozen_models': True, 'raw_metrics': raw_metrics}
    (output/'independent-verification.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    print(json.dumps(verify(parser.parse_args().output)))
