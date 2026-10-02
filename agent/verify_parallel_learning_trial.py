"""Independent receipt checks for matched experience and frozen policy scoring."""
import json
from pathlib import Path

import numpy as np

from agent.neural_connectome import digest

ROOT = Path('.work/agent/spectator-learning')


def verify():
    protocol = json.loads((ROOT/'learning/protocol.json').read_text())
    comparison = json.loads((ROOT/'learning/comparison.json').read_text())
    held_out = np.load(ROOT/'learning/held-out.npz')
    training_ids = {scene['id'] for rows in protocol['training'].values() for scene in rows}
    assert not training_ids.intersection(scene['id'] for scene in protocol['held_out'])
    assert len(protocol['held_out']) == 10
    maximum_error = 0.
    physical_exchanges = 0
    for seed in protocol['seeds']:
        arrays = {}
        for count in (1,10):
            directory = ROOT/f'learning/seed-{seed}-c{count}'
            metrics = json.loads((directory/'metrics.json').read_text())
            data = np.load(directory/'samples.npz')
            assert data['features'].shape == (10,2606)
            assert data['inputs'].shape == (10,8,8)
            assert data['rewards'].shape == (10,2)
            assert all(np.isfinite(data[name]).all() for name in data.files)
            assert metrics['scenes'] == protocol['training'][str(seed)]
            assert metrics['physical_body_exchanges'] == 2680
            physical_exchanges += metrics['physical_body_exchanges']
            outcomes = json.loads((directory/'physical-outcomes.json').read_text())
            observed = []
            for group in outcomes:
                rewards = []
                for arm in group:
                    assert min(arm['minimum_upright']) >= .85
                    assert all(row['physics_steps'] == 13800 for row in arm['bodies'])
                    rewards.append([row['ingested'] for row in arm['bodies']])
                observed.extend(np.asarray(rewards).T)
            assert np.array_equal(data['rewards'],observed)
            arrays[count] = data
        assert np.array_equal(arrays[1]['inputs'],arrays[10]['inputs'])
        assert np.array_equal(arrays[1]['rewards'],arrays[10]['rewards'])
        maximum_error = max(maximum_error,float(np.max(np.abs(arrays[1]['features']-arrays[10]['features']))))
    assert maximum_error <= 1e-5
    for row in comparison['results']:
        path = ROOT/f"learning/seed-{row['seed']}-c{row['count']}-{row['criterion']}.npz"
        assert digest(path) == row['model_sha256']
        model = np.load(path)
        f = held_out['features']
        predictions = np.column_stack(((f-model['mean'])/model['scale'],np.ones(len(f))))@model['weights']
        choices = (predictions[:,1]<=predictions[:,0]).astype(int)
        assert choices.tolist() == row['choices']
        intake = held_out['rewards'][np.arange(10),choices]
        assert int(np.count_nonzero(intake>=.05)) == row['successes']
        data = np.load(ROOT/f"learning/seed-{row['seed']}-c{row['count']}/samples.npz")
        expected = 10 if row['criterion']=='equal_experience' else int(np.count_nonzero(
            data['completion_wall_seconds']<=row['collection_wall_budget_seconds']))
        assert expected == row['scenes']
    viewer = json.loads((ROOT/'spectator-comparison.json').read_text())
    times = {v:[row['wall_seconds_measured'] for row in viewer['pairs'] if row['visible']==v] for v in (False,True)}
    assert all(len(values)==2 for values in times.values())
    spectator = json.loads((ROOT/'visible/metrics.json').read_text())
    assert spectator['learning_updates'] == 0
    assert all(row['minimum_upright']>=.85 for row in spectator['episodes'])
    for row in viewer['pairs']:
        bodies = json.loads((ROOT/f"paired-{row['round']}-{int(row['visible'])}/exchanges.json").read_text())
        assert len(bodies)==23
        assert abs(sum(step['loop_s'] for step in bodies[3:])-row['wall_seconds_measured'])<1e-9
    report = {'status':'PASS','training_physical_body_exchanges':physical_exchanges,
              'training_neural_batch_feature_error':maximum_error,
              'held_out_unique_layouts':10,'seed_layout_policy_scores':len(comparison['results'])*10,
              'viewer_overhead_fraction':float(np.mean(times[True])/np.mean(times[False])-1),
              'scope':'matched independent arena experience and offline frozen policy choice over actual physical branches'}
    optimized_root = ROOT/'optimized-viewer'
    optimized = json.loads((optimized_root/'spectator-comparison.json').read_text())
    optimized_times = {v:[row['wall_seconds_measured'] for row in optimized['pairs'] if row['visible']==v]
                       for v in (False,True)}
    for row in optimized['pairs']:
        bodies = json.loads((optimized_root/f"paired-{row['round']}-{int(row['visible'])}/exchanges.json").read_text())
        assert len(bodies)==23
        assert abs(sum(step['loop_s'] for step in bodies[3:])-row['wall_seconds_measured'])<1e-9
    report['optimized_viewer_overhead_fraction'] = float(np.mean(optimized_times[True])/np.mean(optimized_times[False])-1)
    reload = json.loads((ROOT/'policy-reload-verification.json').read_text())
    assert reload['status']=='PASS' and reload['restored_model_observed']
    report['read_only_snapshot_reload_verified'] = True
    (ROOT/'independent-verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    verify()
