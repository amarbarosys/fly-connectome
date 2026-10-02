"""Matched ingestion-labelled steering experience, serial versus ten arenas.

Both actions are physically tried from the same saved sensory prefix. Ridge
fitting runs outside the GPU lease. This is the existing contextual-bandit task.
"""
import argparse
import json
from multiprocessing.connection import Client
from pathlib import Path
import time

import numpy as np

from agent.neural_connectome import CPUBrain, Readout, Worker, digest
from agent.odor_memory_gate import GAITS, scenes

ROOT = Path('.work/agent/spectator-learning')
SOCKET = ROOT/'physics.sock'
SEEDS = (11, 22, 33)


def request(connection, op, values):
    connection.send({'op': op, 'actors': list(enumerate(values))})
    response = connection.recv()
    if 'error' in response:
        raise RuntimeError(response['error'])
    return response['rows']


def experience(connection, brain, configs, continuous=False):
    count = len(configs)
    physical = request(connection, 'configure', configs)
    brain.reset()
    inputs = []
    for _ in range(8):
        values = np.asarray([row['inputs'] for row in physical], np.float32)
        inputs.append(values)
        neural = brain.step(values)
        physical = request(connection, 'step', [('IDLE', [0.,0.])]*count)
    features = np.asarray(neural['features'], np.float64)
    request(connection, 'checkpoint', [None]*count)
    outcomes = []
    for name in ('left', 'right'):
        physical = request(connection, 'reset', [None]*count)
        if continuous:
            brain.reset()
            for values in inputs:
                brain.step(values)
        minima = np.ones(count)
        for step in range(130):
            if continuous:
                neural = brain.step(np.asarray([row['inputs'] for row in physical],np.float32))
            action = 'WALK' if step < 50 else 'FEED'
            gait = GAITS[name] if action == 'WALK' else [0.,0.]
            physical = request(connection, 'step', [(action, gait)]*count)
            minima = np.minimum(minima, [row['upright'] for row in physical])
        assert np.min(minima) >= .85
        outcome = {'name': name, 'bodies': physical, 'minimum_upright': minima.tolist()}
        if continuous:
            outcome['neural_counts'] = neural['counts']
        outcomes.append(outcome)
    rewards = np.asarray([[row['ingested'] for row in outcome['bodies']] for outcome in outcomes]).T
    return features, rewards, np.asarray(inputs), outcomes


def freeze():
    output = ROOT/'learning'
    output.mkdir(parents=True, exist_ok=False)
    protocol = {'status': 'FROZEN', 'seeds': list(SEEDS),
        'training': {str(seed): scenes(range(1000+seed*10, 1005+seed*10)) for seed in SEEDS},
        'held_out': scenes(range(9000,9005)), 'equal_experience_scenes': 10,
        'arms': [1,10], 'physical_initialization_seed': 0,
        'reward': 'actual physical ingestion, both turn primitives from identical sensory checkpoint',
        'fit': 'ridge residual over straight gait, regularization 1; identical for both arms',
        'wall_budget': 'serial samples completed by the ten-arena collection wall time; fit/evaluation excluded',
        'criteria': {'success_ingested': .05, 'minimum_upright': .85,
                     'minimum_success_gain_for_learning_advantage': .25},
        'scope': 'bounded one-decision odor task, no general RL or biological claim',
        'source_sha256': {str(path): digest(path) for path in [Path(__file__), Path('agent/batch_arenas.py')]}}
    (output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')


def collect(preflight=False):
    protocol = json.loads((ROOT/'learning/protocol.json').read_text())
    connection = Client(str(SOCKET), family='AF_UNIX')
    try:
        for count in ((1,) if preflight else (1,10)):
            binary = Path(f'.work/agent/ten-fly/bin/worker-c{count}-optimized')
            brain = CPUBrain(flies=count) if preflight else Worker(binary=binary, flies=count)
            try:
                for seed in ((11,) if preflight else SEEDS):
                    output = ROOT/'learning'/('cpu-preflight' if preflight else f'seed-{seed}-c{count}')
                    if not preflight and (output/'metrics.json').exists():
                        saved = json.loads((output/'metrics.json').read_text())
                        assert saved['status'] == 'PASS' and saved['count'] == count and saved['seed'] == seed
                        print(json.dumps({'reuse_completed_seed':seed,'count':count}),flush=True)
                        continue
                    configs = protocol['training'][str(seed)][:1 if preflight else 10]
                    all_features, all_rewards, all_inputs, outcomes, completion = [], [], [], [], []
                    start = time.perf_counter()
                    for offset in range(0, len(configs), count):
                        features, rewards, inputs, physical = experience(connection, brain, configs[offset:offset+count])
                        elapsed = time.perf_counter()-start
                        all_features.extend(features); all_rewards.extend(rewards)
                        all_inputs.extend(np.swapaxes(inputs,0,1)); outcomes.append(physical)
                        completion.extend([elapsed]*len(features))
                    wall = time.perf_counter()-start
                    output.mkdir(exist_ok=False)
                    np.savez(output/'samples.npz', features=all_features, rewards=all_rewards, inputs=all_inputs,
                             completion_wall_seconds=completion)
                    metrics = {'status': 'PASS', 'count': count, 'seed': seed, 'scenes': configs,
                               'collection_wall_seconds': wall, 'completion_wall_seconds': completion,
                               'physical_body_exchanges': len(configs)*268,
                               'neural_exchanges': len(configs)*8,
                               'binary_sha256': None if preflight else digest(binary)}
                    (output/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
                    (output/'physical-outcomes.json').write_text(json.dumps(outcomes)+'\n')
                    print(json.dumps({'seed': seed, 'count': count, 'wall': wall, 'rewards': all_rewards},
                                     default=lambda x:x.tolist()),flush=True)
            finally:
                if not preflight:
                    brain.close()
    finally:
        connection.close()


def validate():
    protocol = json.loads((ROOT/'learning/protocol.json').read_text())
    assert protocol['seeds']==list(SEEDS)
    for count in (1,10):
        assert Path(f'.work/agent/ten-fly/bin/worker-c{count}-optimized').is_file()
    connection = Client(str(SOCKET),family='AF_UNIX')
    try:
        rows = request(connection,'configure',protocol['training']['11'])
        assert len(rows)==10 and np.isfinite([row['inputs'] for row in rows]).all()
    finally:
        connection.close()
    print(json.dumps({'status':'PASS','live_cpu_bodies':10,'resumable_complete_seeds':True}))


def fit(features, rewards, path):
    useful = np.abs(rewards[:,0]-rewards[:,1]) >= .01
    x = features[useful]
    targets = np.asarray([GAITS['left'] if row[0] >= row[1] else GAITS['right'] for row in rewards[useful]])
    model = Readout()
    model.mean = x.mean(0) if len(x) else np.zeros(2606)
    model.scale = x.std(0) if len(x) else np.ones(2606)
    model.scale[model.scale < 1e-6] = 1.
    model.weights = np.zeros((2607,5)); model.weights[-1,:2] = GAITS['straight']
    if len(x):
        z = np.column_stack(((x-model.mean)/model.scale, np.ones(len(x))))
        model.weights[:,:2] += z.T @ np.linalg.solve(z@z.T+np.eye(len(z)),targets-np.asarray(GAITS['straight']))
    model.save(path)
    return model, int(useful.sum())


def evaluate():
    protocol = json.loads((ROOT/'learning/protocol.json').read_text())
    connection = Client(str(SOCKET), family='AF_UNIX')
    brain = CPUBrain(flies=10)
    try:
        # Actual held-out physical counterfactual rollouts, cached identically for all policies.
        features, rewards, inputs, outcomes = experience(connection, brain, protocol['held_out'])
        request(connection, 'reset', [None]*10)
        for step in range(130):
            action = 'WALK' if step < 50 else 'FEED'
            bodies = request(connection, 'step', [(action,GAITS['straight'] if action=='WALK' else [0.,0.])]*10)
        straight_rewards = np.asarray([row['ingested'] for row in bodies])
        np.savez(ROOT/'learning/held-out.npz', features=features,rewards=rewards,inputs=inputs)
        (ROOT/'learning/held-out-physical.json').write_text(json.dumps(outcomes)+'\n')
    finally:
        connection.close()
    summaries = []; controls = []
    for seed in SEEDS:
        datasets = {count: np.load(ROOT/f'learning/seed-{seed}-c{count}/samples.npz') for count in (1,10)}
        assert np.array_equal(datasets[1]['rewards'],datasets[10]['rewards']), 'physical experience differs'
        error = float(np.max(np.abs(datasets[1]['features']-datasets[10]['features'])))
        assert error <= 1e-5, error
        budget = json.loads((ROOT/f'learning/seed-{seed}-c10/metrics.json').read_text())['collection_wall_seconds']
        for count, data in datasets.items():
            for criterion in ('equal_experience', 'equal_collection_wall'):
                mask = np.ones(len(data['features']), bool) if criterion == 'equal_experience' else data['completion_wall_seconds']<=budget
                path = ROOT/f'learning/seed-{seed}-c{count}-{criterion}.npz'
                model, useful = fit(data['features'][mask],data['rewards'][mask],path)
                if count == 10 and criterion == 'equal_experience':
                    temporary = ROOT/'live-readout.pending.npz'
                    model.save(temporary); temporary.replace(ROOT/'live-readout.npz')
                predictions = model.predict(features)
                choices = (predictions[:,1] <= predictions[:,0]).astype(int)
                intake = rewards[np.arange(len(rewards)),choices]
                summaries.append({'seed':seed,'count':count,'criterion':criterion,'scenes':int(mask.sum()),
                    'informative_pairs': useful, 'collection_wall_budget_seconds': budget,
                    'successes':int(np.count_nonzero(intake>=.05)), 'trials':len(intake),
                    'mean_ingested':float(intake.mean()), 'choices':choices.tolist(),
                    'model_sha256':digest(path), 'feature_batch_error':error})
        data = datasets[10]
        permutation = np.random.default_rng(seed).permutation(10)
        shuffled, _ = fit(data['features'],data['rewards'][permutation],ROOT/f'learning/seed-{seed}-shuffled.npz')
        predictions = shuffled.predict(features)
        choices = (predictions[:,1]<=predictions[:,0]).astype(int)
        intake = rewards[np.arange(10),choices]
        controls.append({'seed':seed,'control':'shuffled_reward','successes':int(np.count_nonzero(intake>=.05)),
                         'trials':10,'permutation':permutation.tolist(),
                         'changed_reward_labels':int(np.count_nonzero(np.argmax(data['rewards'],axis=1)!=
                                                        np.argmax(data['rewards'][permutation],axis=1)))})
    controls.append({'control':'untrained_straight','successes':int(np.count_nonzero(straight_rewards>=.05)),
                     'trials':10,'ingested':straight_rewards.tolist()})
    metrics = {'status':'PASS', 'results':summaries, 'held_out_scenes':protocol['held_out'],
               'controls':controls,
               'evaluation':'offline frozen policy selection over actual held-out physical left/right rollouts',
               'claim':'matched-budget contextual-bandit comparison, not repeated-decision competence'}
    (ROOT/'learning/comparison.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps(metrics),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('freeze','preflight','validate','collect','evaluate'))
    args=parser.parse_args()
    if args.stage=='freeze': freeze()
    elif args.stage=='validate': validate()
    elif args.stage=='evaluate': evaluate()
    else: collect(args.stage=='preflight')
