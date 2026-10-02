"""Reward-labelled odor steering pilot with fresh-process frozen evaluation.

One learned decision selects a turn; existing CPG and feed gestures execute it.
This is a bounded contextual-bandit pilot, not autonomous lifelong learning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np

from agent.neural_connectome import CPUBrain, Readout, ROOT, digest
from agent.neural_single import EthologyArena, Environment, Organism, encode, step_signed
from fly_arena.behavior import BehaviorDecision


ABSENT = SimpleNamespace(body_contact=False, detected=False, bearing=0., distance_mm=None)
GAITS = {'left': [.2, .65], 'right': [.65, .2], 'straight': [.425, .425]}


def scenes(seeds):
    result = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        distance, bearing, odor = float(rng.uniform(2.4, 2.6)), float(rng.uniform(.5, .65)), float(rng.uniform(.08, .12))
        for side in (-1, 1):
            result.append({'id': f'{seed}-{side}', 'seed': seed, 'side': side,
                'food': {'id': 'food', 'x': .55282 + distance * np.cos(bearing),
                         'y': side * distance * np.sin(bearing), 'radius': .85,
                         'surface_z': .01, 'odor': odor, 'odor_length_mm': 2.}})
    return result


def freeze(output, evaluation_seeds=8):
    """evaluation_seeds defaults to 8, the frozen original: 8 seeds x 2 sides = 16 layouts.
    Raising it widens the held-out set for statistical power; training stays fixed so the
    learning conditions are unchanged and arms remain comparable."""
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        'schema': 'fly-world.odor-memory-pilot.v1', 'mode': 'engineering_agent',
        'task': 'one odor-guided turn followed by existing physical feeding gesture',
        'training': scenes(range(4)), 'evaluation': scenes(range(100, 100+evaluation_seeds)),
        'neural': 'CPU fixed connectome; eight natural sensory exchanges; no vision or injected pulses',
        'physics_dt_s': .0001, 'exchange_dt_s': .01,
        'turn_steps': 50, 'feed_steps': 80,
        'learning': 'choose better of two actual ingestion-reward rollouts; ridge motor readout fit, regularization 1',
        'controls': ['trained', 'untrained_straight', 'shuffled_reward', 'odor_absent'],
        'criteria': {'trained_success_fraction': .75, 'success_ingested_food_u': .05,
            'minimum_success_improvement_over_each_control': .25,
            'minimum_upright': .85, 'restart_max_abs_prediction_error': 1e-12},
        'conditional_next': 'avoidance training and food retention only if this gate passes',
        'source_sha256': {str(path): digest(path) for path in (
            Path(__file__), ROOT/'agent/neural_connectome.py', ROOT/'agent/neural_single.py', ROOT/'agent/neural_duel.py')},
    }
    (output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    return protocol


def sensory(arena, environment, organism, brain, odor=True):
    brain.reset()
    rows = []
    for _ in range(8):
        frame = environment.sense(arena, organism)
        inputs = encode(frame, ABSENT)
        if not odor:
            inputs[:2] = 0
        packed = np.zeros((2, 8), np.float32)
        packed[0] = inputs
        neural = brain.step(packed)
        assert neural['counts'][1] == [0., 0.]
        physical = step_signed(arena, BehaviorDecision('IDLE'), [0., 0.])
        environment.apply_physical_events(physical, organism, .01)
        organism.advance(.01, 'IDLE')
        rows.append(inputs.tolist())
    features = np.asarray(neural['features'][0], np.float64)
    assert np.isfinite(features).all()
    return features, rows


def execute(arena, environment, organism, gait):
    rows = []
    initial_ingested = organism.state.ingested_total
    for i in range(130):
        action = 'WALK' if i < 50 else 'FEED'
        controls = gait if action == 'WALK' else [0., 0.]
        events = step_signed(arena, BehaviorDecision(action), controls)
        environment.apply_physical_events(events, organism, .01)
        organism.advance(.01, action)
        assert np.isfinite(arena.sim.mj_data.qvel).all()
        rows.append({'step': i, 'action': action, 'gait': list(controls),
            'position': arena.position.tolist(), 'upright': float(arena.upright),
            'ingested': float(organism.state.ingested_total),
            'mouth_contact_s': dict(events.contact_s_by_food)})
    return {'ingested': float(organism.state.ingested_total-initial_ingested),
            'minimum_upright': min(row['upright'] for row in rows), 'trajectory': rows}


def train(output, protocol):
    brain = CPUBrain()
    samples, features, targets = [], [], []
    for scene in protocol['training']:
        arena = EthologyArena(seed=scene['seed'], food_patches=[scene['food']])
        try:
            environment = Environment([scene['food']], seed=scene['seed'])
            organism = Organism({})
            f, inputs = sensory(arena, environment, organism, brain)
            initial = arena.get_state()
            environment_state, organism_state = environment.get_state(), organism.get_state()
            outcomes = {}
            for name in ('left', 'right'):
                arena.set_state(initial)
                environment.set_state(environment_state)
                organism.set_state(organism_state)
                outcomes[name] = execute(arena, environment, organism, GAITS[name])
            rewards = [outcomes[name]['ingested'] for name in ('left', 'right')]
            # Reward is physical ingestion, never the layout's side or bearing.
            chosen = ('left', 'right')[int(np.argmax(rewards))]
            tied = abs(rewards[0]-rewards[1]) < .01
            if not tied:
                features.append(f)
                targets.append(GAITS[chosen])
            sample = {'scene': scene, 'inputs': inputs, 'chosen_from_reward': chosen,
                      'reward_tie': tied, 'outcomes': outcomes}
            samples.append(sample)
            (output/f'train-{scene["id"]}.json').write_text(json.dumps(sample)+'\n')
            print('train', scene['id'], rewards, 'tie', tied, flush=True)
        finally:
            arena.close()
    if len(features) < 4:
        report = {'status': 'FAIL', 'reason': 'fewer than four informative ingestion-reward pairs',
                  'informative_pairs': len(features)}
        (output/'training.json').write_text(json.dumps(report, indent=2)+'\n')
        raise RuntimeError(report['reason'])
    x, target = np.stack(features), np.asarray(targets)
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1e-6] = 1.
    z = np.column_stack(((x-mean)/scale, np.ones(len(x))))
    baseline = np.zeros((2607, 5)); baseline[-1, :2] = GAITS['straight']
    permutation = np.random.default_rng(42).permutation(len(target))
    report = {'status': 'PASS', 'informative_pairs': len(features),
              'training_pid': __import__('os').getpid(), 'shuffle_permutation': permutation.tolist(),
              'changed_shuffle_labels': int(np.count_nonzero(np.any(target != target[permutation], axis=1))),
              'models': {}, 'learning': protocol['learning']}
    for name, y in [('trained', target), ('shuffled_reward', target[permutation])]:
        weights = baseline.copy()
        weights[:, :2] += z.T @ np.linalg.solve(z@z.T + np.eye(len(z)), y-np.asarray(GAITS['straight']))
        model = Readout()
        model.mean, model.scale, model.weights = mean.copy(), scale.copy(), weights
        path = output/f'{name}.npz'
        model.save(path)
        report['models'][name] = {'sha256': digest(path),
                                 'prediction': model.predict(x).tolist()}
    np.savez(output/'restart-fixture.npz', features=x)
    (output/'training.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def evaluate(output, protocol):
    import os
    training = json.loads((output/'training.json').read_text())
    assert training['status'] == 'PASS'
    assert os.getpid() != training['training_pid'], 'evaluation must be a fresh process'
    models = {name: Readout(output/f'{name}.npz') for name in ('trained', 'shuffled_reward')}
    x = np.load(output/'restart-fixture.npz')['features']
    restart_errors = {}
    for name, model in models.items():
        assert digest(output/f'{name}.npz') == training['models'][name]['sha256']
        restart_errors[name] = float(np.max(np.abs(model.predict(x)-training['models'][name]['prediction'])))
    brain = CPUBrain()
    summaries = []
    for scene in protocol['evaluation']:
        arena = EthologyArena(seed=scene['seed'], food_patches=[scene['food']])
        try:
            initial = arena.get_state()
            for control in protocol['controls']:
                arena.set_state(initial)
                environment = Environment([scene['food']], seed=scene['seed'])
                organism = Organism({})
                f, inputs = sensory(arena, environment, organism, brain, odor=control != 'odor_absent')
                model = models['trained'] if control == 'odor_absent' else models.get(control)
                gait = model.predict(f[None])[0, :2] if model else GAITS['straight']
                # The learned outputs select the two disclosed motor primitives.
                if model and np.max(np.abs(f)) > 1e-8:
                    gait = GAITS['left'] if gait[1] > gait[0] else GAITS['right']
                else:
                    gait = GAITS['straight']
                result = execute(arena, environment, organism, gait)
                row = {'scene': scene, 'control': control, 'inputs': inputs, 'gait': list(gait), **result}
                (output/f'eval-{scene["id"]}-{control}.json').write_text(json.dumps(row)+'\n')
                summaries.append({key: row[key] for key in ('control', 'ingested', 'minimum_upright')} | {'scene_id': scene['id']})
                print('eval', scene['id'], control, result['ingested'], flush=True)
        finally:
            arena.close()
    criteria = protocol['criteria']
    metrics = {}
    for control in protocol['controls']:
        rows = [row for row in summaries if row['control'] == control]
        metrics[control] = {'success_fraction': float(np.mean([r['ingested'] >= criteria['success_ingested_food_u'] for r in rows])),
                           'mean_ingested': float(np.mean([r['ingested'] for r in rows])),
                           'minimum_upright': min(r['minimum_upright'] for r in rows), 'trials': len(rows)}
    trained = metrics['trained']['success_fraction']
    checks = {'trained_success': trained >= criteria['trained_success_fraction'],
              'finite_upright_body': all(v['minimum_upright'] >= criteria['minimum_upright'] for v in metrics.values()),
              'restart_predictions': max(restart_errors.values()) <= criteria['restart_max_abs_prediction_error'],
              'evaluation_weights_unchanged': all(digest(output/f'{name}.npz') == training['models'][name]['sha256'] for name in models),
              'shuffled_labels_changed': training['changed_shuffle_labels'] >= len(x)//2}
    for control in protocol['controls'][1:]:
        checks['beats_'+control] = trained-metrics[control]['success_fraction'] >= criteria['minimum_success_improvement_over_each_control']
    report = {'status': 'PASS' if all(checks.values()) else 'FAIL', 'checks': checks,
        'metrics': metrics, 'restart_prediction_max_abs_error': restart_errors,
        'evaluation_pid': os.getpid(), 'learning_enabled_during_evaluation': False,
        'protocol_sha256': digest(output/'protocol.json'), 'trials': summaries,
        'scope': 'one-decision odor steering with existing CPG/feed gestures; not general memory or biological learning'}
    (output/'metrics.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('freeze', 'train', 'evaluate'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--evaluation-seeds', type=int, default=8,
                        help='held-out seeds; each yields 2 layouts. 8 reproduces the frozen original.')
    args = parser.parse_args()
    if args.stage == 'freeze':
        freeze(args.output, args.evaluation_seeds)
    else:
        protocol = json.loads((args.output/'protocol.json').read_text())
        assert all(digest(Path(path)) == sha for path, sha in protocol['source_sha256'].items()), 'source differs from freeze'
        started = time.perf_counter()
        report = train(args.output, protocol) if args.stage == 'train' else evaluate(args.output, protocol)
        print(json.dumps({'stage': args.stage, 'status': report['status'], 'wall_seconds': time.perf_counter()-started}), flush=True)
        if report['status'] != 'PASS':
            raise SystemExit(1)
