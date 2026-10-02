"""Learn rendered-obstacle steering, rehearse food examples, test both frozen."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import mujoco
import numpy as np
from agent.neural_connectome import CPUBrain, Readout, digest
from agent.odor_memory_gate import ABSENT, GAITS, sensory, execute
from agent.neural_single import EthologyArena, Environment, Organism, encode, step_signed
from agent.vision import EyeVision
from fly_arena.behavior import BehaviorDecision


class HazardArena(EthologyArena):
    def __init__(self, scene):
        self.hazard = scene['hazard']
        super().__init__(seed=scene['seed'], eye_size=48)

    def _configure_world(self):
        super()._configure_world()
        geom = self.world.mjcf_root.worldbody.add_geom(name='memory_hazard',
            type=mujoco.mjtGeom.mjGEOM_BOX, pos=self.hazard['position'],
            size=self.hazard['size'], rgba=[.03, .03, .03, 1], contype=0, conaffinity=0)
        self.world.ground_geoms.append(geom)


def layouts(seeds):
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        jitter = rng.uniform(-.04, .04, 2)
        for side in (-1, 1):
            x, y = 3.1, side*1.65
            rows.append({'id': f'{seed}-{side}', 'seed': seed, 'side': side,
                'hazard': {'position': [float(x+jitter[0]), float(y+jitter[1]), 1.1],
                           'size': [.35, .35, 1.1]}})
    return rows


def visual_sensory(arena, brain):
    brain.reset(); vision = EyeVision(); inputs_trace = []; images = []
    environment, organism = Environment([]), Organism({})
    for _ in range(8):
        visual, threat, info = vision.step(arena.eyes())
        frame = environment.sense(arena, organism)
        frame.looming_left, frame.looming_right = map(float, threat)
        inputs = encode(frame, ABSENT); inputs[6:8] = visual
        packed = np.zeros((2, 8), np.float32); packed[0] = inputs
        neural = brain.step(packed)
        assert neural['counts'][1] == [0., 0.]
        events = step_signed(arena, BehaviorDecision('IDLE'), [0., 0.])
        environment.apply_physical_events(events, organism, .01)
        organism.advance(.01, 'IDLE')
        inputs_trace.append(inputs.tolist()); images.append(info['image_sha256'])
    features = np.asarray(neural['features'][0], np.float64)
    assert np.isfinite(features).all()
    return features, inputs_trace, images


def avoid(arena, gait):
    gid = arena.sim.mj_model.geom('memory_hazard').id
    trajectory = []
    for step in range(50):
        step_signed(arena, BehaviorDecision('WALK'), gait)
        force = 0.
        for index, contact in enumerate(arena.sim.mj_data.contact[:arena.sim.mj_data.ncon]):
            if gid in contact.geom:
                value = np.zeros(6)
                mujoco.mj_contactForce(arena.sim.mj_model, arena.sim.mj_data, index, value)
                force += max(0., float(value[0]))
        trajectory.append({'step': step, 'position': arena.position.tolist(),
                           'upright': float(arena.upright), 'hazard_normal_force': force})
    count = sum(row['hazard_normal_force'] > .01 for row in trajectory)
    return {'contact_exchanges': count, 'reward': -float(count),
            'minimum_upright': min(row['upright'] for row in trajectory), 'trajectory': trajectory}


def choose(model, features):
    if model is None or np.max(np.abs(features)) < 1e-8:
        return GAITS['straight']
    output = model.predict(features[None])[0]
    return GAITS['left'] if output[1] > output[0] else GAITS['right']


def freeze(output, food):
    assert json.loads((food/'metrics.json').read_text())['status'] == 'PASS'
    output.mkdir(parents=True, exist_ok=False)
    protocol = {'schema': 'fly-world.sequential-memory-pilot.v1',
        'mode': 'engineering_agent', 'food_gate': str(food.resolve()),
        'food_model_sha256': digest(food/'trained.npz'),
        'food_metrics_sha256': digest(food/'metrics.json'),
        'training': layouts(range(200, 204)), 'evaluation': layouts(range(300, 308)),
        'senses': 'eight rendered-eye exchanges through existing coarse visual encoder; no coordinate/task-id policy inputs',
        'reward': 'negative number of exchange-sampled positive-force hazard contacts during 0.5-second turn',
        'learning': 'ridge motor readout fit on rewarded avoidance examples plus eight previous food training examples',
        'rehearsal_budget_bytes': 1024*1024,
        'controls': ['trained', 'untrained_straight', 'shuffled_reward'],
        'criteria': {'avoid_success_fraction': .75, 'maximum_contact_exchanges': 1,
                     'minimum_success_improvement': .25, 'minimum_upright': .85,
                     'maximum_food_success_regression': .125},
        'source_sha256': {str(path): digest(path) for path in [Path(__file__),
            Path(__file__).with_name('odor_memory_gate.py'), Path(__file__).with_name('vision.py')]}}
    (output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')


def train(output, protocol):
    food = Path(protocol['food_gate']); brain = CPUBrain()
    old_features = np.load(food/'restart-fixture.npz')['features']
    old_protocol = json.loads((food/'protocol.json').read_text())
    old_targets = np.array([GAITS[json.loads((food/f'train-{scene["id"]}.json').read_text())['chosen_from_reward']]
                            for scene in old_protocol['training']])
    features, targets = [], []
    for scene in protocol['training']:
        arena = HazardArena(scene)
        try:
            f, inputs, images = visual_sensory(arena, brain)
            initial = arena.get_state(); outcomes = {}
            for name in ('left', 'right'):
                arena.set_state(initial); outcomes[name] = avoid(arena, GAITS[name])
            rewards = [outcomes[name]['reward'] for name in ('left', 'right')]
            tied = rewards[0] == rewards[1]
            chosen = ('left', 'right')[int(np.argmax(rewards))]
            if not tied:
                features.append(f); targets.append(GAITS[chosen])
            (output/f'train-{scene["id"]}.json').write_text(json.dumps({'scene': scene,
                'inputs': inputs, 'eye_image_sha256': images, 'outcomes': outcomes,
                'reward_tie': tied, 'chosen_from_reward': chosen})+'\n')
            print('train avoidance', scene['id'], rewards, 'tie', tied, flush=True)
        finally:
            arena.close()
    if len(features) < 4:
        (output/'training.json').write_text(json.dumps({'status': 'FAIL',
            'reason': 'insufficient informative physical-contact reward pairs', 'informative_pairs': len(features)})+'\n')
        raise RuntimeError('insufficient informative physical-contact reward pairs')
    x, target = np.stack(features), np.asarray(targets)
    if len(np.unique(target, axis=0)) < 2:
        (output/'training.json').write_text(json.dumps({'status': 'FAIL',
            'reason': 'physical rewards select only one turn; shuffled control would be identical'})+'\n')
        raise RuntimeError('reward preferences must contain both turn directions')
    combined = np.vstack((old_features, x))
    mean, scale = combined.mean(0), combined.std(0); scale[scale < 1e-6] = 1.
    z = np.column_stack(((combined-mean)/scale, np.ones(len(combined))))
    permutation = np.random.default_rng(42).permutation(len(target))
    report = {'status': 'PASS', 'training_pid': os.getpid(), 'informative_pairs': len(target),
        'rehearsed_food_examples': len(old_features), 'rehearsal_bytes': old_features.nbytes+old_targets.nbytes,
        'shuffle_permutation': permutation.tolist(), 'models': {}}
    assert report['rehearsal_bytes'] <= protocol['rehearsal_budget_bytes']
    for name, labels in [('trained', target), ('shuffled_reward', target[permutation])]:
        y = np.vstack((old_targets, labels))-np.asarray(GAITS['straight'])
        model = Readout(); model.mean, model.scale = mean.copy(), scale.copy()
        model.weights = np.zeros((2607, 5)); model.weights[-1, :2] = GAITS['straight']
        model.weights[:, :2] += z.T @ np.linalg.solve(z@z.T+np.eye(len(z)), y)
        model.save(output/f'{name}.npz')
        report['models'][name] = {'sha256': digest(output/f'{name}.npz'), 'predictions': model.predict(combined).tolist()}
    np.savez(output/'restart-fixture.npz', features=combined)
    (output/'training.json').write_text(json.dumps(report, indent=2)+'\n')


def evaluate(output, protocol):
    food = Path(protocol['food_gate']); training = json.loads((output/'training.json').read_text())
    assert training['training_pid'] != os.getpid()
    models = {name: Readout(output/f'{name}.npz') for name in ('trained', 'shuffled_reward')}
    x = np.load(output/'restart-fixture.npz')['features']
    restart = {name: float(np.max(np.abs(model.predict(x)-training['models'][name]['predictions']))) for name, model in models.items()}
    brain = CPUBrain(); results = []
    for scene in protocol['evaluation']:
        arena = HazardArena(scene)
        try:
            initial = arena.get_state()
            for control in protocol['controls']:
                arena.set_state(initial)
                f, inputs, images = visual_sensory(arena, brain)
                gait = choose(models.get(control), f)
                result = avoid(arena, gait)
                (output/f'eval-{scene["id"]}-{control}.json').write_text(json.dumps({
                    'scene': scene, 'control': control, 'inputs': inputs,
                    'eye_image_sha256': images, 'gait': gait, **result})+'\n')
                results.append({'scene_id': scene['id'], 'control': control,
                    **{key: result[key] for key in ('contact_exchanges', 'minimum_upright')}})
                print('eval avoidance', scene['id'], control, result['contact_exchanges'], flush=True)
        finally:
            arena.close()
    metrics = {}
    for control in protocol['controls']:
        rows = [r for r in results if r['control'] == control]
        metrics[control] = {'success_fraction': float(np.mean([r['contact_exchanges'] <= protocol['criteria']['maximum_contact_exchanges'] for r in rows])),
            'mean_contact_exchanges': float(np.mean([r['contact_exchanges'] for r in rows])),
            'minimum_upright': min(r['minimum_upright'] for r in rows)}
    food_protocol = json.loads((food/'protocol.json').read_text()); retention = []
    for scene in food_protocol['evaluation']:
        arena = EthologyArena(seed=scene['seed'], food_patches=[scene['food']])
        try:
            environment, organism = Environment([scene['food']], seed=scene['seed']), Organism({})
            f, inputs = sensory(arena, environment, organism, brain)
            gait = choose(models['trained'], f)
            result = execute(arena, environment, organism, gait)
            (output/f'food-retention-{scene["id"]}.json').write_text(json.dumps({'scene': scene, 'inputs': inputs, 'gait': gait, **result})+'\n')
            retention.append({'scene_id': scene['id'], 'ingested': result['ingested'], 'minimum_upright': result['minimum_upright']})
            print('food retention', scene['id'], result['ingested'], flush=True)
        finally:
            arena.close()
    food_success = float(np.mean([r['ingested'] >= food_protocol['criteria']['success_ingested_food_u'] for r in retention]))
    previous = json.loads((food/'metrics.json').read_text())['metrics']['trained']['success_fraction']
    trained = metrics['trained']['success_fraction']; c = protocol['criteria']
    checks = {'avoidance_success': trained >= c['avoid_success_fraction'],
        'beats_each_control': all(trained-metrics[name]['success_fraction'] >= c['minimum_success_improvement'] for name in protocol['controls'][1:]),
        'food_retention': food_success >= previous-c['maximum_food_success_regression'],
        'body_upright': min([r['minimum_upright'] for r in retention]+[r['minimum_upright'] for r in results]) >= c['minimum_upright'],
        'restart_exact': max(restart.values()) <= 1e-12,
        'evaluation_models_unchanged': all(digest(output/f'{name}.npz') == training['models'][name]['sha256'] for name in models)}
    report = {'status': 'PASS' if all(checks.values()) else 'FAIL', 'checks': checks,
        'avoidance': metrics, 'food_success_before': previous, 'food_success_after': food_success,
        'food_retention': retention, 'avoidance_trials': results, 'restart_max_abs_error': restart,
        'evaluation_pid': os.getpid(), 'learning_during_evaluation': False,
        'scope': 'two bounded turn-selection tasks with eight-example rehearsal; not unassisted lifelong memory'}
    (output/'metrics.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('freeze', 'train', 'evaluate'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--food-gate', type=Path)
    args = parser.parse_args()
    if args.stage == 'freeze':
        freeze(args.output, args.food_gate)
    else:
        protocol = json.loads((args.output/'protocol.json').read_text())
        assert all(digest(Path(path)) == sha for path, sha in protocol['source_sha256'].items())
        assert digest(Path(protocol['food_gate'])/'trained.npz') == protocol['food_model_sha256']
        started = time.perf_counter()
        if args.stage == 'train':
            train(args.output, protocol)
        else:
            report = evaluate(args.output, protocol)
            print(json.dumps({'status': report['status'], 'wall_seconds': time.perf_counter()-started}), flush=True)
            if report['status'] != 'PASS':
                raise SystemExit(1)
