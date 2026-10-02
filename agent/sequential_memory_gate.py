"""Learn rendered-obstacle steering, rehearse food examples, test both frozen."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mujoco
import numpy as np
from agent.neural_connectome import CPUBrain, Readout, digest
from agent.odor_memory_gate import ABSENT, GAITS, sensory, execute
from agent.neural_single import EthologyArena, Environment, Organism, encode, step_signed
from agent.vision import EyeVision
from agent.hazard_curriculum import layouts as curriculum_layouts
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


def layouts(split, seeds):
    return curriculum_layouts(split, seeds)


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


def _validate_layout_splits(calibration, training, heldout):
    split_rows = {'calibration': calibration, 'training': training, 'heldout': heldout}
    ids = {name: {row['id'] for row in rows} for name, rows in split_rows.items()}
    seeds = {name: {row['seed'] for row in rows} for name, rows in split_rows.items()}
    for left, right in (('calibration', 'training'), ('calibration', 'heldout'), ('training', 'heldout')):
        if ids[left] & ids[right] or seeds[left] & seeds[right]:
            raise ValueError(f'curriculum {left}/{right} layouts or seeds overlap')


def calibrate(output, seeds=8):
    seed_values = list(range(1000, 1000 + seeds))
    scenes = layouts('calibration', seed_values)
    if not scenes:
        raise ValueError('calibration curriculum returned no layouts')
    arms = {'straight': {'success_count': 0, 'gait': list(GAITS['straight'])},
            'shuffled': {'success_count': 0, 'gait': None}}
    correct_turn = {'success_count': 0, 'left_target_count': 0, 'right_target_count': 0,
                    'left_success_count': 0, 'right_success_count': 0}
    upright = []
    for scene in scenes:
        arena = HazardArena(scene)
        try:
            initial = arena.get_state()
            for arm, config in arms.items():
                if arm == 'straight':
                    gait = GAITS['straight']
                else:
                    # Deterministic shuffled-label control: assign either turn
                    # direction from a seed-local permutation, with no scene input.
                    choice = int(np.random.default_rng(scene['seed'] * 100 + (1 if scene['side'] == 1 else 0)).integers(0, 2))
                    gait = GAITS[('left', 'right')[choice]]
                    config['gait'] = list(gait)
                result = avoid(arena, gait)
                config['success_count'] += int(result['contact_exchanges'] <= 1)
                upright.append(result['minimum_upright'])
                print('calibrate', arm, scene['id'], 'gait', list(gait),
                      'contact_exchanges', result['contact_exchanges'], flush=True)
                arena.set_state(initial)
            action = scene['correct_action']
            direction = 'left' if action == 'turn_left' else 'right'
            result = avoid(arena, GAITS[direction])
            passed = result['contact_exchanges'] <= 1
            correct_turn['success_count'] += int(passed)
            target_key = f'{direction}_target_count'
            success_key = f'{direction}_success_count'
            correct_turn[target_key] += 1
            correct_turn[success_key] += int(passed)
            upright.append(result['minimum_upright'])
            print('calibrate correct_turn_diagnostic', scene['id'], 'action', action,
                  'gait', list(GAITS[direction]), 'contact_exchanges', result['contact_exchanges'], flush=True)
            arena.set_state(initial)
        finally:
            arena.close()
    total = len(scenes)
    upright_minimum = min(upright)
    if upright_minimum < .85:
        print(f'FAIL calibrate: upright threshold broken, minimum={upright_minimum:.6f}, need 0.85', flush=True)
        raise SystemExit(1)
    arm_results = []
    for arm, config in arms.items():
        rate = config['success_count'] / total
        arm_results.append({'arm': arm, 'success_count': config['success_count'], 'total': total,
            'success_rate': rate, 'headroom_percentage_points': (1. - rate) * 100.,
            'gait_parameters_observed': config['gait']})
    report = {'schema': 'fly-world.sequential-memory-calibration.v1',
        'arm_results': arm_results,
        'arm_definitions': {'straight': {'gait': list(GAITS['straight'])},
            'shuffled': {'assignment': 'seed-local deterministic uniform left/right choice',
                'gaits': {'left': list(GAITS['left']), 'right': list(GAITS['right'])}}},
        'correct_turn_diagnostic': {'success_count': correct_turn['success_count'],
            'total': total, 'success_rate': correct_turn['success_count'] / total,
            'left_target_count': correct_turn['left_target_count'],
            'right_target_count': correct_turn['right_target_count'],
            'left_success_count': correct_turn['left_success_count'],
            'right_success_count': correct_turn['right_success_count']},
        'upright_minimum': upright_minimum, 'success_definition': 'at most one exchange-sampled positive-force hazard contact',
        'seed_values': seed_values, 'layout_ids': [scene['id'] for scene in scenes]}
    output.mkdir(parents=True, exist_ok=True)
    (output/'calibration.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'calibration': report['arm_results']}, indent=2), flush=True)


def freeze(output, food):
    calibration_path = output/'calibration.json'
    if not calibration_path.is_file():
        print('FAIL freeze: no calibration.json, run calibrate first', flush=True)
        raise SystemExit(1)
    calibration = json.loads(calibration_path.read_text())
    if not isinstance(calibration.get('upright_minimum'), (int, float)) or calibration['upright_minimum'] < .85:
        print(f"FAIL freeze: calibration upright threshold broken, minimum={calibration.get('upright_minimum')}, need 0.85", flush=True)
        raise SystemExit(1)
    by_arm = {row['arm']: row for row in calibration.get('arm_results', [])}
    if not calibration.get('seed_values'):
        print('FAIL freeze: calibration missing seed values', flush=True)
        raise SystemExit(1)
    training = layouts('training', range(2000, 2004))
    evaluation = layouts('heldout', range(5000, 5008))
    calibration_layouts = layouts('calibration', calibration.get('seed_values', []))
    _validate_layout_splits(calibration_layouts, training, evaluation)
    criteria = {'avoid_success_fraction': .75, 'maximum_contact_exchanges': 1,
                'minimum_success_improvement': .25, 'minimum_upright': .85,
                'maximum_food_success_regression': .125}
    required = criteria['minimum_success_improvement'] * 100.
    ceiling_result = calibration.get('correct_turn_diagnostic', {})
    if (not isinstance(ceiling_result.get('success_rate'), (int, float)) or
            not isinstance(ceiling_result.get('success_count'), int) or
            not isinstance(ceiling_result.get('total'), int) or
            ceiling_result['total'] <= 0 or
            not 0 <= ceiling_result['success_count'] <= ceiling_result['total']):
        print('FAIL freeze: calibration missing valid correct-turn ceiling', flush=True)
        raise SystemExit(1)
    ceiling = ceiling_result['success_count'] / ceiling_result['total']
    if (not 0. <= float(ceiling_result['success_rate']) <= 1. or
            abs(ceiling - float(ceiling_result['success_rate'])) > 1e-9):
        print('FAIL freeze: calibration correct-turn ceiling count/rate mismatch', flush=True)
        raise SystemExit(1)
    for arm in ('straight', 'shuffled'):
        entry = by_arm.get(arm)
        if entry is None or not isinstance(entry.get('success_rate'), (int, float)):
            print(f'FAIL freeze: calibration missing valid {arm} arm data', flush=True)
            raise SystemExit(1)
        if not 0. <= float(entry['success_rate']) <= 1.:
            print(f'FAIL freeze: calibration has invalid {arm} success rate', flush=True)
            raise SystemExit(1)
        if not isinstance(entry.get('success_count'), int) or not isinstance(entry.get('total'), int) or entry['total'] <= 0 or not 0 <= entry['success_count'] <= entry['total']:
            print(f'FAIL freeze: calibration has invalid {arm} counts', flush=True)
            raise SystemExit(1)
        rate = entry['success_count'] / entry['total']
        if abs(rate-float(entry['success_rate'])) > 1e-9:
            print(f'FAIL freeze: calibration {arm} count/rate mismatch', flush=True)
            raise SystemExit(1)
        headroom_fraction = ceiling - rate
        headroom = headroom_fraction * 100.
        if headroom_fraction + 1e-9 < criteria['minimum_success_improvement']:
            rate_percent = float(entry['success_rate']) * 100.
            print(f'FAIL freeze: criterion unreachable, {arm}={rate_percent:g}%, ceiling={ceiling*100.:g}% leaves {headroom:g}pp, need {required:g}pp', flush=True)
            raise SystemExit(1)
    expected_straight = list(GAITS['straight'])
    observed_straight = by_arm['straight'].get('gait_parameters_observed')
    definitions = calibration.get('arm_definitions', {})
    observed_turns = definitions.get('shuffled', {}).get('gaits', {})
    expected_turns = {'left': list(GAITS['left']), 'right': list(GAITS['right'])}
    if observed_straight != expected_straight or observed_turns != expected_turns:
        print(f'FAIL freeze: calibration arm parameters mismatch, straight={observed_straight}, expected={expected_straight}; shuffled={observed_turns}, expected={expected_turns}', flush=True)
        raise SystemExit(1)
    assert json.loads((food/'metrics.json').read_text())['status'] == 'PASS'
    output.mkdir(parents=True, exist_ok=True)
    if (output/'protocol.json').exists():
        raise FileExistsError(output/'protocol.json')
    protocol = {'schema': 'fly-world.sequential-memory-pilot.v1',
        'mode': 'engineering_agent', 'food_gate': str(food.resolve()),
        'food_model_sha256': digest(food/'trained.npz'),
        'food_metrics_sha256': digest(food/'metrics.json'),
        'training': training, 'evaluation': evaluation,
        'layout_seed_partition': {'calibration': '1000-1999', 'training': '2000-4999', 'heldout': '5000-7999'},
        'calibration_feasibility': {'rule': 'for each learned-beats-arm criterion, require correct-turn diagnostic ceiling minus comparator success rate >= required percentage-point improvement',
            'ceiling_success_rate': ceiling,
            'ceiling_success_percentage': ceiling*100.,
            'checked_against': {arm: {'success_rate': by_arm[arm]['success_rate'],
                'success_percentage': by_arm[arm]['success_rate']*100.,
                'headroom_percentage_points': (ceiling-by_arm[arm]['success_rate'])*100.,
                'required_percentage_points': required} for arm in ('straight', 'shuffled')}},
        'calibration_arm_parameters': {'straight': expected_straight,
            'shuffled_gaits': expected_turns,
            'shuffled_assignment': definitions['shuffled']['assignment']},
        'senses': 'eight rendered-eye exchanges through existing coarse visual encoder; no coordinate/task-id policy inputs',
        'reward': 'negative number of exchange-sampled positive-force hazard contacts during 0.5-second turn',
        'learning': 'ridge motor readout fit on rewarded avoidance examples plus eight previous food training examples',
        'rehearsal_budget_bytes': 1024*1024,
        'controls': ['trained', 'untrained_straight', 'shuffled_reward'],
        'criteria': criteria,
        'source_sha256': {str(path): digest(path) for path in [Path(__file__),
            Path(__file__).with_name('odor_memory_gate.py'), Path(__file__).with_name('vision.py'),
            Path(__file__).with_name('hazard_curriculum.py')]}}
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
    parser.add_argument('stage', choices=('calibrate', 'freeze', 'train', 'evaluate'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--food-gate', type=Path)
    parser.add_argument('--seeds', type=int, default=8)
    args = parser.parse_args()
    if args.stage == 'calibrate':
        calibrate(args.output, args.seeds)
    elif args.stage == 'freeze':
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
