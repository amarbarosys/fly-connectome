"""CPU sensor-to-body replay fixture using existing arena state contracts.

Vision, viewer controls and direct stimulation are intentionally outside this fixture.
"""
from __future__ import annotations

import argparse
from collections import deque
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import mujoco

from agent.neural_connectome import CPUBrain, Readout, ROOT, digest
from agent.neural_checkpoint import STATE_FIELDS, identity, save_checkpoint as save_neural, restore_checkpoint
from agent.neural_single import EthologyArena, MotorDecision, encode, step_signed, wall_observation
from agent.desire import Desire
from fly_arena.checkpoint import save_checkpoint, load_checkpoint, code_digest
from fly_arena.environment import Environment
from fly_arena.organism import Organism
from fly_duel.duel import encounter_scene


class EpisodeFixture:
    """Bounded no-vision diagnostic, not a replacement for the interactive runner."""
    def __init__(self, seed=1):
        config, _, _ = encounter_scene(resources='scarce')
        self.environment = Environment(config['food'], config.get('environment', {}), seed=seed)
        self.organism = Organism(config.get('organism', {}), initial=config.get('initial', {}))
        patches = [{'id': p.id, 'x': p.x, 'y': p.y, 'radius': p.radius, 'surface_z': p.surface_z}
                   for p in self.environment.food]
        self.arena = EthologyArena(seed=seed, food_patches=patches)
        self.brain = CPUBrain()
        self.readout = Readout()
        self.desire = Desire()
        self.decision = MotorDecision()
        self.steps = 0
        self.absent = SimpleNamespace(body_contact=False, detected=False, bearing=0., distance_mm=None)
        self.compatibility = {'schema': 'fly-world.no-vision-episode-fixture.v1',
            'neural': identity(self.brain), 'arena_code_sha256': code_digest(),
            'model_sha256': hashlib.sha256(self.arena.sim.mj_model.__getstate__()).hexdigest(),
            'mujoco': mujoco.__version__, 'numpy': np.__version__,
            'sources': {p.name: digest(p) for p in [Path(__file__), ROOT/'agent/desire.py',
                ROOT/'agent/neural_single.py', ROOT/'agent/neural_duel.py']}}

    def step(self):
        frame = self.environment.sense(self.arena, self.organism)
        inputs = encode(frame, self.absent)
        contact, turn = wall_observation(self.arena)
        driven, motivation = self.desire.evaluate(self.organism.hunger, inputs,
            self.arena.position, frame.mouth_contact, frame.upright,
            wall_contact=contact, wall_turn=turn)
        packed = np.zeros((2, 8), np.float32)
        packed[0] = driven
        teacher = self.steps % 7 == 0
        # Declared synthetic sensory pulse, confined to this persistence fixture.
        if teacher:
            packed[0] = np.maximum(packed[0], .8)
        neural = self.brain.step(packed)
        features = np.asarray(neural['features'][0])
        self.decision.signal = bool(np.max(np.abs(features)) > 1e-8)
        self.decision.output = self.desire.modulate(self.readout.predict(features[None])[0], self.decision.signal)
        chosen = self.decision.decide(frame, self.organism, None)
        if chosen.action == 'FENCE':
            chosen = replace(chosen, action='WALK')
        gait = self.decision.output[:2] if chosen.action == 'WALK' else np.zeros(2)
        physical = step_signed(self.arena, chosen, gait)
        self.environment.apply_physical_events(physical, self.organism, .01)
        self.organism.advance(.01, chosen.action, movement_proxy=frame.movement)
        self.environment.advance_field(.01)
        # A declared artificial teacher pulse exercises readout persistence.
        feedback = -.5 if self.readout.predict(features[None])[0, 2] >= .5 else .5
        update = self.readout.reinforce(features, 'FEED', feedback) if teacher else 0.
        self.steps += 1
        return {'step': self.steps, 'action': chosen.action, 'inputs': packed[0].tolist(),
                'observed_inputs': driven.tolist(),
                'output': self.decision.output.tolist(), 'position': self.arena.position.tolist(),
                'teacher_sensory_pulse': teacher, 'teacher_action': 'FEED' if teacher else None,
                'teacher_feedback': feedback if teacher else None,
                'upright': self.arena.upright, 'counts': neural['counts'],
                'motivation': motivation, 'readout_update_norm': update,
                'ingested': self.organism.state.ingested_total,
                'energy': self.organism.state.energy}

    def state(self):
        desire = copy.deepcopy(vars(self.desire))
        desire['progress'] = list(desire['progress'])
        return {'arena': self.arena.get_state(), 'environment': self.environment.get_state(),
                'organism': self.organism.get_state(), 'desire': desire,
                'decision': copy.deepcopy(vars(self.decision)), 'steps': self.steps,
                'readout': {name: getattr(self.readout, name).copy() for name in ('mean', 'scale', 'weights')}}

    def restore(self, saved):
        self.arena.set_state(saved['arena'])
        self.environment.set_state(saved['environment'])
        self.organism.set_state(saved['organism'])
        self.desire.__dict__.update(copy.deepcopy(saved['desire']))
        self.desire.progress = deque(self.desire.progress)
        self.decision.__dict__.update(copy.deepcopy(saved['decision']))
        for name, value in saved['readout'].items():
            setattr(self.readout, name, value.copy())
        self.steps = saved['steps']

    def close(self):
        self.arena.close()


def tree_equal(a, b):
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(tree_equal(v, b[k]) for k, v in a.items())
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(tree_equal(x, y) for x, y in zip(a, b))
    return a == b


def verify(out):
    out.mkdir(parents=True, exist_ok=False)
    episode = None
    started = time.perf_counter()
    try:
        episode = EpisodeFixture()
        prefix = [episode.step() for _ in range(20)]
        assert any(row['readout_update_norm'] > 0 for row in prefix), 'learning persistence fixture was not exercised'
        save_neural(episode.brain, out/'neural.npz')
        save_checkpoint(out/'episode.npz', episode.state(), episode.compatibility)
        first = [episode.step() for _ in range(30)]
        final = episode.state()
        final_neural = {name: getattr(episode.brain, name).copy() for name in STATE_FIELDS}
        wrong = dict(episode.compatibility, schema='wrong')
        try:
            load_checkpoint(out/'episode.npz', wrong)
        except ValueError:
            pass
        else:
            raise AssertionError('incompatible body checkpoint accepted')
        assert tree_equal(final, episode.state()), 'compatibility rejection changed episode'
        saved, _ = load_checkpoint(out/'episode.npz', episode.compatibility)
        restore_checkpoint(episode.brain, out/'neural.npz')
        episode.restore(saved)
        second = [episode.step() for _ in range(30)]
        assert first == second, 'body/action/resource replay differs'
        assert tree_equal(final, episode.state()), 'final body/controller/learning/world state differs'
        assert all(np.array_equal(getattr(episode.brain, k), v) for k, v in final_neural.items())
        assert np.isfinite(episode.arena.sim.mj_data.qpos).all()
        (out/'events.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in first))
        manifest = {'compatibility': episode.compatibility, 'software_rendering_requested': True,
                    'neural_checkpoint_sha256': digest(out/'neural.npz'),
                    'episode_checkpoint_sha256': digest(out/'episode.npz'),
                    'config': {'seed': 1, 'prefix_exchanges': 20, 'suffix_exchanges': 30,
                               'dt_s': .01, 'teacher_feedback': 'FEED +/- .5 opposing predicted feed; all eight sensory ports >=.8 every seventh step',
                               'vision': False, 'direct_stimulation': False}}
        (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        result = {'status': 'PASS', 'exact_event_and_final_state_replay': True,
                  'replayed_simulated_seconds': .3, 'prefix_simulated_seconds': .2,
                  'replayed_exchanges': 30, 'finite_physics': True,
                  'learning_updates_in_prefix': sum(r['readout_update_norm']>0 for r in prefix),
                  'incompatible_archive_rejected_before_restore': True,
                  'wall_seconds': time.perf_counter()-started,
                  'scope': 'same-instance CPU no-vision fixture; physics, controller RNG/phases, neural, readout, organism, world, engineered drives',
                  'unverified': ['cross-instance restore', 'interactive viewer state', 'vision history',
                                 'GPU worker', 'long-run balance', 'biological behavior']}
        (out/'metrics.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result, indent=2), flush=True)
    finally:
        if episode is not None:
            episode.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    verify(parser.parse_args().out)
