"""Two physical bodies controlled by independent full-graph neural states.

The connectome is fixed. A trained readout imitates a disclosed sensory teacher;
this prototype does not claim biological aggression or trained flight.
"""
from __future__ import annotations
import argparse
import copy
from dataclasses import replace
import json
import math
import os
from queue import SimpleQueue
from pathlib import Path
import signal
import sys
import time

import numpy as np
from agent.neural_connectome import ROOT, ASSETS, Worker, CPUBrain, Readout, digest

UPSTREAM = ROOT/'.work/fly-research/repos/artem-x-meta/fly-arena'
sys.path.insert(0, str(UPSTREAM))
from fly_duel.duel import Duel, encounter_scene
from fly_duel.runner import telemetry, _camera, _json_default
from fly_arena.behavior import BehaviorDecision


def encode(frame, social):
    odor = np.tanh(np.maximum([frame.odor_left, frame.odor_right], 0)*10)
    taste = max(frame.mouth_taste, max(frame.tarsal_taste.values(), default=0))
    touch = max(float(social.body_contact), max(frame.tactile_by_region.values(), default=0))
    proximity = 1. if social.body_contact else 0. if not social.detected else max(0., 1.-(social.distance_mm or 0.)/3.)
    side = math.sin(social.bearing)
    rival_left, rival_right = proximity*(.5+.5*side), proximity*(.5-.5*side)
    return np.clip([*odor, taste, touch, frame.looming_left, frame.looming_right,
                    rival_left, rival_right], 0, 1).astype(np.float32)


class MotorDecision:
    """Action selection from readout; physical safety and gesture timing only."""
    def __init__(self):
        self.output = np.zeros(5)
        self.signal = False
        self.current = 'IDLE'
        self.since = 0.

    def decide(self, frame, organism, readout, dt=.01, mode=None):
        stable = frame.upright >= .8 and frame.ground_support >= 3
        if not stable or not self.signal:
            self.current = 'IDLE'
            self.since = frame.time_s
        else:
            left, right, feed, fence, retreat = self.output
            action = ('WALK' if retreat > .55 else 'FENCE' if fence > .45
                      else 'FEED' if feed > .45 else 'WALK')
            # Let feeding/fencing motor programs complete a bounded gesture.
            minimum = .65 if self.current == 'FENCE' else .8 if self.current == 'FEED' else .15
            if frame.time_s-self.since >= minimum and action != self.current:
                self.current, self.since = action, frame.time_s
        return BehaviorDecision(self.current, reason='learned descending voltage/trace readout',
            source='experimental_connectome_readout', gates={'stable': stable},
            scores=dict(zip(('left','right','feed','fence','retreat'), self.output.tolist())))


class MotorNavigation:
    def __init__(self, decision):
        self.decision = decision

    def step(self, frame, readout, mode, dt, active=False):
        return self.decision.output[:2].copy() if active and self.decision.signal else np.zeros(2)


class NeuralDuel(Duel):
    def __init__(self, brain, readout, *, seed=1, control='intact', resources='scarce'):
        config, start, headings = encounter_scene(resources=resources)
        super().__init__(config, seed=seed, start=start, start_headings=headings, social=False)
        self.brain, self.control = brain, control
        self.readouts = [copy.deepcopy(readout), copy.deepcopy(readout)]
        self.stimulus = np.zeros((2,8),np.float32)
        self.reward_updates = [0,0]
        self.reward_update_norms = [0.,0.]
        self.last_neural = None
        self.last_inputs = None
        self.last_outputs = None
        self.initial_qpos = self.arena.sim.mj_data.qpos.copy()
        self.initial_ctrl = self.arena.sim.mj_data.ctrl.copy()
        self.body_resets = 0
        self.permutation = np.random.default_rng(73).permutation(2606)
        for c in self.contestants:
            c.behavior = MotorDecision()
            c.navigator = MotorNavigation(c.behavior)

    def reset_bodies(self):
        """Explicit operator intervention; retain decoder learning and food history."""
        import mujoco
        data = self.arena.sim.mj_data
        data.qpos[:] = self.initial_qpos
        data.qvel[:] = 0
        data.qacc_warmstart[:] = 0
        data.ctrl[:] = self.initial_ctrl
        data.act[:] = 0
        self.brain.reset()
        self.stimulus[:] = 0
        self.last_neural = None
        for i,c in enumerate(self.contestants):
            c.unit.controller.reset(seed=i+1)
            c.unit.motor = type(c.unit.motor)(c.unit)
            c.behavior.current = 'IDLE'
            c.behavior.since = self.elapsed_s
            c.behavior.signal = False
        mujoco.mj_forward(self.arena.sim.mj_model,data)
        self.body_resets += 1

    def step(self, dt=.01):
        social_frames = self.arena.social_frames()
        inputs = np.array([encode(self.environment.sense(c.unit, c.organism, None), social)
                           for c, social in zip(self.contestants, social_frames)])
        inputs = np.maximum(inputs,self.stimulus)
        if self.control == 'null': inputs[:] = 0
        result = self.brain.step(inputs)
        features = np.asarray(result['features'])
        if self.control == 'shuffled': features = features[:, self.permutation]
        output = np.array([model.predict(features[i:i+1])[0] for i,model in enumerate(self.readouts)])
        for i, c in enumerate(self.contestants):
            c.behavior.output = output[i]
            c.behavior.signal = bool(np.max(np.abs(features[i])) > 1e-8)
        self.last_inputs, self.last_outputs, self.last_neural = inputs, output, result
        before = [c.organism.state.ingested_total for c in self.contestants]
        super().step(dt)
        for i,c in enumerate(self.contestants):
            food_reward = c.organism.state.ingested_total-before[i]
            if self.control == 'intact' and food_reward>0:
                self.feedback(i,min(1.,food_reward*10))

    def feedback(self,index,reward):
        if self.last_neural is None or self.control != 'intact':return 0.
        norm = self.readouts[index].reinforce(self.last_neural['features'][index],
                                             self.contestants[index].decision.action,reward)
        self.reward_updates[index] += int(norm>0)
        self.reward_update_norms[index] += norm
        return norm


def run(*, seconds=0., viewer=True, control='intact', output=None, seed=1, backend='gpu'):
    import mujoco
    import mujoco.viewer
    out = Path(output or ROOT/'.work/neural-flies/live')
    out.mkdir(parents=True, exist_ok=True)
    brain = view = duel = None
    stopping, paused = [False], [False]
    focus, pulse_until = [0], [0.]
    teaching_events = []
    keys = SimpleQueue()
    signal.signal(signal.SIGINT, lambda *_: stopping.__setitem__(0, True))
    signal.signal(signal.SIGTERM, lambda *_: stopping.__setitem__(0, True))
    started = time.perf_counter()
    report = {'controller': 'learned_motor_readout', 'neural_backend': 'Mojo GPU' if backend=='gpu' else 'SciPy CPU oracle',
              'control': control, 'seed': seed, 'brain_neurons_per_fly': 139255,
              'brain_edges_shared': 10572269, 'independent_neural_states': 2,
              'readout_sha256': digest(ASSETS/'readout.npz'),
              'biology': 'unvalidated model interface', 'flight': 'not implemented'}
    failure = None
    try:
        gate = json.loads((ASSETS/'neural-gate.json').read_text())
        training = json.loads((ASSETS/'training.json').read_text())
        if gate['status'] != 'PASS' or training['rmse'] >= training['constant_rmse']:
            raise RuntimeError('neural/readout deployment gates did not pass')
        brain = Worker(disconnected=control == 'disconnected') if backend=='gpu' else CPUBrain(disconnected=control=='disconnected')
        readout = Readout()
        duel = NeuralDuel(brain, readout, seed=seed, control=control)
        def apply_key(keycode):
            event_count = len(teaching_events)
            if keycode == 32:paused[0] = not paused[0]
            elif keycode == 258:focus[0] = 1-focus[0]
            elif keycode == 78:
                duel.reset_bodies()
                teaching_events.append({'time_s':duel.elapsed_s,'operator_body_reset':duel.body_resets})
            elif keycode in (49,50,51,52,53):
                channels = {49:[0,1],50:[2],51:[3],52:[4,5],53:[6,7]}[keycode]
                duel.stimulus[:] = 0
                duel.stimulus[focus[0],channels] = .8
                pulse_until[0] = duel.elapsed_s+.3
                teaching_events.append({'time_s':duel.elapsed_s,'fly':focus[0],'stimulus_channels':channels})
            elif keycode in (82,80):
                reward = 1. if keycode==82 else -1.
                norm = duel.feedback(focus[0],reward)
                teaching_events.append({'time_s':duel.elapsed_s,'fly':focus[0],'feedback':reward,'decoder_weight_delta_norm':norm})
            if len(teaching_events)>event_count:
                with (out/'teaching-events.jsonl').open('a') as events:
                    events.write(json.dumps(teaching_events[-1])+'\n')
        if viewer:
            view = mujoco.viewer.launch_passive(duel.arena.sim.mj_model, duel.arena.sim.mj_data,
                key_callback=keys.put,
                show_left_ui=False, show_right_ui=False)
            with view.lock():
                centre, distance = _camera(duel)
                view.cam.lookat[:] = centre
                view.cam.distance, view.cam.azimuth, view.cam.elevation = distance, 90., -58.
        print('NEURAL_DUEL_READY: two independent full-graph states; Space pauses; close window stops.',flush=True)
        report['physics_timestep_s'] = duel.arena.sim.timestep
        report['started_wall_time'] = time.time()
        (out/'launch.json').write_text(json.dumps(report,indent=2,default=_json_default)+'\n')
        with (out/'telemetry.jsonl').open('w', buffering=1) as trace:
            last_report = time.perf_counter()
            while not stopping[0] and (seconds == 0 or duel.elapsed_s < seconds):
                while not keys.empty():apply_key(keys.get())
                if view and not view.is_running(): break
                if paused[0]:
                    view.sync(); time.sleep(.03); continue
                if duel.elapsed_s>=pulse_until[0]:duel.stimulus[:]=0
                step_start = time.perf_counter()
                duel.step()
                row = telemetry(duel)
                row.update(sensory_inputs=duel.last_inputs.tolist(), neural_outputs=duel.last_outputs.tolist(),
                           neural_counts=duel.last_neural['counts'], body_exchange_wall_s=time.perf_counter()-step_start,
                           control=control, reward_updates=duel.reward_updates.copy(), reward_update_norms=duel.reward_update_norms.copy(), neural_features_nonzero=[int(np.count_nonzero(f)) for f in duel.last_neural['features']])
                trace.write(json.dumps(row,allow_nan=False,default=_json_default)+'\n')
                if view:
                    with view.lock():
                        centre, distance = _camera(duel)
                        view.cam.lookat[:] = centre; view.cam.distance = distance
                    text = f'NEURAL / {duel.elapsed_s:.2f}s / {control} / Space: pause\n'
                    for c, counts in zip(duel.contestants,duel.last_neural['counts']):
                        text += f'{c.name}: {c.decision.action} | active {int(counts[0])} | recruited {int(counts[1])}/139255\n'
                    text += f'Tab: focus {duel.contestants[focus[0]].name} | 1 odor 2 taste 3 touch 4 threat 5 rival | R reward P penalty\n'
                    text += f'N: reset bodies ({duel.body_resets}) | Decoder updates: {duel.reward_updates} | CPU body, GPU brain'
                    view.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100,mujoco.mjtGridPos.mjGRID_TOPLEFT,text,''))
                    view.sync()
                if time.perf_counter()-last_report >= 5:
                    print(f'{duel.elapsed_s:.2f}s | '+ ' | '.join(f'{c.name}: {c.decision.action}' for c in duel.contestants),flush=True)
                    last_report = time.perf_counter()
    except BaseException as exc:
        failure = repr(exc)
        raise
    finally:
        if view: view.close()
        if duel:
            report.update(duel.summary())
            report.update(body_resets=duel.body_resets,reward_updates=duel.reward_updates,reward_update_norms=duel.reward_update_norms,teaching_events=teaching_events)
            for i,model in enumerate(duel.readouts):model.save(out/f'readout-fly-{i}.npz')
            duel.arena.close()
        if isinstance(brain,Worker): brain.close()
        report.update(wall_seconds=time.perf_counter()-started,error=failure)
        (out/'report.json').write_text(json.dumps(report,indent=2,default=_json_default)+'\n')


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--seconds',type=float,default=0.)
    p.add_argument('--headless',action='store_true')
    p.add_argument('--control',choices=('intact','null','disconnected','shuffled'),default='intact')
    p.add_argument('--output',type=Path)
    p.add_argument('--seed',type=int,default=1)
    p.add_argument('--backend',choices=('cpu','gpu'),default='gpu')
    args=p.parse_args()
    if not np.isfinite(args.seconds) or args.seconds<0:p.error('seconds must be finite and nonnegative')
    run(seconds=args.seconds,viewer=not args.headless,control=args.control,output=args.output,seed=args.seed,backend=args.backend)
