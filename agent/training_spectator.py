"""One read-only evaluation body following atomic shared decoder snapshots."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import signal
import threading
import time

import numpy as np

from agent.neural_connectome import CPUBrain, Readout, digest
from agent.neural_single import EthologyArena, Environment, Organism, encode, step_signed
from agent.odor_memory_gate import ABSENT, GAITS, scenes
from fly_arena.behavior import BehaviorDecision


def run(model_path, output, seconds=60., viewer=True):
    output.mkdir(parents=True, exist_ok=False)
    stopping = False
    def stop(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    scene = scenes([50000])[1]  # evaluation layout never enters training
    arena = EthologyArena(seed=0, food_patches=[scene['food']], eye_size=16)
    arena.eye_renderer.close()
    brain = CPUBrain(flies=1)
    initial = arena.get_state()
    view = None
    viewer_threads = []
    episodes = []
    started = time.perf_counter()
    try:
        # The eight pre-decision observations are deterministic for this fixed
        # evaluation layout. Cache this reference before opening the viewer.
        environment = Environment([scene['food']], seed=scene['seed'])
        organism = Organism({})
        cached_inputs = []
        for _ in range(8):
            values = encode(environment.sense(arena, organism), ABSENT)
            cached_inputs.append(np.asarray(values).copy())
            neural = brain.step(np.asarray(values)[None])
            events = step_signed(arena, BehaviorDecision('IDLE'), [0.,0.])
            environment.apply_physical_events(events, organism, .01)
            organism.advance(.01,'IDLE')
        cached_features = np.asarray(neural['features']).copy()
        arena.set_state(initial)
        if viewer:
            import mujoco.viewer
            threads_before = set(threading.enumerate())
            view = mujoco.viewer.launch_passive(arena.sim.mj_model, arena.sim.mj_data,
                                               show_left_ui=False, show_right_ui=False)
            viewer_threads = [thread for thread in threading.enumerate() if thread not in threads_before]
            with view.lock():
                view.cam.distance = 7
                view.cam.azimuth = 125
                view.cam.elevation = -45
                view.cam.lookat[:] = arena.position
        ready = {'pid': os.getpid(), 'viewer': viewer, 'learning': False,
                 'neural_backend': 'CPU', 'scene': scene, 'started_unix': time.time()}
        ready['sensory_reference_cache'] = 'eight deterministic CPU exchanges, exact observation check each episode'
        (output/'ready.json').write_text(json.dumps(ready, indent=2)+'\n')
        print('EVALUATION_SPECTATOR_READY', flush=True)
        with (output/'trace.jsonl').open('w', buffering=1) as trace:
            while not stopping and time.perf_counter()-started < seconds:
                if view and not view.is_running():
                    break
                with view.lock() if view else nullcontext():
                    arena.set_state(initial)
                environment = Environment([scene['food']], seed=scene['seed'])
                organism = Organism({})
                # Read a complete atomic snapshot once per episode; never write it.
                snapshot = output/f'episode-{len(episodes)}.npz'
                snapshot.write_bytes(model_path.read_bytes())
                model = Readout(snapshot)
                model_hash = digest(snapshot)
                minimum_upright = 1.
                for step in range(138):
                    if stopping or (view and not view.is_running()):
                        break
                    tick = time.perf_counter()
                    if step < 8:
                        inputs = encode(environment.sense(arena, organism), ABSENT)
                        assert np.array_equal(inputs,cached_inputs[step]), 'evaluation sensory cache invalid'
                        if step == 7:
                            motor = model.predict(cached_features)[0]
                            gait = GAITS['left'] if motor[1] > motor[0] else GAITS['right']
                    action = 'IDLE' if step < 8 else 'WALK' if step < 58 else 'FEED'
                    lock = view.lock() if view else nullcontext()
                    with lock:
                        events = step_signed(arena, BehaviorDecision(action), gait if action == 'WALK' else [0., 0.])
                        environment.apply_physical_events(events, organism, .01)
                        organism.advance(.01, action)
                    assert np.isfinite(arena.sim.mj_data.qvel).all()
                    minimum_upright = min(minimum_upright, float(arena.upright))
                    if view:
                        view.sync()
                    trace.write(json.dumps({'unix': time.time(), 'episode': len(episodes), 'step': step,
                        'model_sha256': model_hash, 'action': action, 'position': arena.position.tolist(),
                        'upright': float(arena.upright), 'ingested': organism.state.ingested_total})+'\n')
                    if viewer:
                        time.sleep(max(0., .01-(time.perf_counter()-tick)))
                episodes.append({'model_sha256': model_hash, 'ingested': organism.state.ingested_total,
                                 'minimum_upright': minimum_upright, 'completed_steps': step+1})
                if not viewer:
                    break
        (output/'metrics.json').write_text(json.dumps({'status': 'PASS', 'episodes': episodes,
            'learning_updates': 0, 'wall_seconds': time.perf_counter()-started,
            'scope': 'one visible CPU evaluation fly, snapshot policy, prescribed turn/feed primitives'}, indent=2)+'\n')
    finally:
        if view:
            view.close()
            # close() requests exit; the passive render thread is asynchronous.
            # Wait before destroying native model/data or running GLFW atexit.
            for thread in viewer_threads:
                thread.join(timeout=15)
                if thread.is_alive():
                    raise TimeoutError('spectator render thread did not stop')
        arena.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=60.)
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()
    run(args.model, args.output, args.seconds, not args.headless)
