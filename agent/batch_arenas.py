"""Persistent CPU body workers and a batched resident neural client benchmark."""
from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
from multiprocessing.connection import Client, Listener
import os
from pathlib import Path
import time

import numpy as np
import psutil

from agent.neural_connectome import CPUBrain, Readout, Worker, digest
from agent.neural_single import EthologyArena, Environment, Organism, encode, step_signed
from agent.odor_memory_gate import ABSENT, GAITS, scenes
from fly_arena.behavior import BehaviorDecision


def body_worker(connection, configs):
    bodies = {}
    initial_bodies = {}
    try:
        for index, config in configs:
            arena = EthologyArena(seed=config['seed'], food_patches=[config['food']], eye_size=16)
            # This is an odor-only workload. Free unused rendering resources.
            arena.eye_renderer.close()
            environment, organism = Environment([config['food']], seed=config['seed']), Organism({})
            bodies[index] = (arena, environment, organism,
                             arena.get_state(), environment.get_state(), organism.get_state())
            initial_bodies[index] = copy.deepcopy(bodies[index][3])
        connection.send({'ready': True, 'pid': os.getpid(), 'actors': list(bodies)})
        while True:
            request = connection.recv()
            if request['op'] == 'quit':
                break
            started = time.perf_counter(); rows = []
            for index, values in request['actors']:
                arena, environment, organism, a0, e0, o0 = bodies[index]
                if request['op'] == 'configure':
                    food = copy.deepcopy(values['food'])
                    a0 = copy.deepcopy(initial_bodies[index])
                    a0['food_patches'] = [food]
                    geom = arena.sim.mj_model.geom('food_0')
                    geom.pos[:] = [food['x'], food['y'], food.get('surface_z', .01)-.005]
                    geom.size[:2] = [food['radius'], .005]
                    environment = Environment([food], seed=values['seed'])
                    organism = Organism({})
                    e0, o0 = environment.get_state(), organism.get_state()
                    bodies[index] = (arena, environment, organism, a0, e0, o0)
                    arena.set_state(a0)
                    reward = 0.
                elif request['op'] == 'checkpoint':
                    bodies[index] = (arena, environment, organism, arena.get_state(),
                                     environment.get_state(), organism.get_state())
                    reward = 0.
                elif request['op'] == 'reset':
                    arena.set_state(a0); environment.set_state(e0); organism.set_state(o0)
                    reward = 0.
                else:
                    action, gait = values
                    before = organism.state.ingested_total
                    events = step_signed(arena, BehaviorDecision(action), gait)
                    environment.apply_physical_events(events, organism, .01)
                    organism.advance(.01, action)
                    reward = float(organism.state.ingested_total-before)
                inputs = encode(environment.sense(arena, organism), ABSENT)
                assert np.isfinite(arena.sim.mj_data.qvel).all()
                rows.append({'index': index, 'inputs': inputs.tolist(), 'reward': reward,
                    'position': arena.position.tolist(), 'upright': float(arena.upright),
                    'ingested': float(organism.state.ingested_total),
                    'physics_steps': arena.physics_steps})
            connection.send({'rows': rows, 'body_seconds': time.perf_counter()-started,
                             'rss': psutil.Process().memory_info().rss})
    except BaseException as error:
        connection.send({'error': repr(error)})
        raise
    finally:
        for arena, *_ in bodies.values():
            arena.close()
        connection.close()


class BodyPool:
    def __init__(self, count=10, workers=4, uniform_seed=None):
        context = mp.get_context('spawn')
        configs = list(enumerate(scenes(range(5))[:count]))
        if uniform_seed is not None:
            configs = [(i, dict(config, seed=uniform_seed)) for i, config in configs]
        self.children = []; self.owner = {}; self.metadata = []
        for group in range(min(count, workers)):
            parent, child = context.Pipe()
            process = context.Process(target=body_worker, args=(child, configs[group::min(count, workers)]))
            process.start(); child.close(); self.children.append((process, parent))
        for group, (process, connection) in enumerate(self.children):
            if not connection.poll(120):
                raise TimeoutError('body worker initialization')
            row = connection.recv()
            if 'error' in row:
                raise RuntimeError(row['error'])
            self.metadata.append(row)
            for index in row['actors']:
                self.owner[index] = group

    def request(self, op, actors):
        groups = {}
        for index, values in actors:
            groups.setdefault(self.owner[index], []).append((index, values))
        for group, values in groups.items():
            self.children[group][1].send({'op': op, 'actors': values})
        results = []
        for group in groups:
            connection = self.children[group][1]
            if not connection.poll(60):
                raise TimeoutError('body step')
            row = connection.recv()
            if 'error' in row:
                raise RuntimeError(row['error'])
            results.append(row)
        return {'rows': sorted([r for result in results for r in result['rows']], key=lambda r: r['index']),
                'max_body_seconds': max(r['body_seconds'] for r in results),
                'body_worker_rss': sum(psutil.Process(p.pid).memory_info().rss for p, _ in self.children)}

    def close(self):
        for process, connection in self.children:
            if process.is_alive():
                connection.send({'op': 'quit'})
        for process, connection in self.children:
            process.join(10)
            if process.is_alive():
                process.terminate(); process.join()
            connection.close()


def serve(socket_path, count=10, workers=4, uniform_seed=None):
    pool = BodyPool(count=count, workers=workers, uniform_seed=uniform_seed)
    listener = Listener(str(socket_path), family='AF_UNIX')
    os.chmod(socket_path, 0o600)
    socket_path.with_suffix('.ready.json').write_text(json.dumps({
        'server_pid': os.getpid(), 'body_workers': pool.metadata, 'total_bodies': count,
        'worker_count': workers})+'\n')
    print(json.dumps({'physics_ready': True, 'workers': workers, 'bodies': count}), flush=True)
    running = True
    try:
        while running:
            connection = listener.accept()
            try:
                while True:
                    try:
                        request = connection.recv()
                    except (EOFError, ConnectionResetError):
                        break
                    if request['op'] == 'shutdown':
                        running = False; connection.send({'shutdown': True}); break
                    result = pool.request(request['op'], request['actors'])
                    result['server_rss'] = psutil.Process().memory_info().rss
                    try:
                        connection.send(result)
                    except (BrokenPipeError, ConnectionResetError):
                        # A cancelled queued client must not take down CPU bodies.
                        break
            finally:
                connection.close()
    finally:
        listener.close(); pool.close()
        socket_path.unlink(missing_ok=True)


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    for count in (1, 2, 4, 10):
        brain = CPUBrain(flies=count); inputs = np.zeros((3, count, 8), np.float32)
        # Distinct lane stimuli plus a silent final lane prove independence.
        for step in range(3):
            for lane in range(count if count == 1 else count-1):
                inputs[step, lane, (lane+step)%8] = .4+.1*(lane%4)
        features, counts = [], []
        for values in inputs:
            row = brain.step(values); features.append(row['features']); counts.append(row['counts'])
        np.savez(output/f'oracle-c{count}.npz', inputs=inputs,
                 features=np.asarray(features), counts=np.asarray(counts))
    source = Path(__file__)
    (output/'manifest.json').write_text(json.dumps({'source_sha256': digest(source),
        'counts': [1,2,4,10], 'physics': 'ten complete independent CPU bodies, four persistent workers',
        'neural': 'one resident GPU worker with shared CSR and separate state per lane',
        'task': 'odor-only turn/feed primitives; no vision, viewer, social coupling or GPU physics',
        'dt_s': .01, 'physics_dt_s': .0001,
        'oracle_sha256': {str(path.name): digest(path) for path in output.glob('oracle-*.npz')}},indent=2)+'\n')


def benchmark(socket_path, fixture, output, count, binary=None, steps=20, warmup=3, learn=False, publish=None,
              monitor_pids=()):
    output.mkdir(parents=True, exist_ok=False)
    oracle = np.load(fixture/f'oracle-c{count}.npz')
    vram_path = Path('/sys/class/drm/card1/device/mem_info_vram_used')
    vram_before = int(vram_path.read_text()) if binary and vram_path.exists() else None
    startup = time.perf_counter()
    brain = Worker(binary=binary, flies=count) if binary else CPUBrain(flies=count)
    startup_s = time.perf_counter()-startup
    connection = Client(str(socket_path), family='AF_UNIX')
    model = Readout(Path('.work/agent/odor-memory/gate-v1/trained.npz'))
    rows = []; updates = 0; peak_rss = 0; errors = []; vram_samples = []
    try:
        for index, inputs in enumerate(oracle['inputs']):
            result = brain.step(inputs)
            errors.append(float(np.max(np.abs(np.asarray(result['features'])-oracle['features'][index]))))
            assert np.array_equal(result['counts'], oracle['counts'][index])
        assert max(errors) <= 1e-5, errors
        brain.reset()
        connection.send({'op': 'reset', 'actors': [(i, None) for i in range(count)]})
        physical = connection.recv(); selected = [GAITS['straight'] for _ in range(count)]
        model_before = model.weights.copy()
        if publish:
            publish.parent.mkdir(parents=True, exist_ok=True)
            temporary = publish.with_name(publish.stem+'.pending.npz')
            model.save(temporary); temporary.replace(publish)
        for step in range(steps+warmup):
            started = time.perf_counter()
            inputs = np.asarray([row['inputs'] for row in physical['rows']], np.float32)
            t0 = time.perf_counter(); neural = brain.step(inputs); neural_s = time.perf_counter()-t0
            features = np.asarray(neural['features']); t0 = time.perf_counter()
            decoded = model.predict(features)
            if step == 7:
                selected = [GAITS['left'] if row[1] > row[0] else GAITS['right'] for row in decoded]
            action = 'IDLE' if step < 8 else 'WALK' if step < 58 else 'FEED'
            commands = [(i, (action, selected[i] if action == 'WALK' else [0.,0.])) for i in range(count)]
            decode_s = time.perf_counter()-t0
            t0 = time.perf_counter(); connection.send({'op': 'step', 'actors': commands})
            physical = connection.recv(); physics_s = time.perf_counter()-t0
            if learn:
                for index, body in enumerate(physical['rows']):
                    if body['reward'] > 0:
                        updates += int(model.reinforce(features[index], 'FEED', min(1.,body['reward']*10)) > 0)
            elapsed = time.perf_counter()-started
            if publish and learn and step % 10 == 0:
                model.save(temporary); temporary.replace(publish)
            rss = psutil.Process().memory_info().rss + physical['body_worker_rss']+physical['server_rss']
            if binary:
                rss += psutil.Process(brain.proc.pid).memory_info().rss
            rss += sum(psutil.Process(pid).memory_info().rss for pid in monitor_pids)
            peak_rss = max(peak_rss, rss)
            if vram_before is not None:
                vram_samples.append(int(vram_path.read_text()))
            row = {'step': step, 'loop_s': elapsed, 'neural_s': neural_s, 'decode_s': decode_s,
                   'physics_round_trip_s': physics_s, 'max_body_worker_s': physical['max_body_seconds'],
                   'rss_sample_bytes': rss, 'bodies': physical['rows']}
            assert all(body['upright'] > .85 for body in physical['rows'])
            rows.append(row)
        measured = rows[warmup:]; seconds = sum(row['loop_s'] for row in measured)
        model.save(output/'readout.npz')
        if publish:
            model.save(temporary); temporary.replace(publish)
        server_info = json.loads(socket_path.with_suffix('.ready.json').read_text())
        report = {'status': 'PASS', 'flies': count, 'body_workers': server_info['worker_count'],
            'allocated_bodies': 10, 'steps': steps, 'warmup_steps': warmup,
            'wall_seconds_measured': seconds, 'simulated_seconds_per_fly': steps*.01,
            'aggregate_fly_seconds_per_wall_second': count*steps*.01/seconds,
            'per_fly_real_time_ratio': steps*.01/seconds,
            'median_loop_ms': float(np.median([row['loop_s'] for row in measured])*1000),
            'p95_loop_ms': float(np.quantile([row['loop_s'] for row in measured], .95)*1000),
            'component_seconds': {key: sum(row[key] for row in measured) for key in ['neural_s','decode_s','physics_round_trip_s']},
            'sampled_total_process_rss_bytes': peak_rss, 'cpu_gpu_feature_error': max(errors),
            'additional_monitored_pids': list(monitor_pids),
            'neural_worker_startup_s': startup_s,
            'device_wide_vram_before_bytes': vram_before,
            'device_wide_vram_peak_bytes': max(vram_samples) if vram_samples else None,
            'vram_scope': 'device-wide sampled allocation including desktop, not per-worker allocation',
            'finite_body_exchanges': count*len(rows), 'shared_decoder_updates': updates,
            'readout_weight_delta_norm': float(np.linalg.norm(model.weights-model_before)),
            'ingested_food_units_by_fly': [body['ingested'] for body in physical['rows']],
            'worker_binary_sha256': digest(Path(binary)) if binary else None,
            'scope': 'headless odor-only full body/neural exchanges; no vision, live UI or whole-body GPU claim'}
        (output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
        (output/'exchanges.json').write_text(json.dumps(rows)+'\n')
        print(json.dumps(report),flush=True)
        return report
    finally:
        connection.close()
        if binary:
            brain.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('serve','prepare','benchmark','shutdown'))
    parser.add_argument('--socket', type=Path, default=Path('.work/agent/ten-fly/physics.sock'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--fixture', type=Path, default=Path('.work/agent/ten-fly/fixture'))
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--uniform-seed', type=int)
    parser.add_argument('--publish', type=Path)
    parser.add_argument('--binary', type=Path)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--learn', action='store_true')
    args = parser.parse_args()
    if args.stage == 'serve': serve(args.socket, workers=args.workers, uniform_seed=args.uniform_seed)
    elif args.stage == 'prepare': prepare(args.output)
    elif args.stage == 'shutdown':
        connection = Client(str(args.socket),family='AF_UNIX'); connection.send({'op':'shutdown'});print(connection.recv());connection.close()
    else: benchmark(args.socket,args.fixture,args.output,args.count,args.binary,args.steps,args.warmup,args.learn,args.publish)
