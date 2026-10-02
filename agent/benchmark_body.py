"""Seeded body-only benchmark of installed hybrid CPG plus contact escape.

This engineering controller benchmark excludes neural/vision computation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import time

import mujoco
import numpy as np
from agent.desire import Desire
from agent.neural_connectome import digest
from agent.neural_single import EthologyArena, step_signed, wall_observation
from fly_arena.behavior import BehaviorDecision
from agent.body_suite import FlatArena


def trial(seed, seconds, wall=False, flat=False):
    arena = FlatArena(seed=seed) if flat else EthologyArena(seed=seed)
    desire = Desire()
    started = time.perf_counter()
    records = []
    failure = None
    reset_replay = False
    escaped = None
    contacts = 0
    try:
        terrain = arena.terrain_receipt() if flat else None
        if wall:
            arena.sim.mj_data.qpos[:3] = [19.2, 0, .8]
            arena.sim.mj_data.qpos[3:7] = [1, 0, 0, 0]
            arena.sim.mj_data.qvel[:] = 0
            mujoco.mj_forward(arena.sim.mj_model, arena.sim.mj_data)
        initial = arena.position.copy()
        snapshot = arena.get_state()
        # Verify the simulator/controller reset boundary by replaying one action.
        step_signed(arena, BehaviorDecision('WALK'), [.4, .4])
        expected = arena.sim.mj_data.qpos.copy()
        arena.set_state(snapshot)
        step_signed(arena, BehaviorDecision('WALK'), [.4, .4])
        reset_replay = bool(np.allclose(arena.sim.mj_data.qpos, expected, atol=1e-10, rtol=0))
        arena.set_state(snapshot)
        for step in range(round(seconds / .01)):
            contact, turn = wall_observation(arena)
            contacts += int(contact)
            desire.evaluate(.8, np.zeros(8), arena.position, False, arena.upright,
                            wall_contact=contact, wall_turn=turn)
            drive = desire.modulate([.4, .4, 0, 0, 0], True)[:2]
            step_signed(arena, BehaviorDecision('WALK'), drive)
            finite = bool(np.isfinite(arena.sim.mj_data.qpos).all()
                          and np.isfinite(arena.sim.mj_data.qvel).all())
            record = {'time_s': (step + 1) * .01, 'upright': arena.upright,
                      'position_mm': arena.position.tolist(), 'wall_contact': contact,
                      'drive': drive.tolist(), 'finite': finite}
            records.append(record)
            if (step + 1) % 1000 == 0:
                print(json.dumps({'progress_seed': seed, 'scenario': 'wall' if wall else 'floor',
                                  'simulated_seconds': record['time_s']}), flush=True)
            if not finite:
                failure = 'nonfinite_physics'
                break
            # East-wall fixture: move at least 1 mm inward and face inward.
            if wall and contacts and escaped is None and arena.upright > .85:
                if arena.position[0] <= initial[0] - 1 and np.cos(arena.heading) < 0:
                    escaped = record['time_s']
        upright_fraction = sum(r['upright'] > .85 for r in records) / max(1, len(records))
        if failure is None and upright_fraction < .95:
            failure = 'balance'
        if failure is None and wall and not contacts:
            failure = 'wall_fixture_no_contact'
        if failure is None and wall and escaped is None:
            failure = 'wall_escape_timeout'
        if not reset_replay:
            failure = 'reset_replay_mismatch'
        return {'seed': seed, 'scenario': 'east_wall' if wall else 'central_floor',
                'simulated_seconds': len(records) * .01, 'upright_fraction': upright_fraction,
                'wall_escape_seconds': escaped, 'wall_contact_exchanges': contacts,
                'reset_replay': reset_replay, 'failure': failure,
                'terrain': terrain,
                'wall_seconds': time.perf_counter() - started, 'trajectory': records}
    finally:
        arena.close()


def seed_trials(seed, seconds, flat, output):
    rows = []
    for wall, duration in [(False, seconds), (True, 5.)]:
        row = trial(seed, duration, wall, flat)
        (output / f'{seed:02d}-{row["scenario"]}.json').write_text(json.dumps(row, indent=2)+'\n')
        rows.append({key: value for key, value in row.items() if key != 'trajectory'})
    return rows


def benchmark(output, seeds, seconds, suite=None, workers=1):
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    started = time.perf_counter()
    manifest = json.loads(suite.read_text()) if suite else None
    if manifest:
        for source, expected in manifest['source_sha256'].items():
            if digest(source) != expected:
                raise ValueError(f'frozen suite source changed: {source}')
    if workers == 1:
        batches = (seed_trials(seed, seconds, bool(suite), output) for seed in seeds)
        for batch in batches:
            rows.extend(batch)
            print(json.dumps(batch), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = [pool.submit(seed_trials, seed, seconds, bool(suite), output) for seed in seeds]
            for future in as_completed(futures):
                batch = future.result()
                rows.extend(batch)
                print(json.dumps(batch), flush=True)
    rows.sort(key=lambda row: (row['seed'], row['scenario']))
    if manifest:
        for source, expected in manifest['source_sha256'].items():
            if digest(source) != expected:
                raise ValueError(f'frozen suite source changed during run: {source}')
        assert all(row['terrain'] == manifest['terrain_receipt'] for row in rows)
    flat = [row for row in rows if row['scenario'] == 'central_floor']
    walls = [row for row in rows if row['scenario'] == 'east_wall']
    full_suite = bool(suite) and list(seeds) == list(range(20)) and seconds == 60.
    escapes = sum(row['wall_escape_seconds'] is not None for row in walls) / len(walls)
    passed = (all(row['upright_fraction'] >= .95 and row['failure'] is None for row in flat)
              and escapes >= .9 and all(row['reset_replay'] for row in rows)
              and all(row['failure'] not in ('nonfinite_physics', 'wall_fixture_no_contact') for row in rows))
    report = {'mode': 'engineering_agent', 'strategy': 'installed HybridTurningController CPG/reflexes + Desire contact escape',
              'scope': 'body-only central-floor and east-wall fixtures; no neural/learning/vision, self-righting, climbing or UI reset claim',
              'suite': 'full' if full_suite else 'pilot',
              'status': ('PASS' if passed else 'FAIL') if full_suite else ('PILOT_PASS' if passed else 'PILOT_FAIL'),
              'P1_acceptance': 'UNVERIFIED',
              'body_metric_criteria_pass': passed,
              'remaining_acceptance_gates': ([] if suite else ['verified flat-terrain suite']) + ['actual Fly controls UI reset'],
              'suite_manifest': str(suite) if suite else None,
              'suite_manifest_sha256': digest(suite) if suite else None,
              'workers': workers,
              'seeds': list(seeds), 'flat_seconds_per_seed': seconds,
              'upright_threshold': .85, 'required_upright_fraction': .95,
              'wall_escape_definition': 'east-wall contact observed, thorax >=1mm inward and heading inward, upright, within 5s',
              'wall_escape_fraction': escapes, 'wall_seconds': time.perf_counter() - started,
              'rows': rows,
              'sources': {str(path): digest(path) for path in
                          (Path(__file__), Path(__import__('agent.desire', fromlist=['x']).__file__),
                           Path(__import__('fly_arena.motor', fromlist=['x']).__file__))}}
    (output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, default=20)
    parser.add_argument('--seconds', type=float, default=60.)
    parser.add_argument('--suite', type=Path, help='frozen flat-suite manifest')
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.seeds <= 20 or not np.isfinite(args.seconds) or args.seconds <= 0:
        parser.error('seeds must be 1..20 and seconds finite and positive')
    if not 1 <= args.workers <= 4:
        parser.error('workers must be 1..4')
    result = benchmark(args.output, range(args.seeds), args.seconds, args.suite, args.workers)
    print(json.dumps({key: value for key, value in result.items() if key != 'rows'}), flush=True)
    if result['status'].endswith('FAIL'):
        raise SystemExit(1)
