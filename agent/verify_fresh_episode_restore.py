"""Fresh-process CPU no-vision continuation gate using existing state contracts."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from agent.neural_checkpoint import STATE_FIELDS, restore_checkpoint, save_checkpoint
from agent.verify_episode_replay import EpisodeFixture, tree_equal
from fly_arena.checkpoint import load_checkpoint, save_checkpoint as save_episode
from agent.neural_connectome import digest


def stage(out, mode):
    episode = EpisodeFixture()
    try:
        if mode == 'record':
            prefix = [episode.step() for _ in range(20)]
            assert any(row['readout_update_norm'] > 0 for row in prefix)
            save_checkpoint(episode.brain, out / 'prefix-neural.npz')
            save_episode(out / 'prefix-episode.npz', episode.state(), episode.compatibility)
            rows = [episode.step() for _ in range(30)]
            save_checkpoint(episode.brain, out / 'final-neural.npz')
            save_episode(out / 'final-episode.npz', episode.state(), episode.compatibility)
            (out / 'expected-events.json').write_text(json.dumps(rows, allow_nan=False))
        else:
            saved, _ = load_checkpoint(out / 'prefix-episode.npz', episode.compatibility)
            restore_checkpoint(episode.brain, out / 'prefix-neural.npz')
            episode.restore(saved)
            rows = [episode.step() for _ in range(30)]
            assert rows == json.loads((out / 'expected-events.json').read_text()), 'fresh-process events differ'
            expected, _ = load_checkpoint(out / 'final-episode.npz', episode.compatibility)
            assert tree_equal(expected, episode.state()), 'fresh-process final episode differs'
            actual = {name: getattr(episode.brain, name).copy() for name in STATE_FIELDS}
            restore_checkpoint(episode.brain, out / 'final-neural.npz')
            assert all(np.array_equal(actual[name], getattr(episode.brain, name)) for name in STATE_FIELDS), 'fresh-process final neural state differs'
            assert np.isfinite(episode.arena.sim.mj_data.qpos).all()
        (out / f'{mode}-process.json').write_text(json.dumps({'pid': os.getpid(), 'status': 'PASS'}))
    finally:
        episode.close()


def verify(out):
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    for mode in ('record', 'restore'):
        with (out / f'{mode}.log').open('w') as log:
            subprocess.run([sys.executable, '-m', 'agent.verify_fresh_episode_restore',
                            '--out', str(out), '--stage', mode], stdout=log,
                           stderr=subprocess.STDOUT, check=True, timeout=180)
    receipts = {mode: json.loads((out / f'{mode}-process.json').read_text())
                for mode in ('record', 'restore')}
    assert receipts['record']['pid'] != receipts['restore']['pid']
    result = {'status': 'PASS', 'processes': receipts, 'exact_events': True,
              'exact_final_episode_and_neural_state': True, 'replayed_exchanges': 30,
              'wall_seconds': time.perf_counter() - started,
              'scope': 'fresh-process CPU no-vision engineered teacher fixture',
              'unverified': ['live runner restore', 'vision history', 'GPU worker state',
                             'model promotion', 'biological memory'],
              'source_sha256': {name: digest(Path(__file__).with_name(name)) for name in
                                ('verify_fresh_episode_restore.py', 'verify_episode_replay.py',
                                 'neural_checkpoint.py')},
              'artifact_sha256': {path.name: digest(path) for path in out.glob('*.npz')}}
    (out / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stage', choices=('record', 'restore'))
    args = parser.parse_args()
    if args.stage:
        stage(args.out, args.stage)
    else:
        verify(args.out)
