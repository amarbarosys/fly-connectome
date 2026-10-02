"""Queued, paired headless/viewer measurement with one evaluation spectator."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from agent.batch_arenas import benchmark
from agent.launch_neural_flies import desktop_environment

ROOT = Path('.work/agent/spectator-learning')
BINARY = Path('.work/agent/ten-fly/bin/worker-c10-optimized')
FIXTURE = Path('.work/agent/ten-fly/fixture')
SOCKET = ROOT/'physics.sock'


def run():
    publish = ROOT/'live-readout.npz'
    shutil.copyfile('.work/agent/odor-memory/gate-v1/trained.npz', publish)
    reports = []
    exit_codes = []
    spectator = None
    log = (ROOT/'spectator.log').open('w')
    try:
        for round_index in range(2):
            # Counterbalance order to reduce first-run/cache drift.
            for visible in ([False, True] if round_index == 0 else [True, False]):
                if visible and spectator is None:
                    env = desktop_environment()
                    env.update(LIBGL_ALWAYS_SOFTWARE='1', LP_NUM_THREADS='2',
                               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
                    spectator = subprocess.Popen([sys.executable, '-m', 'agent.training_spectator',
                        '--model', str(publish), '--output', str(ROOT/'visible'), '--seconds', '120'],
                        env=env, stdout=log, stderr=subprocess.STDOUT)
                    deadline = time.monotonic()+30
                    while not (ROOT/'visible/ready.json').exists():
                        if spectator.poll() is not None or time.monotonic()>deadline:
                            raise RuntimeError('spectator failed readiness; inspect spectator.log')
                        time.sleep(.05)
                if not visible and spectator is not None:
                    spectator.terminate(); code = spectator.wait(timeout=20); spectator = None
                    exit_codes.append(code)
                    if code != 0:
                        raise RuntimeError(f'spectator shutdown failed: {code}')
                report = benchmark(SOCKET, FIXTURE, ROOT/f'paired-{round_index}-{int(visible)}',
                                   10, BINARY, steps=20, warmup=3,
                                   monitor_pids=[spectator.pid] if spectator else [])
                reports.append(dict(report, visible=visible, round=round_index))
                if round_index == 1 and visible:
                    # Full feedback test publishes changed shared weights while the spectator watches.
                    benchmark(SOCKET, FIXTURE, ROOT/'training-with-viewer', 10, BINARY,
                              steps=138, warmup=0, learn=True, publish=publish,
                              monitor_pids=[spectator.pid])
        (ROOT/'spectator-comparison.json').write_text(json.dumps({'status': 'PASS', 'pairs': reports,
            'spectator_exit_codes': exit_codes,
            'renderer': 'software; one visible evaluation fly and ten headless training bodies',
            'measurement_scope': 'single paired short-run experiment, two samples per condition'}, indent=2)+'\n')
    finally:
        if spectator is not None:
            spectator.terminate(); spectator.wait(timeout=15)
        log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--output-root', type=Path, default=ROOT)
    args = parser.parse_args()
    ROOT = args.output_root
    ROOT.mkdir(parents=True,exist_ok=True)
    if args.preflight:
        assert BINARY.is_file() and (FIXTURE/'oracle-c10.npz').is_file()
        assert SOCKET.exists()
        env = desktop_environment()
        assert env.get('DISPLAY')
        print(json.dumps({'status': 'PASS', 'display': env['DISPLAY']}))
    else:
        run()
