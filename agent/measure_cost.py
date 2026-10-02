"""Measure the existing single-fly CPU loop, including rendered eyes.

Run with MUJOCO_GL=egl LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe.
Instrumentation is scoped to this process; no runner behavior is replaced.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
import functools
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import time
from unittest.mock import patch

import numpy as np
from agent import neural_single as runner
from agent.neural_connectome import digest


def measure(output: Path, steps: int, warmup: int, backend='cpu'):
    output.mkdir(parents=True, exist_ok=False)
    samples = []
    totals = {}
    index = 0
    boundary = None
    renderer = None
    loop_finished = False
    vram_paths = list(Path('/sys/class/drm').glob('card*/device/mem_info_vram_used'))
    vram_samples = []
    worker_rss_kib = []
    initial_vram = {str(path): int(path.read_text()) for path in vram_paths}

    def sample_worker(function):
        @functools.wraps(function)
        def wrapped(worker, *args, **kwargs):
            result = function(worker, *args, **kwargs)
            status = Path(f'/proc/{worker.proc.pid}/status').read_text()
            worker_rss_kib.append(int(next(line.split()[1] for line in status.splitlines()
                                           if line.startswith('VmRSS:'))))
            return result
        return wrapped

    def finish_loop(function):
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            nonlocal loop_finished
            if boundary is not None and index > warmup:
                samples.append({'step': index, 'loop_seconds': time.perf_counter() - boundary, **totals})
            loop_finished = True
            return function(*args, **kwargs)
        return wrapped

    def timed(name, function):
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            nonlocal index, boundary, renderer
            start = time.perf_counter()
            result = function(*args, **kwargs)
            elapsed = time.perf_counter() - start
            if name == 'eyes':
                from OpenGL.GL import glGetString, GL_RENDERER
                renderer = glGetString(GL_RENDERER).decode()
                if backend == 'cpu' and 'llvmpipe' not in renderer.lower():
                    raise RuntimeError(f'CPU-only renderer gate failed: {renderer}')
                if backend == 'gpu' and ('7900' not in renderer or 'llvmpipe' in renderer.lower()):
                    raise RuntimeError(f'Hardware renderer gate failed: {renderer}')
                vram_samples.append({str(path): int(path.read_text()) for path in vram_paths})
                if boundary is not None and index > warmup:
                    samples.append({'step': index, 'loop_seconds': start - boundary,
                                    **totals})
                boundary = start
                totals.clear()
                index += 1
            totals[name] = totals.get(name, 0.) + elapsed
            return result
        return wrapped

    source_paths = [Path(runner.__file__), Path(__file__),
                    Path(__import__('agent.neural_connectome', fromlist=['x']).__file__),
                    Path(__import__('fly_arena.ethology_arena', fromlist=['x']).__file__)]
    started = time.perf_counter()
    cpu_started = time.process_time()
    error = None
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(runner.EthologyArena, 'close',
                                finish_loop(runner.EthologyArena.close)))
            if backend == 'gpu':
                stack.enter_context(patch.object(runner.Worker, 'step', sample_worker(runner.Worker.step)))
            for owner, attribute, name in [
                (runner.EthologyArena, 'eyes', 'eyes'),
                (runner.EyeVision, 'step', 'vision'),
                (runner.Environment, 'sense', 'sensing'),
                (runner.Desire, 'evaluate', 'motivation'),
                (runner.CPUBrain if backend == 'cpu' else runner.Worker, 'step', 'neural'),
                (runner.Readout, 'predict', 'readout'),
                (runner.MotorDecision, 'decide', 'decision'),
                (runner, 'step_signed', 'controller_physics'),
                (runner.Environment, 'apply_physical_events', 'physical_events'),
                (runner.Organism, 'advance', 'organism'),
            ]:
                stack.enter_context(patch.object(owner, attribute,
                    timed(name, getattr(owner, attribute))))
            runner.run(output / 'run', backend=backend, seconds=(steps + warmup) * .01,
                       viewer=False, audit=False)
        # Last interval ends before cleanup and artifact hashing.
        if not loop_finished and boundary is not None and index > warmup:
            samples.append({'step': index, 'loop_seconds': time.perf_counter() - boundary, **totals})
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        wall = time.perf_counter() - started
        cpu = time.process_time() - cpu_started
        samples = samples[:steps]
        fields = ['step', 'loop_seconds', *sorted(totals)]
        with (output / 'components.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(samples)
        durations = np.array([row['loop_seconds'] for row in samples])
        steady = float(durations.sum())
        report = {
            'status': 'PASS' if error is None and len(samples) == steps else 'INCOMPLETE',
            'error': error, 'mode': 'engineering_agent', 'physical_flies': 1,
            'neural_lanes': 2, 'unused_lane': 'zero input; computed by existing CPUBrain',
            'backend': backend, 'renderer': renderer, 'eye_rgb_shape': [2, 96, 96, 3],
            'viewer': False, 'audit': False, 'initial_sensory_sweep': True,
            'direct_stimulation': 'existing runner default after sweep',
            'steps': len(samples), 'warmup_steps': warmup, 'simulated_seconds': len(samples) * .01,
            'steady_loop_wall_seconds': steady,
            'real_time_factor': len(samples) * .01 / steady if steady else None,
            'loop_ms_median': float(np.median(durations) * 1000) if len(samples) else None,
            'loop_ms_p95': float(np.percentile(durations, 95) * 1000) if len(samples) else None,
            'component_seconds': {name: sum(row.get(name, 0.) for row in samples)
                                  for name in fields[2:]},
            'wall_seconds_including_setup_cleanup': wall, 'process_cpu_seconds': cpu,
            'process_cpu_core_equivalents': cpu / wall,
            'peak_process_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            'sampled_worker_peak_rss_mib': max(worker_rss_kib) / 1024 if worker_rss_kib else None,
            'worker_child_cpu_seconds': sum((resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime,
                                             resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime)),
            'gpu_memory_scope': 'device-wide sysfs samples; includes desktop, not per-process attribution',
            'vram_initial_bytes': initial_vram,
            'vram_peak_bytes': {path: max(row[path] for row in vram_samples) for path in initial_vram} if vram_samples else {},
            'queue_wait_seconds': None if backend == 'gpu' else 0,
            'queue_wait_source': 'attach gpu-wait started_at minus submitted_at after completion' if backend == 'gpu' else 'no queue used',
            'numpy_version': np.__version__,
            'cpu_model': next((line.split(':', 1)[1].strip() for line in
                              Path('/proc/cpuinfo').read_text().splitlines()
                              if line.startswith('model name')), 'unknown'),
            'hardware': platform.uname()._asdict(),
            'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'source_sha256': {str(path): digest(path) for path in source_paths},
            'graph_sha256': {name: digest(runner.ROOT / 'data' / name)
                             for name in ('row_ptr.bin', 'col_idx.bin', 'weights.bin')},
            'asset_sha256': {name: digest(runner.ASSETS / name)
                             for name in ('ports.npz', 'readout.npz')},
            'worker_binary_sha256': digest(runner.ROOT / '.work/neural-flies/neural_worker') if backend == 'gpu' else None,
            'scope': 'one-fly loop with eye rendering; viewer excluded; no multi-fly capacity claim',
        }
        (output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    if report['status'] != 'PASS':
        raise RuntimeError('incomplete cost sample')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--backend', choices=['cpu', 'gpu'], default='cpu')
    args = parser.parse_args()
    if args.steps < 2 or args.warmup < 0:
        parser.error('steps must be >=2 and warmup >=0')
    if args.backend == 'cpu' and os.environ.get('LIBGL_ALWAYS_SOFTWARE') != '1':
        parser.error('set LIBGL_ALWAYS_SOFTWARE=1 for this CPU-only baseline')
    print(json.dumps(measure(args.output, args.steps, args.warmup, args.backend)), flush=True)
