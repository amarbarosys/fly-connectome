"""FLY_TAP (agent/shiu_worker.mojo) checked: the live brain view's activity tap changes nothing and shows the truth.
Two columns of the FlyWire brain, column 0 driven (LC4 + LPLC2 both eyes, 150 Hz), column 1 silent; 30 advances of
1 ms, once with the tap off and once on (every advance). Gates:
  off == on      FLY_RECORD spike counts byte-identical (the tap only reads)
  spiked -> lit  every neuron that spiked in the last advance has glow >= 242 (age <= 10 ticks)
  silent -> 0    every neuron that never spiked has glow 0
  column         path.sel = 1 switches the tap to column 1 (silent: all 0)
  cost           ms per advance with the tap vs without

  .venv/bin/python -m agent.body_engine.brain_tap_check     (builds the worker on the CPU, then a short gpu-wait job)
"""
import os
import subprocess
import sys
import time

import numpy as np

from agent.neural_connectome import ROOT

OUT = ROOT / '.work/agent/body_engine/brain-tap'
STEPS, K = 30, 10


def brain():
    from agent import shiu_g1
    from agent.body_engine.play import resolve
    from agent.experiment import cells
    from agent.shiu import ShiuLive
    data = shiu_g1.flywire()
    graph = next(shiu_g1.arms(data, 'flywire'))[1]
    n = len(data[3])
    b = ShiuLive(*graph, n=n, flies=2, seed=71)
    c = cells()
    b.stimulate(np.concatenate([resolve(t, c) for t in ('LC4', 'LPLC2')]), 150.0, fly=0, key=0)
    return b, n


def run_once(tap):
    b, n = brain()
    rec = OUT / f'rec_{"on" if tap else "off"}.bin'
    rec.unlink(missing_ok=True)
    os.environ.pop('FLY_TAP', None)
    if tap:
        os.environ['FLY_TAP'], os.environ['FLY_TAP_EVERY'] = str(OUT / 'brain.tap'), '1'
        (OUT / 'brain.tap.sel').unlink(missing_ok=True)
    b.start([1], np.array([0], np.int32), record=rec)
    t0 = time.perf_counter()
    taps = []
    for k in range(STEPS):
        b.advance(K)
        if tap and k == STEPS - 2:
            taps.append((OUT / 'brain.tap').read_bytes())
            np.array([1], np.int32).tofile(OUT / 'brain.tap.sel')
    dt = (time.perf_counter() - t0) / STEPS * 1000
    if tap:
        taps.append((OUT / 'brain.tap').read_bytes())
    b.close()
    return rec, n, dt, taps


def main():
    if '--run' not in sys.argv:
        from agent import shiu_g1
        from agent.shiu import worker_binary
        data = shiu_g1.flywire()
        graph = next(shiu_g1.arms(data, 'flywire'))[1]
        print('worker built on the CPU:', worker_binary(len(data[3]), len(graph[1]), 2).name)
        sys.exit(subprocess.run(['gpu-wait', 'run', '--priority', '90', '--vram', '4', '--timeout', '600', '--', sys.executable, '-m',
                                 'agent.body_engine.brain_tap_check', '--run'], cwd=ROOT).returncode)
    OUT.mkdir(parents=True, exist_ok=True)
    rec_off, n, ms_off, _ = run_once(False)
    rec_on, _, ms_on, taps = run_once(True)
    counts = np.fromfile(rec_on, np.int32).reshape(STEPS, 2, n)
    gates = {}
    gates['off == on'] = rec_off.read_bytes() == rec_on.read_bytes()
    head = np.frombuffer(taps[0][:16], np.int32)
    glow = np.frombuffer(taps[0][16:], np.uint8)
    spiked = counts[STEPS - 2, 0] > counts[STEPS - 3, 0]
    never = counts[STEPS - 2, 0] == 0
    print(f'tap header seq {head[0]} column {head[1]} tick {head[2]} N {head[3]}; spiked in the last ms {int(spiked.sum())}, '
          f'their glow min {int(glow[spiked].min()) if spiked.any() else -1}; never spiked {int(never.sum())}, glow max {int(glow[never].max())}; '
          f'lit neurons {int((glow > 0).sum())}')
    gates['spiked -> lit'] = bool(spiked.any()) and int(glow[spiked].min()) >= 242 and head[1] == 0 and head[3] == n
    gates['silent -> 0'] = int(glow[never].max()) == 0
    h1 = np.frombuffer(taps[1][:16], np.int32)
    gates['column'] = h1[1] == 1 and counts[-1, 1].sum() == 0 and not np.frombuffer(taps[1][16:], np.uint8).any()
    print(f'ms per 1 ms advance: tap off {ms_off:.2f}, tap on every advance {ms_on:.2f}')
    for k, v in gates.items():
        print(('PASS' if v else 'FAIL'), k)
    bad = [k for k, v in gates.items() if not v]
    if bad:
        sys.exit(f'FAIL {len(bad)}/{len(gates)} tap gates: {bad}')
    print('PASS brain_tap_check')


if __name__ == '__main__':
    main()
