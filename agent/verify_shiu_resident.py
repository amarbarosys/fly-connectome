"""shiu_resident.mojo (the resident brain on a caller-owned DeviceContext, live.mojo -DBRAIN=3) against ShiuLive.advance, bit for bit.

  gpu-wait run --shared --vram 4 --priority 60 --timeout 1200 --env PYTHONPATH=$PWD -- .venv/bin/python -m agent.verify_shiu_resident

The fly_link brain (real FlyWire wiring, LC4 + LPLC2 looming drive at 150 Hz on both eyes, seed 71, one column) runs 100 + 100 ticks twice: in the resident
worker process (ShiuLive.advance twice, then snapshot()) and in shiu_resident_check.mojo (Resident.advance twice, one DeviceContext, no worker). The per-neuron spike
counts must be identical (and the brain must have spiked: DN counts > 0). Membrane voltages are not bit-exact even between two runs of the worker itself (deliver
adds float32 with atomics: order varies, 182 of 138,639 voltages differ by up to 2.9e-6 mV), so the bar is that run-to-run floor: the worker is run twice, and the resident
brain's voltages must lie within max(1e-5, 4 x floor) of the first. check.sh shiures adds the mutant (a tick counter that skips ticks).
"""
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np

from agent import shiu_g1
from agent.experiment import cells
from agent.neural_connectome import ROOT
from agent.shiu import ShiuLive

A, B, SEED = 100, 100, 71
HZ = 150.0
SRC = [ROOT / 'agent' / f for f in ('shiu_resident_check.mojo', 'shiu_resident.mojo', 'shiu_worker.mojo')]


def binary(n, edges):
    h = hashlib.sha1(b''.join(f.read_bytes() for f in SRC) + f'{n}{edges}'.encode()).hexdigest()[:8]  # content-addressed: mutant-check restores old mtimes
    b = ROOT / f'.work/agent/shiu/build/shiu_resident_check_{n}_{edges}_{h}'
    if not b.is_file():
        b.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(Path.home() / '.local/bin/mojo'), 'build', '-I', str(ROOT / 'agent'), f'-DSHIU_N={n}', f'-DSHIU_E={edges}', '-DSHIU_C=1',
                        str(SRC[0]), '-o', str(b)], check=True, capture_output=True, cwd=ROOT)
    return b


def brain():
    c = cells()
    data = shiu_g1.flywire()
    graph = next(shiu_g1.arms(data, 'flywire'))[1]
    n = len(data[3])
    eye = {s: np.flatnonzero(np.isin(c.type, ('LC4', 'LPLC2')) & (c.side == s)) for s in ('left', 'right')}
    dn = np.flatnonzero(c.super == 'descending')
    b = ShiuLive(*graph, n=n, flies=1, seed=SEED)
    for side in ('left', 'right'):
        b.stimulate(eye[side], HZ, fly=0, key=0)
    return b, n, dn


def worker():
    """One ShiuLive run of A + B ticks: (brain, snapshot, n, dn)."""
    b, n, dn = brain()
    b.start([1], dn.astype(np.int32))
    try:
        b.advance(A)
        b.advance(B)
        return b, b.snapshot(), n, dn
    finally:
        b.close()


def check():
    b, ref, n, dn = worker()
    _, ref2, _, _ = worker()
    exe = binary(n, len(b.col))
    subprocess.run([str(exe), str(b.graph) + '/', str(b.work) + '/', str(SEED), str(b.refractory), str(A), str(B)], check=True, capture_output=True)
    counts = np.fromfile(b.work / 'resident_counts.bin', np.int32)
    u = np.fromfile(b.work / 'resident_u.bin', np.float32)
    same_counts = np.array_equal(counts, ref['counts'])
    floor = float(np.abs(ref['u'] - ref2['u']).max())
    err = float(np.abs(u - ref['u']).max())
    same_u = err <= max(1e-5, 4 * floor)
    spiked = int(counts[dn].sum())
    ok = same_counts and same_u and spiked > 0
    print(('PASS' if ok else 'FAIL'), f'resident brain on a caller-owned context == ShiuLive.advance({A}) + advance({B}): counts equal {same_counts}, membrane voltages within {err:.1e} of the worker\'s (worker vs worker floor {floor:.1e}, bar max(1e-5, 4 x floor)); '
          f'{int(counts.sum())} spikes in the brain, {spiked} in the descending neurons')
    if not ok:
        sys.exit('FAIL verify_shiu_resident')


if __name__ == '__main__':
    check()
