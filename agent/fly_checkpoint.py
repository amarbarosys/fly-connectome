"""Fly checkpoint: one learned fly as a file that rebuilds its own kernel and re-checks its own behaviour.

A fly is one column of the learning setup (agent/learning_loop.py: graph, edge groups, odours): its
per-KC factors are all that learning changes. The file (.npz) holds those factors, the GPU kernel
source that produced its numbers (agent/shiu_worker.mojo, which imports only the Mojo standard
library), the commit, the dataset licence, hashes of the graph and edge groups, and a probe: counts
of KCs and labelled MBONs at 30, 50, 100 and 200 ms for a sniff of odour A, then of odour B, from a
fresh worker loaded with the factors. The harness (setup, odours, this file) is not embedded: it is
what judges the fly, so it comes from git at the stored commit (the self-describing-model rule).

check: extract the kernel into an empty directory, build the worker from it alone, rebuild the graph
with the harness and require its hashes to match, rerun the probe and require the counts to match the
stored ones within the worker's float noise (1e-3 of spikes, 1e-3 of cells; atomics add in any order).

usage: python -m agent.fly_checkpoint train DIR     train a learner and its twin, pack both
       python -m agent.fly_checkpoint check FILE..  rebuild from each file, PASS/FAIL per file
"""
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from agent.learning_loop import HZ, LN, GAIN, SEED, brain, setup
from agent.neural_connectome import ROOT
from agent.shiu import ShiuLive

KERNEL = 'agent/shiu_worker.mojo'
LICENCE = 'FlyWire connectome (Dorkenwald et al. 2024), CC BY-NC 4.0: non-commercial use only'
WINDOWS = [300, 500, 1000, 2000]


def sha(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(memoryview(np.ascontiguousarray(a)).cast('B'))
    return h.hexdigest()[:16]


def git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def probe(world, factor, source=()):
    """Fresh one-column worker with these factors: sniff odour A, then B; counts (2, windows, readout)."""
    pre, post, w, n, grp, kc, approach, avoid, odour = world
    b = brain(ShiuLive, pre, post, w, n, grp, kc, odour, 1)
    b.source = source
    b.factor = np.asarray(factor, np.float32)[None]
    b.start(WINDOWS, np.concatenate([kc, approach, avoid]))
    entries, p = b._stim_arrays()
    out = np.stack([b.sniff(np.where(np.isin(entries % n, ids), p, 0).astype(np.float32)) for ids in odour])
    b.close()
    return out


def valence(counts, world):
    """(approach - avoid) / (approach + avoid) per odour at 200 ms."""
    k, a = len(world[5]), len(world[6])
    ap, av = counts[:, -1, k:k + a].sum(1), counts[:, -1, k + a:].sum(1)
    return ((ap - av) / np.maximum(ap + av, 1)).round(3).tolist()


def pack(path, world, factor, label):
    if git('status', '--porcelain', '--untracked-files=no', 'agent'):
        raise SystemExit('FAIL pack: agent/ has uncommitted changes; commit first, bake second')
    pre, post, w, n, grp, kc = world[:6]
    counts = probe(world, factor)
    meta = {'label': label, 'commit': git('rev-parse', 'HEAD'), 'licence': LICENCE,
            'graph': sha(pre, post, w), 'groups': sha(grp),
            'setup': {'LN': LN, 'GAIN': GAIN, 'HZ': HZ, 'SEED': SEED, 'windows': WINDOWS},
            'kernel': {KERNEL: (ROOT / KERNEL).read_text()}, 'valence_A_B_200ms': valence(counts, world)}
    np.savez_compressed(path, meta=json.dumps(meta), factor=np.asarray(factor, np.float32), kcs=kc.astype(np.int32), probe=counts)
    print(f"packed {path} {label} valence A,B {meta['valence_A_B_200ms']} {Path(path).stat().st_size / 2**20:.2f} MB", flush=True)


def check(path, world):
    f = np.load(path)
    meta = json.loads(str(f['meta']))
    pre, post, w, n, grp, kc = world[:6]
    fails = [name for name, ok in (('graph hash', meta['graph'] == sha(pre, post, w)), ('edge-group hash', meta['groups'] == sha(grp)),
                                   ('KC list', np.array_equal(f['kcs'], kc))) if not ok]
    if fails:
        return False, {'fail': fails}
    with tempfile.TemporaryDirectory(dir=ROOT / '.work') as tmp:  # the kernel comes from the file alone
        for name, text in meta['kernel'].items():
            (Path(tmp) / Path(name).name).write_text(text)
        counts = probe(world, f['factor'], (Path(tmp) / Path(KERNEL).name,))
    ref = f['probe']
    cells = counts.shape[-1] * 2
    differing = int((counts != ref).any(axis=1).sum())
    spikes, spikes_ref = int(counts.sum()), int(ref.sum())
    ok = abs(spikes - spikes_ref) <= max(1e-3 * spikes_ref, 2) and differing <= max(1e-3 * cells, 1)
    return ok, {'label': meta['label'], 'commit': meta['commit'][:7], 'spikes': [spikes, spikes_ref], 'cells_differing': differing,
                'valence_A_B_200ms': [valence(counts, world), meta['valence_A_B_200ms']]}


def train(out):
    """Learner and twin as in verify_plastic: sniff A, dopamine 1 for the learner, 0 for the twin."""
    world = setup()
    pre, post, w, n, grp, kc, approach, avoid, odour = world
    b = brain(ShiuLive, pre, post, w, n, grp, kc, odour, 2)
    b.start(WINDOWS, np.asarray(kc[:1]))
    entries, p = b._stim_arrays()
    b.sniff(np.where(np.isin(entries % n, odour[0]), p, 0).astype(np.float32))
    b.reward(np.float32([1, 0]))
    factor = b.factors()
    b.close()
    Path(out).mkdir(parents=True, exist_ok=True)
    for col, label in enumerate(('learner', 'twin')):
        pack(Path(out) / f"{label}-{git('rev-parse', '--short', 'HEAD')}.npz", world, factor[col], label)


def main():
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == 'train':
        return train(args[0])
    world, bad = setup(), []
    for path in args:
        ok, info = check(path, world)
        print('PASS' if ok else 'FAIL', path, json.dumps(info), flush=True)
        bad += [] if ok else [path]
    if bad:
        print(f'FAIL {len(bad)}/{len(args)} {" ".join(bad)}')
        sys.exit(1)


if __name__ == '__main__':
    main()
