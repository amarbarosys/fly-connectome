"""G1 on the Shiu model: does the real wiring carry more odour identity than matched controls?

The authors' 783 graph (real) against 5 degree-preserving shuffles and 5 block-stratified
shuffles (same generators and seeds as P3, baseline_graphs). Each arm runs flood_code_probe on
the GPU: 6 odours x 3 ORN-jitter columns, 100 Hz, 0.2 s. Per column, odour distinctiveness is
mean similarity to same-odour columns minus to other-odour columns, for the KC sets at each
onset window and for KC rates in the flood. Inference: exact two-sided sign-flip over the 18
columns of real minus the mean of the 5 arms of a family (columns share one graph, so they are
input-jitter samples, not independent animals).
"""
import csv
import hashlib
import json
import time

import numpy as np

from agent.baseline_graphs import block_keys, block_shuffle, repair_edges
from agent.brain_bench import ODORS6
from agent import mcns
from agent.neural_connectome import ANNOTATIONS, ROOT
from agent.paired_significance import _exact_sign_flip
from agent.shiu import ShiuGPU
from agent.verify_shiu import _reference_graph, flood_code_probe

SEEDS = (20261002, 7, 1009, 424242, 31337)
METRICS = ('onset_20ms', 'onset_30ms', 'onset_50ms', 'flood_rate')


def flywire():
    """The authors' 783 graph, classes as cell_class (super_class when empty); cells None = probe default."""
    pre, post, w, index = _reference_graph()
    names = np.full(len(index), 'unannotated', dtype=object)
    for r in csv.DictReader(open(ANNOTATIONS), delimiter='\t'):
        i = index.get(int(r['root_id']))
        if i is not None:
            names[i] = r['cell_class'] or r['super_class'] or 'unannotated'
    return pre, post, w, names, None


def cached_arm(dataset, family, seed, pre, row, post, klass, key):
    """The arm's post column and its checks, built and checked once on the CPU and kept in .work,
    so a GPU job only simulates (rechecking 10 cached arms cost 77 s)."""
    code = hashlib.sha256((ROOT / 'agent/baseline_graphs.py').read_bytes()).hexdigest()[:10]  # a new repair rule never reads an old arm
    path = ROOT / f'.work/agent/shiu/arms/{dataset}-{len(post)}-{family}-s{seed}-{code}-checked.npz'
    if not path.is_file():
        n = len(row) - 1
        generator = np.random.default_rng(seed)
        if family == 'shuffled':
            col, passes = repair_edges(row, generator.permutation(post), generator, unique=True)
        else:
            col, passes = repair_edges(row, block_shuffle(row, post, klass, generator), generator,
                                       key=key, original=post, unique=True)
        checks = {'out_degree_preserved': True,  # rows (presynaptic slots) are untouched
                  'in_degree_preserved': bool(np.array_equal(np.bincount(col, minlength=n), np.bincount(post, minlength=n))),
                  'self_loops': int((pre == col).sum()), 'repair_passes': int(passes),
                  'edges_unchanged': int((col == post).sum()),
                  'repeated_pairs': int(len(col) - np.unique(pre * n + col).size)}
        if family == 'block':
            checks['class_blocks_preserved'] = bool(np.array_equal(np.sort(block_keys(row, col, klass)), np.sort(key)))
        assert checks['in_degree_preserved'] and checks['self_loops'] == 0 and checks['repeated_pairs'] == 0, checks
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, col=col.astype(np.int32), checks=json.dumps(checks))
    z = np.load(path)
    return z['col'], json.loads(str(z['checks']))  # int32 as stored: 60 MB per arm, not 120


def arms(data, dataset='flywire'):
    pre, post, w, names, _ = data
    order = np.argsort(pre, kind='stable')
    pre, post, w = pre[order], post[order].astype(np.int64), w[order]
    n = len(names)
    row = np.zeros(n + 1, np.int64)
    np.cumsum(np.bincount(pre, minlength=n), out=row[1:])
    klass = np.unique(names, return_inverse=True)[1].astype(np.int64)
    key = block_keys(row, post, klass)
    yield 'real', (pre, post, w), {}
    for seed in SEEDS:
        for family in ('shuffled', 'block'):
            col, checks = cached_arm(dataset, family, seed, pre, row, post, klass, key)
            yield f'{family}-s{seed}', (pre, col, w), checks


def run(dataset='flywire'):
    data = {'flywire': flywire, 'mcns': mcns.graph}[dataset]()
    started, results, checks = time.perf_counter(), {}, {}
    for name, graph, arm_checks in arms(data, dataset):
        t = time.perf_counter()
        probe = flood_code_probe(engine=ShiuGPU, arm=graph, odours=ODORS6, cells=data[4])
        results[name] = {'per_column': probe['per_column'], 'flood_decode': probe['flood']['decode_accuracy'],
                         'onset': [{k: r[k] for k in ('ms', 'kc_active', 'decode_accuracy')} for r in probe['onset']],
                         'kc_active_flood': probe['flood']['kc_active'], 'seconds': time.perf_counter() - t}
        checks[name] = arm_checks
        print(name, {m: round(float(np.mean(probe['per_column'][m])), 4) for m in METRICS},
              'decode', probe['flood']['decode_accuracy'], f"{results[name]['seconds']:.1f}s", flush=True)
    tests = {}
    for family in ('shuffled', 'block'):
        names = [f'{family}-s{s}' for s in SEEDS]
        for metric in METRICS:
            real = np.asarray(results['real']['per_column'][metric])
            control = np.mean([results[n]['per_column'][metric] for n in names], axis=0)
            p, nonzero = _exact_sign_flip(list(real - control), scale=10000)
            tests[f'{family}/{metric}'] = {'real_mean': float(real.mean()), 'control_mean': float(control.mean()),
                                           'difference': float((real - control).mean()),
                                           'columns_real_higher': int((real > control).sum()),
                                           'exact_two_sided_sign_flip_p': p, 'nonzero_columns': nonzero}
    out = {'model': 'Shiu 2024 LIF', 'dataset': {'flywire': 'FlyWire 783 (authors graph)', 'mcns': mcns.RELEASE}[dataset], 'odours': [list(o) for o in ODORS6], 'columns': 18,
           'seeds': SEEDS, 'arms': results, 'arm_checks': checks, 'tests': tests,
           'wall_s': time.perf_counter() - started}
    path = ROOT / ('.work/agent/shiu/g1.json' if dataset == 'flywire' else f'.work/agent/shiu/g1-{dataset}.json')
    path.write_text(json.dumps(out, indent=2) + '\n')
    for name, test in tests.items():
        print(f"{name:22s} real {test['real_mean']:.4f} control {test['control_mean']:.4f} "
              f"diff {test['difference']:+.4f} real>control {test['columns_real_higher']}/18 p={test['exact_two_sided_sign_flip_p']:.4g}")
    return out


def prepare(dataset='flywire'):
    """CPU half of run: build and check every arm into the cache."""
    data = {'flywire': flywire, 'mcns': mcns.graph}[dataset]()
    for name, _, arm_checks in arms(data, dataset):
        print(dataset, name, arm_checks, flush=True)


if __name__ == '__main__':
    import sys
    args = [a for a in sys.argv[1:] if a != '--prepare']
    (prepare if '--prepare' in sys.argv else run)(*args[:1])
