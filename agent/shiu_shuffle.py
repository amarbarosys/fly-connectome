"""Degree-preserving shuffled connectome: the control for whether the wiring, rather than the
degrees and signs, carries an effect.

Port of shuffle_edges from Kisame76/drosophila-brain-mlx (src/lif/shuffle_pack.py, MIT), on
edge lists instead of a pack. Every edge keeps its presynaptic cell and its signed synapse count,
so each cell keeps its out-degree, its sign and its total output. Postsynaptic cells are
permuted across all edges, so each cell keeps its in-degree. A (pre, post) pair that lands twice
is swapped with a random edge's post until none repeats. Who reaches whom is random, and with it
each cell's incoming weight and excitatory/inhibitory input mix.

    python -m agent.shiu_shuffle --prepare   # CPU: cache shuffles, build worker
    python -m agent.shiu_shuffle             # real graph + 5 shuffles, GPU, writes .work/agent/shiu/shuffle-control.json
"""
from __future__ import annotations

import csv
import json
import time

import numpy as np

from agent.shiu import ROOT, ShiuGPU

MN9 = 720575940660219265  # proboscis motor neuron read out by the authors' example notebook


def shuffle_edges(pre, post, seed, max_rounds=1000):
    """New post for each edge (same order as the input) and stats: repair_rounds, edges_kept, self_loops."""
    pre, post = np.asarray(pre, np.int64), np.asarray(post, np.int64)
    e = post.size
    if pre.size != e:
        raise ValueError(f'{pre.size} pre for {e} post')
    n = int(max(pre.max(), post.max())) + 1 if e else 0
    real = pre * n + post
    if np.unique(real).size != e:
        raise ValueError('input repeats a (pre, post) pair; merge parallel edges first')
    rng = np.random.default_rng(seed)
    dst = post[rng.permutation(e)]
    for rounds in range(max_rounds + 1):
        flat = pre * n + dst
        order = np.argsort(flat, kind='stable')
        key = flat[order]
        repeat = order[1:][key[1:] == key[:-1]]
        if repeat.size == 0:
            break
        if rounds == max_rounds:
            raise ValueError(f'{repeat.size} parallel edges left after {max_rounds} repair rounds')
        # Swap each repeat's post with a random edge's; no edge takes part in two swaps, so
        # together they permute the posts and in-degrees stay.
        partner = rng.integers(0, e, repeat.size)
        free = ~np.isin(partner, repeat)
        partner, first = np.unique(partner[free], return_index=True)
        repeat = repeat[free][first]
        swap = np.arange(e)
        swap[repeat], swap[partner] = partner, repeat
        dst = dst[swap]
    stats = {'repair_rounds': rounds, 'edges_kept': int(np.isin(pre * n + dst, real).sum()),
             'self_loops': int((pre == dst).sum())}
    return dst, stats


def cached_shuffle(pre, post, seed):
    """shuffle_edges of the reference graph, kept in .work so a GPU job never pays the 22 s."""
    path = ROOT / f'.work/agent/shiu/shuffles/{len(post)}-seed{seed}.npz'
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        dst, stats = shuffle_edges(pre, post, seed)
        np.savez(path, post=dst.astype(np.int32), **stats)
    z = np.load(path)
    return z['post'], {k: int(z[k]) for k in ('repair_rounds', 'edges_kept', 'self_loops')}


def prepare(seeds=range(5), flies=4):
    """CPU half of shuffle_control: cache the shuffles and build the GPU worker."""
    from agent.shiu import worker_binary
    from agent.verify_shiu import _reference_graph
    pre, post, _, index = _reference_graph()
    for seed in seeds:
        print(seed, cached_shuffle(pre, post, seed)[1], flush=True)
    worker_binary(len(index), len(post), flies)


def shuffle_control(seeds=range(5), seconds=.2, sugar_hz=100., orn_hz=100., engine=ShiuGPU):
    """Real graph vs degree-preserving shuffles: sugar GRNs -> MN9 (the source repo's check) and
    the mushroom-body probe (KC/MBON/APL activity for the three verify_sparse_learning odours)."""
    import re
    from agent.neural_connectome import ANNOTATIONS
    from agent.verify_shiu import REF, _reference_graph
    from agent.verify_sparse_learning import ODORS
    pre, post, w, index = _reference_graph()
    n = len(index)
    source = (REF / 'code/benchmark.py').read_text()
    sugar = [index[int(x)] for x in re.findall(r'7205\d{14}', source[source.index("'sugar'"):source.index("'p9'")])]
    mn9 = index[MN9]
    cells, orn = {'KC': [], 'MBON': [], 'APL': []}, {}
    for row in csv.DictReader(open(ANNOTATIONS), delimiter='\t'):
        i = index.get(int(row['root_id']))
        if i is None:
            continue
        name = {'Kenyon_Cell': 'KC', 'MBON': 'MBON'}.get(row['cell_class']) or ('APL' if row['cell_type'] == 'APL' else None)
        if name:
            cells[name].append(i)
        if row['cell_class'] == 'olfactory' and row['cell_type'].startswith('ORN_'):
            orn.setdefault(row['cell_type'][4:], []).append(i)
    steps = round(seconds / 1e-4)
    rows = []
    for seed in [None, *seeds]:
        started = time.perf_counter()
        arm_post, stats = (post, None) if seed is None else cached_shuffle(pre, post, seed)
        # column 0: sugar GRNs; columns 1..3: one odour each
        brain = engine(pre, arm_post, w, n=n, flies=1 + len(ODORS), seed=3)
        brain.stimulate(sugar, sugar_hz, fly=0)
        for column, odor in enumerate(ODORS, 1):
            brain.stimulate(sum((orn[g] for g in odor), []), orn_hz, fly=column)
        counts = brain.run([steps])[0]
        ever = counts > 0
        rows.append({'graph': 'real' if seed is None else f'shuffle-{seed}', 'shuffle': stats,
                     'mn9_hz': float(counts[mn9, 0] / seconds),
                     'sugar_grn_hz': float(counts[sugar, 0].mean() / seconds),
                     'sugar_brain_active': int(ever[:, 0].sum()),
                     'orn_hz': [float(counts[sum((orn[g] for g in o), []), c].mean() / seconds) for c, o in enumerate(ODORS, 1)],
                     'kc_active': ever[cells['KC'], 1:].mean(axis=0).tolist(),
                     'mbon_active': ever[cells['MBON'], 1:].sum(axis=0).tolist(),
                     'apl_hz': (counts[cells['APL'], 1:].sum(axis=0) / seconds).tolist(),
                     'brain_active': ever[:, 1:].mean(axis=0).tolist(),
                     'wall_s': time.perf_counter() - started})
        print(json.dumps({k: rows[-1][k] for k in ('graph', 'mn9_hz', 'sugar_grn_hz', 'kc_active', 'wall_s')}), flush=True)
    out = {'seconds': seconds, 'sugar_hz': sugar_hz, 'orn_hz': orn_hz, 'n_sugar': len(sugar), 'rows': rows}
    path = ROOT / '.work/agent/shiu/shuffle-control.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + '\n')
    return out


if __name__ == '__main__':
    import sys
    prepare() if '--prepare' in sys.argv else shuffle_control()
