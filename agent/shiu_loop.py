"""Closed-loop odour choice on the Shiu brain: does the real wiring find food better than controls?

Arena: a map from agent.maps (tmaze, open, plume, maze): walls, food and decoy odour sources,
mirrored between paired columns. Every step the fly sniffs: the brain is reset and its ORNs
fire for 50 ms at 200 Hz x the map's odour concentration, scaled so the stronger odour fires at
200 Hz (adapted receptors: only the ratio carries position; no odour = silent). Resident GPU
worker, all arms and columns at once. The descending neurons' spike counts in the 30-50 ms
window give p_food = 0.5 + (cos to food centroid - cos to decoy centroid) / 2; centroids per
odour pair are fitted per arm from pure-odour sniffs at random concentrations, so each arm gets
its own best readout. The body is a run-and-tumble walker: 0.15 per step, a new random heading
unless p_food rose (a brain with no signal searches at random) or a wall blocks the step.
Starts, headings and tumble angles are common to all arms.

Metric per column: food bites minus decoy bites over 150 steps.
Inference: exact sign-flip over columns of real minus the mean of the 5 arms of a family.
"""
import csv
import json
import sys
import time

import numpy as np

from agent import maps, mcns, shiu_g1
from agent.brain_bench import ODORS6
from agent.neural_connectome import ANNOTATIONS, ROOT
from agent.paired_significance import _exact_sign_flip
from agent.shiu import ShiuLive, graph_dir, worker_binary
from agent.verify_shiu import _reference_graph

PAIRS = ((0, 1), (2, 3), (4, 5))  # (food, decoy) in ODORS6
STARTS, STEPS, TRAIN, HZ, STEP = 16, 150, 4, 200., .15
WINDOWS = (300, 500)  # 30 and 50 ms


def cells(dataset):
    """ORN indices per glomerulus and descending-neuron indices."""
    if dataset == 'mcns':
        _, _, _, _, (_, _, orn) = mcns.graph()
        return orn, mcns.descending()
    index, orn, dn = _reference_graph()[3], {}, []
    for r in csv.DictReader(open(ANNOTATIONS), delimiter='\t'):
        i = index.get(int(r['root_id']))
        if i is not None and r['cell_class'] == 'olfactory' and r['cell_type'].startswith('ORN_'):
            orn.setdefault(r['cell_type'][4:], []).append(i)
        if i is not None and r['super_class'] == 'descending':
            dn.append(i)
    return orn, dn


def combine(graphs, n):
    """Arm graphs side by side: arm a owns neurons a*n..; int32, a third of the int64 bytes."""
    return [np.concatenate([np.asarray(g[i], np.int32) + np.int32(a * n if i < 2 else 0) for a, g in enumerate(graphs)])
            for i in range(3)]


def prepare(dataset='flywire'):
    """CPU half of run: cached arms, the combined graph directory and the worker binary."""
    data = {'flywire': shiu_g1.flywire, 'mcns': mcns.graph}[dataset]()
    n, graphs = len(data[3]), [g for _, g, _ in shiu_g1.arms(data, dataset)]
    combined = combine(graphs, n)
    print(graph_dir(*combined, len(graphs) * n), worker_binary(len(graphs) * n, len(combined[0]), len(PAIRS) * STARTS), flush=True)


def run_arms(graphs, n, orn, dn, world, m):
    """All arms in one resident worker: their graphs side by side (arm a owns neurons a*n..),
    so one sniff simulates every arm for the same kernel launches. Flat column j = arm * cols + c."""
    arms = len(graphs)
    pair, sign, start, heading, tumble = (np.tile(x, (1, arms)) if x.ndim == 2 and x.shape[0] == STEPS
                                          else np.tile(x, (arms,) + (1,) * (x.ndim - 1)) for x in world)
    cols, flat = len(world[0]), len(pair)
    arm = np.repeat(np.arange(arms), cols)
    brain = ShiuLive(*combine(graphs, n), n=arms * n, flies=cols, seed=31)
    which = []  # per stim entry: (flat column, 0 food / 1 decoy)
    for c in range(cols):
        for a in range(arms):
            for k, o in enumerate(PAIRS[pair[c]]):
                ids = [a * n + i for g in ODORS6[o] for i in orn[g]]
                brain.stimulate(ids, HZ, fly=c)
                which += [(a * cols + c, k)] * len(ids)
    column, kind = np.array(which).T
    j = np.arange(flat)
    readout = ((j % cols) * arms * n + (j // cols) * n + np.asarray(dn)[:, None]).reshape(-1)  # (dn, flat)
    brain.start(WINDOWS, readout)
    dt = brain.dt

    def sniff(conc):  # conc (flat, 2) -> DN window counts (len(dn), flat)
        got = brain.sniff(HZ * dt * conc[column, kind])
        return (got[1] - got[0]).reshape(len(dn), flat).astype(float)

    rng = np.random.default_rng(7)
    centroid = np.zeros((arms, len(PAIRS), 2, len(dn)))
    for _ in range(TRAIN):  # pure-odour sniffs at random concentrations, readout fitted per arm
        label = rng.integers(0, 2, flat)
        conc = np.zeros((flat, 2))
        conc[j, label] = rng.uniform(.1, 1, flat)
        np.add.at(centroid, (arm, pair, label), sniff(conc).T)
    centroid /= np.linalg.norm(centroid, axis=3, keepdims=True) + 1e-9

    walls, sources = maps.mirrored(m, sign)
    pos, head = start.copy(), heading.copy()
    bites, dn_spikes, last = np.zeros((flat, 2)), [], np.full(flat, .5)
    for t in range(STEPS):
        conc = maps.concentration(m, pos, sources, sign)
        strongest = conc.max(axis=1, keepdims=True)
        x = sniff(np.where(strongest > 1e-3, conc / np.maximum(strongest, 1e-12), 0))  # adapted receptors; none = silent
        dn_spikes.append(x.sum(0))
        unit = x / (np.linalg.norm(x, axis=0, keepdims=True) + 1e-9)
        cos = [np.einsum('kd,dk->k', centroid[arm, pair, i], unit) for i in (0, 1)]
        p = np.where(x.sum(0) > 0, .5 + (cos[0] - cos[1]) / 2, .5)
        head = np.where(p <= last, head + tumble[t], head)  # no rise, incl. a silent brain: search
        last = p
        pos, blocked = maps.move(pos, head, STEP, walls)
        head = np.where(blocked, head + tumble[(t + 1) % STEPS], head)
        bites += maps.eat(pos, sources)
    brain.close()
    score, dn_spikes = bites[:, 0] - bites[:, 1], np.asarray(dn_spikes)
    return [{'score': score[a * cols:(a + 1) * cols].tolist(),
             'food_bites': bites[a * cols:(a + 1) * cols, 0].tolist(),
             'decoy_bites': bites[a * cols:(a + 1) * cols, 1].tolist(),
             'dn_spikes_per_sniff': float(dn_spikes[:, a * cols:(a + 1) * cols].mean())} for a in range(arms)]


def run(dataset='flywire', map_name='tmaze'):
    data = {'flywire': shiu_g1.flywire, 'mcns': mcns.graph}[dataset]()
    n = len(data[3])
    orn, dn = cells(dataset)
    m = maps.build(map_name)
    rng = np.random.default_rng(20261003)
    cols = len(PAIRS) * STARTS
    sign = np.where(np.arange(cols) % 2, 1., -1.)
    start, heading = maps.starts(m, sign, rng)
    world = (np.repeat(np.arange(len(PAIRS)), STARTS), sign, start, heading, rng.uniform(-np.pi, np.pi, (STEPS, cols)))
    started = time.perf_counter()
    names, graphs = zip(*[(name, graph) for name, graph, _ in shiu_g1.arms(data, dataset)])
    results = dict(zip(names, run_arms(list(graphs), n, orn, dn, world, m)))
    for name, r in results.items():
        print(f"{name:20s} DN/sniff {r['dn_spikes_per_sniff']:7.1f}  food {sum(r['food_bites']):5.0f}  "
              f"decoy {sum(r['decoy_bites']):5.0f}  score {np.mean(r['score']):+.2f}", flush=True)
    tests = {}
    for family in ('shuffled', 'block'):
        real = np.asarray(results['real']['score'])
        control = np.mean([results[f'{family}-s{s}']['score'] for s in shiu_g1.SEEDS], axis=0)
        p, nonzero = _exact_sign_flip(list(real - control), scale=10000)
        tests[family] = {'real_mean': float(real.mean()), 'control_mean': float(control.mean()),
                         'columns_real_higher': int((real > control).sum()), 'exact_two_sided_sign_flip_p': p}
        print(f"{family:8s} real {real.mean():+.3f} control {control.mean():+.3f} "
              f"real>control {(real > control).sum()}/{cols} p={p:.4g}", flush=True)
    out = {'dataset': dataset, 'map': map_name, 'pairs': [[list(ODORS6[a]), list(ODORS6[b])] for a, b in PAIRS],
           'columns': cols, 'steps': STEPS, 'arms': results, 'tests': tests, 'wall_s': time.perf_counter() - started}
    (ROOT / f'.work/agent/shiu/loop-{dataset}-{map_name}.json').write_text(json.dumps(out, indent=2) + '\n')
    return out


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if a != '--prepare']
    prepare(*args[:1]) if '--prepare' in sys.argv else run(*args[:2])  # [flywire|mcns] [tmaze|open|plume|maze]
