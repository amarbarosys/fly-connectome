"""Is mushroom-body reward learning specific, reversible, and owed to the real wiring? (Astra review 2026-10-06, steps 2-3.)

Same brain as agent/verify_plastic.py (learning_loop.setup: ALLN x LN, KC -> MBON x GAIN, KC -> avoid-MBON plastic), six
odours (brain_bench.ODORS6). Per seed, FLIES columns; the first half are learners (dopamine 1), the rest twins (0).
Schedule: test every odour; sniff A, reward; test every odour; sniff B, reward (reversal); test A and B. Readout per
column over 200 ms: approach- and avoid-MBON spikes, valence (ap - av) / (ap + av), and DN spikes (step 3: does the
change reach descending neurons through the existing MBON path).

Arms: real wiring; degree-preserving shuffles (pre permuted globally: in- and out-degree exact, self-loops repaired);
block shuffles (pre permuted within each (pre class, post class) block: class-pair edge counts and both degrees exact);
random (pre drawn uniformly); causal cuts of the real graph: cut:kc_mbon (KC -> MBON edges removed), cut:mbon_out (every
MBON output edge removed). Edge groups (ALLN inhibition, KC -> approach/avoid) are recomputed on each arm's edges.
A and the held-out odours are also tested at 0.5x and 2x the odour rate (intensity generalisation).

    python -m agent.learning_specificity noise SEEDS               # no reward anywhere: learner-twin gap = column noise
    python -m agent.learning_specificity run ARM FIRST [SEEDS]     # ARM real | shuffle:K | block:K | random:K | cut:kc_mbon | cut:mbon_out
Writes .work/agent/shiu/specificity/<mode>-<arm>-<first>.json, one record per seed, per-column arrays.
"""
import json
import subprocess
import sys
import time

import numpy as np

from agent.brain_bench import ODORS6
from agent.experiment import cells
from agent.learning_loop import GAIN, HZ, LN
from agent.mb_valence import mbons
from agent.neural_connectome import ROOT
from agent.shiu import ShiuLive, graph_dir
from agent.verify_shiu import _reference_graph

OUT = ROOT / '.work/agent/shiu/specificity'
FLIES, WINDOW = 16, 2000  # columns per seed; 200 ms at dt 1e-4
A, B = 1, 0  # verify_plastic's food and decoy; ODORS6[2:] are never rewarded (held out)


INTENSITIES = (0.5, 2.0)


def arm_edges(pre, post, arm, names):
    if arm == 'real' or arm.startswith('cut:'): return pre
    kind, k = arm.split(':')
    if kind not in ('shuffle', 'block', 'random'): raise SystemExit(f'FAIL arm {arm!r}')
    rng = np.random.default_rng(1000 + int(k))
    n = int(max(pre.max(), post.max())) + 1
    if kind == 'shuffle': new = rng.permutation(pre)
    elif kind == 'random': new = rng.integers(0, n, len(pre)).astype(pre.dtype)
    else:  # permute pre inside each (pre class, post class) block
        _, cls = np.unique(names, return_inverse=True)
        block = cls[pre].astype(np.int64) * (cls.max() + 1) + cls[post]
        order = np.argsort(block, kind='stable')
        bounds = np.flatnonzero(np.diff(block[order])) + 1
        new = pre.copy()
        for idx in np.split(order, bounds): new[idx] = pre[idx][rng.permutation(len(idx))]
    # Self-loops: swap each, one at a time, with a random edge of the same block (any edge for shuffle/random) whose swap
    # makes no new loop. One-at-a-time keeps the pre multiset exact; vectorised swaps hit one partner twice and lost values.
    pool = (lambda i: np.flatnonzero(block == block[i])) if kind == 'block' else (lambda i: None)
    for i in np.flatnonzero(new == post):
        cand = pool(i)
        for _ in range(100):
            j = rng.choice(cand) if cand is not None else rng.integers(0, len(new))
            if new[j] != post[i] and new[i] != post[j]:
                new[i], new[j] = new[j], new[i]
                break
    if (new == post).any(): raise SystemExit(f'FAIL {arm}: self-loops left after repair')
    return new


def setup(arm):
    pre, post, w, _ = _reference_graph()
    pre, post, w = np.asarray(pre), np.asarray(post), np.asarray(w)
    c = cells()
    names, n = c.klass, len(c.klass)
    approach, avoid = mbons(c)
    kc = np.flatnonzero(names == 'Kenyon_Cell')
    pre = arm_edges(pre, post, arm, names)
    keep = {'cut:kc_mbon': ~((names[pre] == 'Kenyon_Cell') & (names[post] == 'MBON')),
            'cut:mbon_out': names[pre] != 'MBON'}.get(arm)
    if keep is not None: pre, post, w = pre[keep], post[keep], w[keep]
    order = np.argsort(pre, kind='stable')
    pre, post, w = pre[order], post[order], w[order]
    dn = np.flatnonzero(names == 'descending')
    graph_dir(pre, post, w, n)
    is_kc, is_app, is_avo = (np.isin(np.arange(n), x) for x in (kc, approach, avoid))
    grp = np.zeros(len(pre), np.uint8)
    grp[(names[pre] == 'ALLN') & (w < 0)] = 1
    grp[is_kc[pre] & is_app[post]] = 2
    grp[is_kc[pre] & is_avo[post]] = 3
    odour = [np.flatnonzero((names == 'olfactory') & np.isin(c.type, ['ORN_' + g for g in o])) for o in ODORS6]
    return pre, post, w, n, grp, kc, approach, avoid, dn, odour


def trial(g, seed, reward=True):
    pre, post, w, n, grp, kc, approach, avoid, dn, odour = g
    live = ShiuLive(pre, post, w, n=n, flies=FLIES, seed=seed)
    for f in range(FLIES):
        for ids in odour: live.stimulate(ids, HZ, fly=f)
    live.edge_groups(grp, np.tile(np.float32([1, LN, GAIN, GAIN]), (FLIES, 1)))
    live.plastic(3, kc)
    groups = [approach, avoid, dn]
    readout = (np.arange(FLIES)[None] * n + np.concatenate(groups)[:, None]).reshape(-1)
    live.start([WINDOW], readout)
    entries, p0 = live._stim_arrays()
    only = [np.where(np.isin(entries % n, ids), p0, 0).astype(np.float32) for ids in odour]
    cut = np.cumsum([len(x) for x in groups])[:-1]
    dop = np.where(np.arange(FLIES) < FLIES // 2, 1 if reward else 0, 0).astype(np.float32)

    def test(i, scale=1.0):
        x = live.sniff((only[i] * scale).astype(np.float32))[-1].reshape(-1, FLIES).astype(float)
        ap, av, d = (s.sum(0) for s in np.split(x, cut))
        return {'approach': ap.tolist(), 'avoid': av.tolist(), 'dn': d.tolist(), 'valence': ((ap - av) / np.maximum(ap + av, 1)).tolist()}

    rec = {'seed': seed, 'reward': reward, 'pre': {f'o{i}': test(i) for i in range(len(odour))}}
    live.sniff(only[A])
    live.reward(dop)
    rec['after_A'] = {f'o{i}': test(i) for i in range(len(odour))}
    rec['after_A_intensity'] = {f'o{i}x{s}': test(i, s) for i in [A, *range(2, len(odour))] for s in INTENSITIES}
    live.sniff(only[B])
    live.reward(dop)
    rec['after_reversal'] = {f'o{i}': test(i) for i in (A, B)}
    live.close()
    return rec


def main(mode, *args):
    if mode not in ('noise', 'run'): raise SystemExit('usage: learning_specificity noise SEEDS | run ARM FIRST [SEEDS]')
    OUT.mkdir(parents=True, exist_ok=True)
    arm = 'real' if mode == 'noise' else args[0]
    first = 0 if mode == 'noise' else int(args[1])
    seeds = int(args[0]) if mode == 'noise' else int(args[2]) if len(args) > 2 else 1
    g = setup(arm)
    head = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    recs = []
    for s in range(first, first + seeds):
        t = time.time()
        recs.append(trial(g, 7000 + s, reward=(mode != 'noise')))
        print(f'{mode} {arm} seed {s} {time.time() - t:.1f}s', flush=True)
    path = OUT / f"{mode}-{arm.replace(':', '')}-{first}.json"
    path.write_text(json.dumps({'mode': mode, 'arm': arm, 'flies': FLIES, 'git_head': head, 'odours': ODORS6, 'A': A, 'B': B,
                                'records': recs}) + '\n')
    print('written', path)


if __name__ == '__main__':
    main(*sys.argv[1:])
