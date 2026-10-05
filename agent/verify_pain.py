"""Pain learning: punishment dopamine (PPL1 in the fly) depresses KC -> approach-MBON synapses, the
mirror of agent/verify_plastic.py's reward on KC -> avoid. Both run as two factor planes in one worker.

Graph and odours as agent/learning_loop.py (group 2 KC -> approach, group 3 KC -> avoid).
(a) The CPU oracle and the resident GPU worker run sniff, reward (dopamine 0, 1, 0, .5 and
punishment 1, 0, .5, 0 by column), sniff for 60 ms, odour A only; counts agree up to the worker's
float noise and the factors read back from the device equal the oracle's.
(b) Learners (punishment 1) and twins (0) sniff A, are punished, then sniff A and B. Pass at 200 ms:
A's approach-MBON spikes fall below half the twins', A's valence loss over the twins exceeds 0.3 and
three times B's, and the reward plane is untouched (every reward factor still 1.0). A spike criterion
(avoid within 20% of twins) was the first instrument and failed at 24%: avoid MBONs lose input from the
depressed approach side through the circuit, so spikes cannot isolate the plane; factors can.
"""
import os
import json
import sys

import numpy as np

from agent.learning_loop import GAIN, HZ, LN, SEED, setup
from agent.neural_connectome import ROOT
from agent.shiu import ShiuBrain, ShiuLive


def brain(engine, pre, post, w, n, grp, kc, odour, flies):
    b = engine(pre, post, w, n=n, flies=flies, seed=SEED)
    for f in range(flies):
        for ids in odour:
            b.stimulate(ids, HZ, fly=f)
    b.edge_groups(grp, np.tile(np.float32([1, LN, GAIN, GAIN]), (flies, 1)))
    b.plastic(3, kc, punish=2)
    return b


def main():
    pre, post, w, n, grp, kc, approach, avoid, odour = setup()
    watch = np.concatenate([kc, approach, avoid])
    out = {}

    # (a) CPU oracle vs resident GPU: sniff, reward + punish, sniff (60 ms, 4 columns, odour A only)
    flies, steps = 4, 600
    d, pun = np.float32([0, 1, 0, .5]), np.float32([1, 0, .5, 0])
    cpu = brain(ShiuBrain, pre, post, w, n, grp, kc, odour[:1], flies)
    gpu = brain(ShiuLive, pre, post, w, n, grp, kc, odour[:1], flies)
    gpu.start([steps], (np.arange(flies)[None] * n + watch[:, None]).reshape(-1))
    p = np.full(len(gpu._stim_arrays()[1]), HZ * gpu.dt, np.float32)
    rows = []
    for k in range(2):
        cpu.reset()
        cpu.seed = SEED + k
        cpu.step(steps)
        a = cpu.spikes[watch]
        b = gpu.sniff(p)[-1].reshape(len(watch), flies)
        rows.append({'sniff': k, 'spikes_cpu': int(a.sum()), 'spikes_gpu': int(b.sum()), 'cells_differing': int((a != b).any(axis=1).sum()),
                     'kc_spiking_cpu': int((a[:len(kc)] > 0).sum())})
        if k == 0:
            cpu.reward(d, pun)
            gpu.reward(d, pun)
    factor_diff = float(np.abs(gpu.factors() - cpu.factor).max())
    gpu.close()
    out['cpu_gpu'], out['factor_max_diff'] = rows, factor_diff
    print('cpu vs gpu', rows, 'factor max diff', factor_diff, flush=True)

    # (b) learners vs twins: sniff A, punish learners, sniff A and B
    flies, windows = 16, [300, 500, 1000, 2000]
    live = brain(ShiuLive, pre, post, w, n, grp, kc, odour, flies)
    live.start(windows, (np.arange(flies)[None] * n + np.concatenate([approach, avoid])[:, None]).reshape(-1))
    entries, p0 = live._stim_arrays()
    only = [np.where(np.isin(entries % n, ids), p0, 0).astype(np.float32) for ids in odour]

    def sniff(p):
        x = live.sniff(p).reshape(len(windows), len(approach) + len(avoid), flies)
        return x[:, :len(approach)].sum(1).astype(float), x[:, len(approach):].sum(1).astype(float)

    learners = np.arange(flies) < flies // 2
    sniff(only[0])
    live.reward(np.zeros(flies, np.float32), np.where(learners, 0 if os.environ.get("MUTANT") else 1, 0).astype(np.float32))  # MUTANT: no punishment
    after_a, after_b = sniff(only[0]), sniff(only[1])
    k = len(kc)
    f = live.factors()
    live.close()
    out['reward_plane_min_max'] = [float(f[:, :k].min()), float(f[:, :k].max())]
    out['punish_plane_learner_kcs_depressed'] = int((f[learners, k:] < 1).sum())
    valence = lambda ap, av: (ap - av) / np.maximum(ap + av, 1)
    pair = lambda x: [float(x[learners].mean()), float(x[~learners].mean())]
    t = {}
    for i, ms in enumerate((30, 50, 100, 200)):
        t[f'{ms}ms'] = {
            'approach_A_learner_twin': pair(after_a[0][i]), 'avoid_A_learner_twin': pair(after_a[1][i]),
            'approach_B_learner_twin': pair(after_b[0][i]),
            'valence_A_learner_twin': pair(valence(after_a[0][i], after_a[1][i])),
            'valence_B_learner_twin': pair(valence(after_b[0][i], after_b[1][i]))}
        print(ms, 'ms', {k: [round(v, 2) for v in x] for k, x in t[f'{ms}ms'].items()}, flush=True)
    out['learning'] = t
    late = t['200ms']
    loss = lambda v: v[1] - v[0]
    ok = (all(r['cells_differing'] <= 1e-3 * len(watch) * 4 and abs(r['spikes_cpu'] - r['spikes_gpu']) <= max(1e-3 * r['spikes_cpu'], 2) for r in rows)
          and rows[0]['kc_spiking_cpu'] > 0 and factor_diff < 1e-6
          and late['approach_A_learner_twin'][0] < .5 * late['approach_A_learner_twin'][1]
          and loss(late['valence_A_learner_twin']) > max(.3, 3 * abs(loss(late['valence_B_learner_twin'])))
          and out['reward_plane_min_max'] == [1.0, 1.0] and out['punish_plane_learner_kcs_depressed'] > 0)
    out['status'] = 'PASS' if ok else 'FAIL'
    print(out['status'])
    (ROOT / '.work/agent/shiu/verify-pain.json').write_text(json.dumps(out, indent=2) + '\n')
    return ok


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
