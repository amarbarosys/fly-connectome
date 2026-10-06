"""Plastic KC factors in the Shiu engines: does reward change the right synapses, on CPU and GPU alike?

Graph and odours as agent/learning_loop.py: ALLN inhibition x12 (no flood), KC -> MBON x4,
KC -> avoid-MBON edges plastic; A = the loop's food (VM2+DM1+VA2), B = its decoy (DA1+VA1d+DL3). (a) The CPU oracle and the resident GPU worker run sniff, reward (dopamine
1, 0, .5, 0 by column), sniff for 60 ms; KC and MBON counts must agree up to the worker's float
noise. (b) Learners (dopamine 1) and twins (0) sniff A, get rewarded, then sniff A and B. Pass: at
200 ms A's avoid MBON spikes fall below half the twins', and A's valence gain over the twins
(valence = (approach - avoid) / (approach + avoid)) exceeds 0.3 and three times B's. B's few avoid
spikes do fall too (KCs shared with A lose their factor), which is generalisation, not an error.
(c) Valence at 30, 50, 100 and 200 ms, to choose the loop window.
"""
import os
import json
import subprocess
import sys
import time

import numpy as np

from agent.learning_loop import HZ, SEED, brain, setup
from agent.neural_connectome import ROOT
from agent.shiu import ShiuBrain, ShiuLive

def main():
    pre, post, w, n, grp, kc, approach, avoid, odour = setup()
    watch = np.concatenate([kc, approach, avoid])
    out = {}

    # (a) CPU oracle vs resident GPU: sniff, reward, sniff (60 ms, 4 columns, odour A only)
    flies, steps, d = 4, 600, np.float32([1, 0, .5, 0])
    cpu = brain(ShiuBrain, pre, post, w, n, grp, kc, odour[:1], flies)
    gpu = brain(ShiuLive, pre, post, w, n, grp, kc, odour[:1], flies)
    readout = (np.arange(flies)[None] * n + watch[:, None]).reshape(-1)
    gpu.start([steps], readout)
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
            cpu.reward(d)
            gpu.reward(d)
    gpu.close()
    out['cpu_gpu'] = rows
    print('cpu vs gpu', rows, flush=True)

    # (b, c) learners vs twins: sniff A, reward learners, sniff A and B
    flies, windows = 16, [300, 500, 1000, 2000]
    live = brain(ShiuLive, pre, post, w, n, grp, kc, odour, flies)
    readout = (np.arange(flies)[None] * n + np.concatenate([approach, avoid])[:, None]).reshape(-1)
    live.start(windows, readout)
    entries, p0 = live._stim_arrays()
    only = [np.where(np.isin(entries % n, ids), p0, 0).astype(np.float32) for ids in odour]

    def sniff(p):
        x = live.sniff(p).reshape(len(windows), len(approach) + len(avoid), flies)
        return x[:, :len(approach)].sum(1).astype(float), x[:, len(approach):].sum(1).astype(float)

    learners = np.arange(flies) < flies // 2
    before = sniff(only[0])
    live.reward(np.where(learners, 0 if os.environ.get("MUTANT") else 1, 0).astype(np.float32))  # MUTANT: no reward
    after_a, after_b = sniff(only[0]), sniff(only[1])
    live.close()
    valence = lambda ap, av: (ap - av) / np.maximum(ap + av, 1)
    t = {}
    for i, ms in enumerate((30, 50, 100, 200)):
        t[f'{ms}ms'] = {
            'avoid_A_learner_twin': [float(after_a[1][i, learners].mean()), float(after_a[1][i, ~learners].mean())],
            'avoid_B_learner_twin': [float(after_b[1][i, learners].mean()), float(after_b[1][i, ~learners].mean())],
            'valence_A_learner_twin': [float(valence(*[x[i, learners] for x in after_a]).mean()), float(valence(*[x[i, ~learners] for x in after_a]).mean())],
            'valence_B_learner_twin': [float(valence(*[x[i, learners] for x in after_b]).mean()), float(valence(*[x[i, ~learners] for x in after_b]).mean())]}
        print(ms, 'ms', {k: [round(v, 2) for v in x] for k, x in t[f'{ms}ms'].items()}, flush=True)
    out['learning'] = t
    late = t['200ms']
    gain = lambda v: v[0] - v[1]
    ok = (all(r['cells_differing'] <= 1e-3 * len(watch) * 4 and abs(r['spikes_cpu'] - r['spikes_gpu']) <= max(1e-3 * r['spikes_cpu'], 2) for r in rows)
          and rows[0]['kc_spiking_cpu'] > 0
          and late['avoid_A_learner_twin'][0] < .5 * late['avoid_A_learner_twin'][1]
          and gain(late['valence_A_learner_twin']) > max(.3, 3 * abs(gain(late['valence_B_learner_twin']))))
    out['status'] = 'PASS' if ok else 'FAIL'
    # Baseline and mutant keep separate receipts: one shared path let the no-reward mutant's FAIL overwrite the PASS.
    mutant = bool(os.environ.get('MUTANT'))
    head = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(['git', 'status', '--porcelain', '--', 'agent'], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    out['provenance'] = {'mode': 'mutant_no_reward' if mutant else 'baseline', 'git_head': head, 'agent_dirty': dirty,
                         'written_unix': time.time(), 'expect': 'FAIL' if mutant else 'PASS'}
    print(out['status'], out['provenance'])
    (ROOT / f".work/agent/shiu/verify-plastic{'-mutant' if mutant else ''}.json").write_text(json.dumps(out, indent=2) + '\n')
    return ok


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
