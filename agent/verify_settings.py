"""Brain settings (agent/settings.py) do exactly what they claim, on the CPU oracle and the GPU worker.

cpu  silence  settings {'silence': X} spikes == the graph with X's output edges removed by hand (every cell, 100 ms of
              sugar drive); mutant: the hand edit also removes the next most active cell's edges -> must differ
     cut      settings {'cut': X -> Y} == the graph without the X -> Y edges
     window   cells driven only 5-15 ms: 0 spikes before 5 ms, some by 15.1 ms (a last-step kick fires one step
              later), none added after
     inject   a constant current on a cell with no inputs: spike count == a one-cell scalar model of the same rules
     digest   same setting -> same digest, any change -> a different one
gpu  exact    silence + windowed drive + inject: GPU spike counts == CPU per cell at 50 / 100 / 200 steps (2 flies)
     live     a gain changed between sniffs (ShiuLive.push) == a fresh worker started with that gain, != one without it

usage: .venv/bin/python -m agent.verify_settings cpu
       gpu-wait run --priority 90 --timeout 1800 -- .venv/bin/python -m agent.verify_settings gpu
"""
import json
import re
import sys

import numpy as np

from agent.experiment import cells
from agent.settings import apply, digest
from agent.shiu import V_TH, ShiuBrain, ShiuGPU, ShiuLive
from agent.verify_shiu import REF, _reference_graph

_G = _reference_graph()
PRE, POST, W, INDEX = np.asarray(_G[0]), np.asarray(_G[1]), np.asarray(_G[2]), _G[3]
N = len(INDEX)
SRC = (REF / 'code/benchmark.py').read_text()
SUGAR = [INDEX[int(x)] for x in re.findall(r'7205\d{14}', SRC[SRC.index("'sugar'"):SRC.index("'p9'")])]
CELLS = cells()
FREE = np.flatnonzero((np.bincount(POST, minlength=N) == 0) & (np.bincount(PRE, minlength=N) > 0))  # no inputs


def brain(engine=ShiuBrain, keep=None, flies=1, seed=7):
    keep = slice(None) if keep is None else keep
    b = engine(PRE[keep], POST[keep], W[keep], n=N, flies=flies, seed=seed)
    b.stimulate(SUGAR, 200.0)
    return b


def counts(b, steps=1000):
    b.step(steps)
    return b.counts.copy()


def cpu():
    out = {}
    plain = counts(brain())
    x, z = [int(c) for c in np.argsort(-plain) if c not in SUGAR and (PRE == c).any()][:2]
    s = brain()
    apply(s, {'silence': [{'cells': {'ids': [x]}}]}, CELLS)
    got, hand, mutant = counts(s), counts(brain(keep=PRE != x)), counts(brain(keep=(PRE != x) & (PRE != z)))
    out['silence'] = {'cell': x, 'cells_differing': int((got != hand).sum()), 'mutant_cells_differing': int((got != mutant).sum()),
                      'pass': bool((got == hand).all() and (got != mutant).any())}
    y = int(np.bincount(POST[PRE == x], weights=np.abs(W[PRE == x]), minlength=N).argmax())
    s = brain()
    apply(s, {'cut': [{'pre': {'ids': [x]}, 'post': {'ids': [y]}, 'scale': 0}]}, CELLS)
    got, hand = counts(s), counts(brain(keep=~((PRE == x) & (POST == y))))
    out['cut'] = {'path': [x, y], 'cells_differing': int((got != hand).sum()), 'pass': bool((got == hand).all())}
    q = FREE[:20]
    s = ShiuBrain(PRE, POST, W, n=N, seed=7)
    apply(s, {'stimulate': [{'cells': {'ids': q.tolist()}, 'hz': 200, 'from_ms': 5, 'to_ms': 15}]}, CELLS)
    c = [int(r[q].sum()) for r in s.run([50, 151, 300])]  # a kick on the window's last step fires on the next
    out['window'] = {'spikes_at_5_15_30_ms': c, 'pass': c[0] == 0 and c[1] > 0 and c[2] == c[1]}
    cell, current, T = int(FREE[0]), 5.0, 3000
    s = ShiuBrain(PRE, POST, W, n=N, seed=7)
    apply(s, {'inject': [{'cells': {'ids': [cell]}, 'current': current}]}, CELLS)
    got = int(counts(s, T)[cell])
    a, u, last, want = np.float32(current * s.dt * 1e3), np.float32(0), -10**9, 0  # one-cell model, the step's order
    for t in range(T):
        if t - 1 - last >= s.refractory:
            u = np.float32(u * s.em)
        if t - last >= s.refractory and u > V_TH:
            u, last, want = np.float32(0), t, want + 1
        if last != t:
            u = np.float32(u + a)
    out['inject'] = {'cell': cell, 'current_mV_per_ms': current, 'spikes': got, 'scalar_model': want, 'pass': got == want and want > 0}
    d = []
    for st in ({'silence': [{'cells': {'ids': [x]}}]}, {'silence': [{'cells': {'ids': [x]}}]}, {'silence': [{'cells': {'ids': [z]}}]}):
        b = brain()
        d.append(apply(b, st, CELLS))
    out['digest'] = {'pass': d[0] == d[1] != d[2]}
    return out


def gpu():
    out = {}
    st = {'silence': [{'cells': {'ids': [int(FREE[3])]}}], 'inject': [{'cells': {'ids': [int(FREE[0])]}, 'current': 5.0}],
          'stimulate': [{'cells': {'ids': FREE[:20].tolist()}, 'hz': 200, 'from_ms': 5, 'to_ms': 15}]}
    runs = {}
    for engine in (ShiuBrain, ShiuGPU):
        b = brain(engine, flies=2)
        apply(b, st, CELLS)
        runs[engine.__name__] = b.run([50, 100, 200])
    rows = [int((a != b).any(axis=1).sum()) for a, b in zip(runs['ShiuBrain'], runs['ShiuGPU'])]
    out['exact'] = {'cells_differing_at_50_100_200': rows, 'pass': rows == [0, 0, 0]}
    plain = counts(brain(), 300)
    readout = np.argsort(-plain)[:200]  # the cells sugar actually drives, so a gain change shows
    rule = lambda k: {'gain': [{'pre': {'ids': list(map(int, SUGAR))}, 'scale': k}]}
    live = brain(ShiuLive)
    apply(live, rule(1.0), CELLS)
    live.start([300], readout)
    _, p = live._stim_arrays()
    live.sniff(p)
    live.scale[:, -1] = 4.0
    changed = live.sniff(live.push())
    live.close()
    sniffs = {}
    for k in (1.0, 4.0):  # fresh workers; the first worker's second sniff drew with seed + 1
        fresh = brain(ShiuLive, seed=8)
        apply(fresh, rule(k), CELLS)
        fresh.start([300], readout)
        sniffs[k] = fresh.sniff(p)
        fresh.close()
    out['live'] = {'spikes_changed_fresh4_fresh1': [int(changed.sum()), int(sniffs[4.0].sum()), int(sniffs[1.0].sum())],
                   'pass': bool((changed == sniffs[4.0]).all() and (changed != sniffs[1.0]).any())}
    return out


if __name__ == '__main__':
    r = {'cpu': cpu, 'gpu': gpu}[sys.argv[1] if len(sys.argv) > 1 else 'cpu']()
    for k, v in r.items():
        print('PASS' if v['pass'] else 'FAIL', k, json.dumps(v))
    bad = [k for k, v in r.items() if not v['pass']]
    sys.exit(f'FAIL {len(bad)}/{len(r)} settings checks: {" ".join(bad)}' if bad else 0)
