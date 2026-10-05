"""Designed jumping-spider brain scaffold (step 2 of the empty-pool plan, Brain 2026-10-04 empty-neuron-pool-idea):
regions, sizes and tracts, each with its source, and empty candidate synapses between tract-connected regions.
A MODEL connectome: topology from the spider corpus (.work/papers/spider), sizes inferred (no measured salticid
neuron count exists; Portia "likely < 100,000" neurons, Wikipedia Portia_(spider) ref 18). Synapses start at weight
0 ("empty"): learning (tools/spider_blank_pool_test.py rule, one teaching signal per region) fills them in step 3.

Graph: neurons numbered region by region; each post neuron of a tract gets FANIN candidate pre slots drawn from the
pre region (seeded), weight 0. Sensory input = Poisson on retina neurons; motor output = leg/palp/chelicera neuropils.

check (CPU, then GPU under gpu-wait):
  S1 total neurons within 90k-120k; every region and tract names a source ('inferred: ...' allowed for sizes)
  S2 every region is reachable from a retina and reaches a motor neuropil along tracts
  S3 Shiu GPU run of the whole scaffold, 100 ms of AM-retina drive: retinas fire, no other region fires (empty),
     then the same run with tract weights set to the Shiu stimulus scale fires every region downstream of the retinas
     in tract order (the plumbing carries spikes once synapses exist)
usage: .venv/bin/python -m agent.spider_scaffold check  -> .work/agent/spider/scaffold.npz + scaffold.json
"""
import json
import sys
from collections import deque

import numpy as np

from agent.neural_connectome import ROOT

FANIN = 20
OUT = ROOT / '.work/agent/spider'
MOTOR = ('WLN1', 'WLN2', 'WLN3', 'WLN4', 'PPN', 'ChN')
# name: (neurons per side, sides, source)
REGIONS = {
    'AM_retina': (1200, 2, 'winsor2023: principal retina < 1500 receptors (Land 1969-71, Blest 1985); 1200 = model choice'),
    'ALE_retina': (1000, 2, 'inferred: no count in corpus'),
    'PLE_retina': (1500, 2, 'inferred: no count in corpus'),
    'PME_retina': (50, 2, 'inferred: small posterior median eyes, no count in corpus'),
    'AM1': (2000, 2, 'topology: principal-eye first optic neuropil (winsor2023 Fig 10.2); size inferred'),
    'AM2': (2000, 2, 'topology: principal-eye second optic neuropil (Steinhoff et al. 2017, cited by artiushin2026; not in corpus); size inferred'),
    'AL1': (1500, 2, 'deagro2024 appendix: first neuropil of the ALE; size inferred'),
    'PL1': (2000, 2, 'deagro2024 appendix: first neuropil of the PLE; size inferred'),
    'L2': (3000, 2, 'deagro2024 appendix: second neuropil shared by ALE and PLE; size inferred'),
    'MB': (15000, 2, 'deagro2024 appendix, jin2023 Fig: mushroom bodies; size inferred'),
    'AB': (10000, 1, 'deagro2024 (arcuate body), jin2023 CB (central body); unpaired; size inferred'),
    'PC': (5000, 1, 'inferred: remaining protocerebrum (decision / premotor), no count in corpus'),
    'PPN': (1500, 2, 'jin2023, artiushin2026: pedipalpal neuropil; size inferred'),
    'ChN': (1000, 2, 'artiushin2026: cheliceral neuropil; size inferred'),
    'WLN1': (3000, 2, 'jin2023: walking leg neuropil 1; size inferred'),
    'WLN2': (3000, 2, 'jin2023: walking leg neuropil 2; size inferred'),
    'WLN3': (3000, 2, 'jin2023: walking leg neuropil 3; size inferred'),
    'WLN4': (3000, 2, 'jin2023: walking leg neuropil 4; size inferred'),
}
# (pre, post, ipsilateral only, source)
TRACTS = [
    ('AM_retina', 'AM1', True, 'winsor2023 Fig 10.2: optic nerve -> first optic neuropil of the principal eye'),
    ('AM1', 'AM2', True, 'Steinhoff et al. 2017 (cited by artiushin2026; not in corpus)'),
    ('AM2', 'AB', False, 'Steinhoff et al. 2017 (principal-eye pathway to the arcuate body; not in corpus)'),
    ('ALE_retina', 'AL1', True, 'deagro2024 appendix'),
    ('PLE_retina', 'PL1', True, 'deagro2024 appendix'),
    ('PME_retina', 'L2', True, 'inferred: PME pathway not described in corpus'),
    ('AL1', 'L2', True, 'deagro2024 appendix'),
    ('PL1', 'L2', True, 'deagro2024 appendix'),
    ('L2', 'MB', True, 'deagro2024 appendix (L2 -> mushroom bodies)'),
    ('L2', 'AB', False, 'deagro2024 appendix (secondary-eye input reaches the arcuate body)'),
    ('MB', 'AB', False, 'inferred: MB output to the central complex, as in insects'),
    ('AB', 'PC', False, 'inferred: arcuate body to premotor protocerebrum'),
    ('MB', 'PC', False, 'inferred'),
    *[(('PC', m, False, 'inferred: descending premotor tract to the appendage neuropils')) for m in MOTOR],
]


def layout():
    """Neuron ranges per (region, side): {(name, side): (start, stop)} and the total."""
    at, n = {}, 0
    for name, (k, sides, _) in REGIONS.items():
        for s in range(sides):
            at[(name, s)] = (n, n + k); n += k
    return at, n


def build(seed=5):
    rng = np.random.default_rng(seed)
    at, n = layout()
    pre, post, tract = [], [], []
    for t, (a, b, ipsi, _) in enumerate(TRACTS):
        for sb in range(REGIONS[b][1]):
            lo, hi = at[(b, sb)]
            sides = [sb] if ipsi and REGIONS[a][1] == REGIONS[b][1] else range(REGIONS[a][1])
            pool = np.concatenate([np.arange(*at[(a, sa)]) for sa in sides])
            for j in range(lo, hi):
                pre.append(rng.choice(pool, FANIN, replace=False)); post.append(np.full(FANIN, j)); tract.append(np.full(FANIN, t))
    return at, n, np.concatenate(pre), np.concatenate(post), np.concatenate(tract)


def reach(start, edges):
    seen, q = set(start), deque(start)
    while q:
        x = q.popleft()
        for y in edges.get(x, ()):
            if y not in seen:
                seen.add(y); q.append(y)
    return seen


def check():
    at, n, pre, post, tract = build()
    fails = []
    srcs = all(r[2] for r in REGIONS.values()) and all(t[3] for t in TRACTS)
    ok1 = 90_000 <= n <= 120_000 and srcs
    print(f"{'PASS' if ok1 else 'FAIL'} S1 {n} neurons in {len(at)} region-sides, {len(TRACTS)} tracts, {len(pre)} empty candidate synapses; sources on every row: {srcs}")
    fwd, back = {}, {}
    for a, b, _, _ in TRACTS:
        fwd.setdefault(a, []).append(b); back.setdefault(b, []).append(a)
    eyes = [r for r in REGIONS if r.endswith('_retina')]
    down, up = reach(eyes, fwd), reach(list(MOTOR), back)
    orphan = [r for r in REGIONS if r not in down or r not in up]
    ok2 = not orphan
    print(f"{'PASS' if ok2 else 'FAIL'} S2 every region on a retina-to-motor path; orphans: {orphan}")
    fails += [] if ok1 else ['S1']
    fails += [] if ok2 else ['S2']
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / 'scaffold.npz', pre=pre.astype(np.int32), post=post.astype(np.int32), tract=tract.astype(np.int16), n=n)
    (OUT / 'scaffold.json').write_text(json.dumps({'neurons': n, 'regions': {k: {'per_side': v[0], 'sides': v[1], 'source': v[2]} for k, v in REGIONS.items()},
        'tracts': [{'pre': a, 'post': b, 'ipsilateral': i, 'source': s} for a, b, i, s in TRACTS],
        'ranges': {f'{k[0]}/{k[1]}': v for k, v in at.items()}}, indent=1) + '\n')
    if 'gpu' in sys.argv:
        from agent.shiu import ShiuGPU, W_SYN
        drive = np.concatenate([np.arange(*at[('AM_retina', s)]) for s in range(2)])
        fired = {}
        for label, wt in (('empty', 0.0), ('wired', 10.0)):  # wired: 10 synapses per edge, a typical Shiu count
            b = ShiuGPU(pre, post, np.full(len(pre), wt, np.float32), n=n, flies=1, seed=3)
            b.stimulate(drive, 150.0)
            c = b.run([1000])[0][:, 0]
            fired[label] = {r: int(sum(c[slice(*at[(r, s)])].sum() for s in range(REGIONS[r][1]))) for r in REGIONS}
            print(label, {r: v for r, v in fired[label].items() if v})
        path = ['AM_retina', 'AM1', 'AM2', 'AB', 'PC', 'WLN1']
        ok3 = (fired['empty']['AM_retina'] > 0 and all(v == 0 for r, v in fired['empty'].items() if r != 'AM_retina')
               and all(fired['wired'][r] > 0 for r in path) and fired['wired']['MB'] == 0)
        print(f"{'PASS' if ok3 else 'FAIL'} S3 empty scaffold: only the driven retina fires; wired: spikes reach {path} and not the "
              f"unconnected secondary-eye branch (MB {fired['wired']['MB']})")
        fails += [] if ok3 else ['S3']
    if fails:
        sys.exit(f'FAIL {len(fails)} checks: {fails}')
    print('PASS spider scaffold')


if __name__ == '__main__':
    check()
