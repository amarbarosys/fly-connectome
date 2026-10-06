"""Touch-reflex check for agent/worm.py: anterior touch (ALM, AVM) should raise AVA over AVB, posterior
touch (PLM) AVB over AVA; real Cook 2019 graph vs 5 degree-preserving shuffles. Run: .venv/bin/python -I tools/worm_reflex_check.py"""
import sys, time, tracemalloc; sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__))))
import numpy as np
from agent import worm
from agent.shiu_shuffle import shuffle_edges
names, chem, gap, rev = worm.load(); n = len(names); ix = {k: i for i, k in enumerate(names)}
def arm(seed):
    if seed is None: return worm.dense(n, *chem), worm.dense(n, *gap)
    (cp, cq, cw), (gp, gq, gw) = chem, gap
    return worm.dense(n, cp, shuffle_edges(cp, cq, seed)[0], cw), worm.dense(n, gp, shuffle_edges(gp, gq, seed)[0], gw)
arms = [arm(s) for s in (None, 1, 2, 3, 4, 5)]
I = np.zeros((1, n, 3), np.float32)
for nm in ('ALML', 'ALMR', 'AVM'): I[0, ix[nm], 1] = 1e4
for nm in ('PLML', 'PLMR'): I[0, ix[nm], 2] = 1e4
tracemalloc.start()
b = worm.WormBrain([a[0] for a in arms], [a[1] for a in arms], rev, 3)
t = time.perf_counter(); v = b.run(I, 1.0); el = time.perf_counter() - t
peak = tracemalloc.get_traced_memory()[1]
bw = lambda v, a, b_: v[:, [ix[a + 'L'], ix[a + 'R']]].mean(1) - v[:, [ix[b_ + 'L'], ix[b_ + 'R']]].mean(1)
d = bw(v, 'AVA', 'AVB'); d = d - d[:, :1]
for k, name in enumerate(['real'] + [f'shuffle{s}' for s in range(1, 6)]):
    print(f'{name:9s} ant dAVA-AVB {d[k,1]:+.4f} (want +)  post {d[k,2]:+.4f} (want -)  pass={d[k,1] > 0 and d[k,2] < 0}')
print(f'6 arms x 3 cols, 1 s = 10k steps: {el:.2f} s, {el/1e4*1e6:.0f} us/step, engine peak {peak/1e6:.1f} MB, finite {np.isfinite(v).all()}')
# baseline: explicit Euler needs dt < 2.6e-5 for this gap spectrum -> >= 4x the steps per simulated second
