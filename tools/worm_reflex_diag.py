"""Diagnosis of the worm touch-reflex NEGATIVE (agent/worm.py): saturation, Vth, Gs and Wicks 1996 sign sweeps."""
import sys; sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__))))
import numpy as np
from agent import worm
names, chem, gap, rev = worm.load(); n = len(names); ix = {k: i for i, k in enumerate(names)}
c, g = worm.dense(n, *chem), worm.dense(n, *gap)
def stim(amp):
    I = np.zeros((1, n, 3), np.float32)
    for nm in ('ALML', 'ALMR', 'AVM'): I[0, ix[nm], 1] = amp
    for nm in ('PLML', 'PLMR'): I[0, ix[nm], 2] = amp
    return I
def score(v):
    d = v[0, [ix['AVAL'], ix['AVAR']]].mean(0) - v[0, [ix['AVBL'], ix['AVBR']]].mean(0)
    return d[1] - d[0], d[2] - d[0]
b = worm.WormBrain([c], [g], rev, 3); v = b.run(stim(0), 1.0)[0, :, 0]; s = b.s[0, :, 0]
print(f'baseline rest V median {np.median(v):.1f} [{v.min():.1f},{v.max():.1f}]  s mean {s.mean():.3f} max-possible {worm.AR/(worm.AR+worm.AD):.3f}  sigmoid(V-Vth) median {np.median(1/(1+np.exp(-worm.BETA*(v-worm.VTH)))):.3f}')
rest = v.copy()
for label, vth in (('Vth=-35', worm.VTH), ('Vth=rest (Kunert)', rest[:, None])):
    for amp in (1e3, 1e4, 1e5):
        worm.VTH = vth; b = worm.WormBrain([c], [g], rev, 3); a, p = score(b.run(stim(amp), 1.0))
        print(f'{label:18s} amp {amp:.0e}: ant {a:+.3f} post {p:+.3f}  pass={a > 0 and p < 0}')
worm.VTH = rest[:, None]
print('--- Wicks signs on touch->opposite command edges only ---')
flip = [(p, t) for p in ('PLML', 'PLMR') for t in ('AVAL', 'AVAR', 'AVDL', 'AVDR')] + \
       [(p, t) for p in ('ALML', 'ALMR', 'AVM') for t in ('AVBL', 'AVBR', 'PVCL', 'PVCR')]
for label, vth in (('Vth=-35', -35.0), ('Vth=rest', rest[:, None])):
    for amp in (1e3, 1e4, 1e5):
        worm.VTH = vth; b = worm.WormBrain([c], [g], rev, 3)
        for p, t in flip: b.syn[0, n + ix[t], ix[p]] = b.syn[0, ix[t], ix[p]] * worm.E_INH
        a, p_ = score(b.run(stim(amp), 1.0))
        print(f'{label:9s} amp {amp:.0e}: ant {a:+.3f} post {p_:+.3f}  pass={a > 0 and p_ < 0}')
print('--- Gs sweep, Vth = own rest, GABA-only vs Wicks touch signs ---')
for gs in (100.0, 10.0, 1.0):
    worm.GS, worm.VTH = gs, -35.0
    b = worm.WormBrain([c], [g], rev, 1); r = b.run(np.zeros((1, n, 1), np.float32), 1.0)[0, :, 0].copy()
    for _ in range(3):  # settle Vth to the rest it produces
        worm.VTH = r[:, None]; b = worm.WormBrain([c], [g], rev, 1); r = b.run(np.zeros((1, n, 1), np.float32), 1.0)[0, :, 0].copy()
    for signs in ('gaba', 'wicks'):
        for amp in (1e3, 1e4):
            b = worm.WormBrain([c], [g], rev, 3)
            if signs == 'wicks':
                for p, t in flip: b.syn[0, n + ix[t], ix[p]] = b.syn[0, ix[t], ix[p]] * worm.E_INH
            a, p_ = score(b.run(stim(amp), 1.0))
            print(f'Gs {gs:5.0f} rest med {np.median(r):6.1f} {signs:5s} amp {amp:.0e}: ant {a:+.3f} post {p_:+.3f} pass={a > 0 and p_ < 0}')
print('--- + cross-inhibition between command classes (Wicks 1996) ---')
back, fwd = ('AVAL', 'AVAR', 'AVDL', 'AVDR'), ('AVBL', 'AVBR', 'PVCL', 'PVCR')
cross = flip + [(p, t) for p in back for t in fwd] + [(p, t) for p in fwd for t in back]
for gs in (100.0, 10.0):
    worm.GS, worm.VTH = gs, -35.0
    r = worm.WormBrain([c], [g], rev, 1).run(np.zeros((1, n, 1), np.float32), 1.0)[0, :, 0].copy()
    for _ in range(3):
        worm.VTH = r[:, None]; r = worm.WormBrain([c], [g], rev, 1).run(np.zeros((1, n, 1), np.float32), 1.0)[0, :, 0].copy()
    for amp in (1e3, 1e4):
        b = worm.WormBrain([c], [g], rev, 3)
        for p, t in cross: b.syn[0, n + ix[t], ix[p]] = b.syn[0, ix[t], ix[p]] * worm.E_INH
        a, p_ = score(b.run(stim(amp), 1.0))
        print(f'Gs {gs:5.0f} amp {amp:.0e}: ant {a:+.3f} post {p_:+.3f} pass={a > 0 and p_ < 0}')
