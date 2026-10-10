"""Crossover sweep: ms per frame (K=10, BRAIN=2, no display) for F spiders on the host (CPU=1), one GPU wave per
animal (WAVE=1) and one GPU thread per animal (WAVE=0). build: python -m ... build   run (gpu-wait): python -m ... run"""
import subprocess, sys
import numpy as np
from agent.body_engine.spider_link import files, build, CAM

FS = (1, 16, 64, 256, 1024)
ARMS = {'cpu': dict(cpu=True), 'wave': dict(cpu=False, wave=True), 'thread': dict(cpu=False, wave=False)}
fails = []
for F in FS:
    d = files(F)
    np.array([1, 2.0], np.float32).tofile(d / 'roots.bin')
    for name, kw in ARMS.items():
        b = build(d, F, 10, 2, w=64, h=64, profile=True, **kw)
        if sys.argv[1] != 'run':
            continue
        r = subprocess.run([str(b), str(d) + '/', '30', *CAM, '0', '0'], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        cost = [l for l in r.stderr.splitlines() if l.startswith('COST ')]
        if r.returncode or not cost:
            fails.append(f'{name}@{F}')
            continue
        phys = float(cost[-1].split()[2])
        print(f'F {F:5d} {name:6s} physics {phys:8.2f} ms/frame  {1000 * phys / 10 / F:8.2f} us/animal-step', flush=True)
if fails:
    sys.exit(f'FAIL {len(fails)}/{len(FS) * len(ARMS)} {" ".join(fails)}')
