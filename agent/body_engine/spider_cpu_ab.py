"""A/B: spider live viewer physics on one GPU wave (WAVE=1) vs the host (CPU=1); 640 x 360, K=10, BRAIN=2, PROFILE.
build: python spider_cpu_ab.py build   run (under gpu-wait): python spider_cpu_ab.py run [frames]"""
import subprocess, sys
import numpy as np
from agent.body_engine.spider_link import files, build, run, yaw_of, CAM

d = files(1)
np.array([1, 2.0], np.float32).tofile(d / 'roots.bin')  # PROFILE traces the root (as game_check)
arms = {'gpu': build(d, 1, 10, 2, w=640, h=360, profile=True, cpu=False), 'cpu': build(d, 1, 10, 2, w=640, h=360, profile=True, cpu=True)}
if sys.argv[1] == 'run':
    frames = sys.argv[2] if len(sys.argv) > 2 else '300'
    for name, b in arms.items():
        r = subprocess.run([str(b), str(d) + '/', frames, *CAM, '0', '0'], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if r.returncode:
            sys.exit(f'FAIL {name}: exit {r.returncode}\n{r.stderr[-800:]}')
        print(name, *[l for l in r.stderr.splitlines() if l.startswith('COST ') or 'ms per frame' in l][-2:])
    out = {c: run(d, 1, 10, 2, 300, packet=50, cpu=c)[0] for c in (False, True)}  # 300 control steps, 0.3 s walking
    for (_, qg, _, _), (_, qc, _, _) in zip(out[False], out[True]):
        dx = float(np.abs(qg[:3, 0] - qc[:3, 0]).max())
        print(f'root gpu {qg[:3, 0].round(4)} cpu {qc[:3, 0].round(4)} |d| {dx:.2e} yaw {yaw_of(qg[:, 0]):.2f} / {yaw_of(qc[:, 0]):.2f}')
    if dx > 1e-2 * max(1.0, float(np.abs(qg[:2, 0]).max())):  # float32 contact chaos: GPU and host part after ~100 steps
        sys.exit(f'FAIL parity: root differs {dx:.2e}')
