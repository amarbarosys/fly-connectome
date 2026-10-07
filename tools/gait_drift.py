"""Body + gait drift without a brain (astra review item 1): constant drives straight into fly_link.Gait on the flat floor,
1 s, one column each: (1,1) must walk straight; (1,.5) and (.5,1) must turn opposite ways by about the same amount.
usage: gpu-wait run --priority 90 --vram 4 --timeout 900 --env PYTHONPATH=$PWD -- .venv/bin/python tools/gait_drift.py [mujoco]
mujoco: the same rollout in MuJoCo on the CPU (no GPU needed), to split engine from controller/body."""
import json
import sys

import mujoco
import numpy as np
from agent.body_engine import fly_link as fl, live
from agent.body_engine.play import OUT, setup

DRIVES = np.array([(1, 1), (1, .5), (.5, 1), (.5, .5)], float)
if sys.argv[1:] == ['mujoco']:
    d, m, _ = fl.prepare(OUT / 'fly')
    gaits, cols = [fl.Gait(m, d, fl.K) for _ in DRIVES], []
    for g, v in zip(gaits, DRIVES):
        x = mujoco.MjData(m)
        for n, a in (('qpos', x.qpos), ('qvel', x.qvel), ('ctrl', x.ctrl)):
            a[:] = np.fromfile(d / f'{n}.bin', np.float32)
        col = []
        for i in range(fl.STEPS):
            x.ctrl[:] = g.ctrl(v)
            for _ in range(fl.K):
                mujoco.mj_step(m, x)
            col.append(x.qpos.copy())
        cols.append(col)
    tr = np.array(cols).transpose(1, 0, 2)
else:
    d, m, eng = setup(len(DRIVES))
    gaits, tr = [fl.Gait(m, d, fl.K) for _ in DRIVES], []
    for i in range(fl.STEPS):
        eng.send(np.stack([g.ctrl(v) for g, v in zip(gaits, DRIVES)]))
        tr.append(eng.recv())
    eng.close()
    tr = np.array(tr)
root = live.roots(m)[0]
out = {}
for c, v in enumerate(DRIVES):
    r = fl.metrics(m, root, tr[:, c])
    out[str(v.tolist())] = {'yaw_deg': float(r['yaw_deg'][-1]), 'forward_BL': float(r['forward'][-1]) / fl.BODY_LENGTH, 'tilt_max': float(r['tilt_max'])}
print(json.dumps(out, indent=1))
