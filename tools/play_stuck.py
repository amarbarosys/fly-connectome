"""Why the live fly sits and shakes: play's world (seed 1, play's start pose), 2 s, four columns: idle + eyes, W + eyes,
idle blind, W blind. Per column: forward (body lengths), root path length, yaw, tilt max, decoded drive mean/max, eye
proximity drive mean, leg joint speed (rad/s, the shaking). usage: gpu-wait run --priority 90 --vram 6 --timeout 1200
--env PYTHONPATH=$PWD -- .venv/bin/python tools/play_stuck.py"""
import json
import tomllib

import numpy as np

from agent.body_engine import fly_link as fl, live, play

COLS = ('idle eyes', 'w eyes', 'idle blind', 'w blind')
km = tomllib.loads(play.KEYMAP.read_text())
brain = play.Brain(len(COLS), km)
d, m, eng = play.setup(len(COLS), seed=1)
gaits = [fl.Gait(m, d, fl.K) for _ in COLS]
eyes = [play.Eyes(m, km['loom']) for _ in range(2)]
w, lm = brain.names.index('w'), [brain.names.index('loom-' + s) for s in ('left', 'right')]
loom, drives, looms = np.zeros((len(COLS), 2)), [], []


def rates_at(i, game):
    r = np.zeros((len(COLS), len(brain.names)), np.float32)
    r[[1, 3], w] = brain.hz[w]
    if i % play.EYE_EVERY == 0 and getattr(eng, 'q', None) is not None:
        for c, col in enumerate((0, 1)):
            loom[col] = eyes[c](eng.q[col])
        looms.append(loom.copy())
    r[:, lm] = brain.hz[lm] * loom
    return r if i < 2000 else None


tr, ref, cost = play.loop(brain, eng, gaits, rates_at, 2000, lambda i, win, game, drive: drives.append(drive.copy()))
brain.brain.close()
eng.close()
root, dr, lo = live.roots(m)[0], np.array(drives), np.array(looms)
legs = [m.jnt_qposadr[j] for j in range(m.njnt) if m.jnt_type[j] == 3 and any(s in m.joint(j).name for s in ('coxa', 'femur', 'tibia'))]
out = {}
for c, n in enumerate(COLS):
    q = tr[:, c]
    r = fl.metrics(m, root, np.r_[q[:1], q])
    out[n] = {'forward_BL': round(float(r['forward'][-1]) / fl.BODY_LENGTH, 2), 'path_mm': round(float(np.linalg.norm(np.diff(q[:, root:root + 2], axis=0), axis=1).sum()), 2),
              'yaw_deg': round(float(r['yaw_deg'][-1]), 1), 'tilt_max': round(float(r['tilt_max']), 1),
              'drive mean [L,R]': np.round(dr[:, c].mean(0), 2).tolist(), 'drive max': np.round(dr[:, c].max(0), 2).tolist(),
              'eye drive mean [L,R]': np.round(lo[:, c].mean(0), 2).tolist(),
              'leg joint speed rad/s': round(float(np.abs(np.diff(q[:, legs], axis=0)).mean() / (fl.K * fl.DT)), 2)}
print(json.dumps({'ref': ref.tolist(), **out}, indent=1))
