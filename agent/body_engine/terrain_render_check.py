"""Renderer terrain gate: amplitude-0 terrain (TERRAIN=1, 4 octaves) vs the old z = 0 floor on the SAME engine state,
frame 1, max channel diff <= 1. usage: gpu-wait run --priority 90 --timeout 900 -- .venv/bin/python -m agent.body_engine.terrain_render_check"""
import shutil, subprocess, sys
import numpy as np
from pathlib import Path
from agent.body_engine.model import terrain
from agent.body_engine.live import prepare, build
B = '.work/agent/gpu-body-port/body-v2/body.xml'
frames = {}
for name, mjcf in (('flat', B), ('terrain0', terrain(B, (0, 0, 0.0, 8.0, 4, 1, 2.0, 0.5))[0])):
    out = Path(f'.work/agent/body_engine/render-equiv/{name}')
    cam, m = prepare(out, mjcf=mjcf)
    if name == 'terrain0':  # the flat run's exact state
        for f in ('qpos.bin', 'qvel.bin', 'ctrl.bin'):
            shutil.copy(Path('.work/agent/body_engine/render-equiv/flat') / f, out / f)
        cam = flatcam
    flatcam = cam
    subprocess.run([str(build(out, m, 1)), str(out) + '/', '1', *cam, str(out / 'f.rgba')], check=True, stderr=subprocess.DEVNULL)
    frames[name] = np.frombuffer((out / 'f.rgba').read_bytes(), np.uint8).reshape(-1, 4).astype(int)
d = np.abs(frames['flat'] - frames['terrain0'])
print(f'{"PASS" if d.max() <= 1 else "FAIL"} render flat-equivalence: max channel diff {d.max()}, pixels differing {(d.max(1) > 0).sum()} of {len(d)}')
sys.exit(0 if d.max() <= 1 else 1)
