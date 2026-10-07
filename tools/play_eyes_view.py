"""Render the fly's two eye cameras at viewing size (320x240 each) for the probe poses: open, wall ahead, wall left,
wall right. For looking only; the brain samples ~800 columns per eye whatever the render size (agent/spatial_vision.py).
usage: PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python tools/play_eyes_view.py [OUT.png]"""
import sys
import mujoco
import numpy as np
from PIL import Image
from agent.body_engine import live, fly_link as fl
from agent.body_engine.play import OUT

m = live.load(live.terrain(fl.MJCF, live.WORLD[:5] + (1,) + live.WORLD[6:])[0])
d, r = mujoco.MjData(m), mujoco.Renderer(m, 240, 320)
q0 = np.fromfile(OUT / 'fly-world1/qpos.bin', np.float32).astype(float)
rows = []
for x, yaw in ((-7, 0), (-17.5, np.pi), (-17.5, np.pi / 2), (-17.5, -np.pi / 2)):
    d.qpos[:] = q0
    d.qpos[0], d.qpos[1], d.qpos[3:7] = x, 0, (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
    mujoco.mj_kinematics(m, d)
    mujoco.mj_camlight(m, d)
    ims = []
    for s in 'lr':
        r.update_scene(d, camera=f'fly/{s}_eye_cam_camera')
        ims.append(r.render())
    rows.append(np.concatenate(ims, 1))
out = sys.argv[1] if len(sys.argv) > 1 else '.work/play-eyes/eyes-sharp.png'
Image.fromarray(np.concatenate(rows, 0)).save(out)
print(out)
