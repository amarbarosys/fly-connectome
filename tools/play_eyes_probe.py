"""Does lamina input from the rendered world reach the DNs? 5 columns, fixed frames, no keys, 0.5 s."""
import json, numpy as np
from PIL import Image
from agent.body_engine import live, fly_link as fl
from agent.body_engine.play import Brain, Eyes, OUT
p = live.WORLD[:5] + (1,) + live.WORLD[6:]
m = live.load(live.terrain(fl.MJCF, p)[0])
q0 = np.fromfile(OUT / 'fly-world1/qpos.bin', np.float32).astype(float)
eyes = Eyes(m)
def pose(x, y, yaw):
    q = q0.copy(); q[0], q[1] = x, y; q[3:7] = np.cos(yaw / 2), 0, 0, np.sin(yaw / 2); return q
poses = {'open': pose(-7, 0, 0), 'ahead': pose(-17.5, 0, np.pi), 'left': pose(-17.5, 0, np.pi / 2), 'right': pose(-17.5, 0, -np.pi / 2)}
vis, ims = [], []
for k, q in poses.items():
    eyes.v.previous = None
    vis.append(eyes(q)); ims.append(np.concatenate(eyes.last, 1))
Image.fromarray(np.concatenate(ims, 0)).resize((512, 768), Image.NEAREST).save('.work/play-eyes/frames.png')
vis.append(np.zeros_like(vis[0]))
names = list(poses) + ['blind']
b = Brain(5, eyes=True)
b.vis = np.array(vis)
b.set(np.zeros((5, len(b.names)), np.float32))
tot = np.zeros((5, 2), np.int64)
s0, _ = b.advance(fl.K)
for _ in range(500):
    s, _ = b.advance(fl.K)
dn = s - s0
b.brain.close()
out = {n: {'drive_mean': float(v.mean()), 'DN L': int(dn[i, 0]), 'DN R': int(dn[i, 1])} for i, (n, v) in enumerate(zip(names, vis))}
print(json.dumps(out, indent=1))
