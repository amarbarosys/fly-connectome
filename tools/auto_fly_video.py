"""Headless video of an auto_fly run: the saved qpos trajectories (auto-fly-SEED[-drop]-traj.npz) rendered with one
mujoco.Renderer (EGL, a second Renderer in a process renders a constant image), arms side by side, a tracking camera
on each fly, piped to ffmpeg as H.264.
  MUJOCO_GL=egl .venv/bin/python tools/auto_fly_video.py TRAJ.npz OUT.mp4 [EVERY]"""
import subprocess
import sys

import mujoco
import numpy as np

from agent.body_engine import fly_link as fl, live

W, H, FPS = 480, 360, 50


def main(path, out, every=20, together=0):
    """together=1: all flies in one frame (mjv_addGeoms: drawn together, simulated apart; they do not collide)."""
    every, together = int(every), int(together)
    z = np.load(path, allow_pickle=True)
    arms, tr = list(z['arms']), z['traj']  # (columns, steps, nq)
    params = live.WORLD[:5] + (1,) + live.WORLD[6:]
    m = live.load(live.terrain(fl.MJCF, params)[0])
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, H, W)
    cam = mujoco.MjvCamera()
    cam.type, cam.distance, cam.elevation = mujoco.mjtCamera.mjCAMERA_FREE, 9.0, -20.0
    if together:
        return _together(m, d, r, cam, arms, tr, out, every)
    ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W * len(arms)}x{H}', '-r', str(FPS),
                           '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', out], stdin=subprocess.PIPE)
    for t in range(0, tr.shape[1], every):
        row = []
        for c in range(len(arms)):
            d.qpos[:] = tr[c, t]
            mujoco.mj_forward(m, d)
            cam.lookat[:] = d.qpos[:3]
            cam.azimuth = 120.0
            r.update_scene(d, cam)
            row.append(r.render())
        ff.stdin.write(np.concatenate(row, 1).tobytes())
    ff.stdin.close()
    if ff.wait():
        sys.exit(f'FAIL ffmpeg {out}')
    print(f'{out}: {len(arms)} arms {arms}, {tr.shape[1] // every} frames at {FPS} fps ({every} ms of sim per frame, real time at every 20)')


def _together(m, d, r, cam, arms, tr, out, every):
    W2, H2 = 960, 720
    r = mujoco.Renderer(m, H2, W2) if (r.width, r.height) != (W2, H2) else r
    others = [mujoco.MjData(m) for _ in arms[1:]]
    opt, pert = mujoco.MjvOption(), mujoco.MjvPerturb()
    cam.distance, cam.elevation = 30.0, -70.0
    ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W2}x{H2}', '-r', str(FPS),
                           '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', out], stdin=subprocess.PIPE)
    for t in range(0, tr.shape[1], every):
        d.qpos[:] = tr[0, t]
        mujoco.mj_forward(m, d)
        for k, dk in enumerate(others):
            dk.qpos[:] = tr[k + 1, t]
            mujoco.mj_forward(m, dk)
        cam.lookat[:] = tr[:, t, :3].mean(0)
        cam.azimuth = 90.0
        cam.distance = 30.0 + max(0.0, float(tr[:, t, 2].mean()) * 1.2)  # pull back while they are high
        r.update_scene(d, cam)
        for dk in others:
            mujoco.mjv_addGeoms(m, dk, opt, pert, mujoco.mjtCatBit.mjCAT_DYNAMIC, r.scene)
        ff.stdin.write(r.render().tobytes())
    ff.stdin.close()
    if ff.wait():
        sys.exit(f'FAIL ffmpeg {out}')
    print(f'{out}: {len(arms)} flies in one scene {arms}, {tr.shape[1] // every} frames at {FPS} fps ({every} ms of sim per frame)')


if __name__ == '__main__':
    main(*sys.argv[1:])
