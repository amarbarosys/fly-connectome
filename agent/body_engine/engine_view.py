"""The fly project's GUI on an engine run: MuJoCo's interactive viewer (mouse orbit / pan / zoom, as neural_single uses it)
showing the engine's flies, and a clickable control window in FlyControls' style driving EngineControl in process.

The engine computes; the viewer only draws: each frame the engine's qpos of every fly goes into a display model (the run's
world MJCF with F-1 more copies of the fly attached, as kinestride.model.combine attaches the spider) and mj_kinematics
places it. MuJoCo's renderer draws, not Kinestride's (its keys belong to MuJoCo; use the control window).
"""
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import mujoco
import mujoco.viewer
import numpy as np

INPUTS = ('w', 'a', 'd', 'space', 'threat')


def display_model(base, fly, flies):
    """base MJCF (world + fly 0) plus flies-1 copies of `fly`, prefixed f1_.., their own ground planes hidden."""
    a = mujoco.MjSpec.from_file(str(base))
    for k in range(1, flies):
        b = mujoco.MjSpec.from_file(str(fly))
        root = Path(fly).parent / (b.meshdir or '')
        for mesh in b.meshes:  # absolute mesh files: the copy compiles inside the base's spec
            mesh.file = str((root / mesh.file).resolve()) if mesh.file else mesh.file
        b.meshdir = ''
        a.attach(b, frame=a.worldbody.add_frame(), prefix=f'f{k}_')
    m = a.compile()
    for g in range(m.ngeom):
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE and mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g).startswith('f'):
            m.geom_rgba[g, 3] = 0  # the copies' floors: not drawn
    return m


class EngineView:
    def __init__(self, base, fly, flies, nq, control, roles):
        self.m = display_model(base, fly, flies)
        if self.m.nq != flies * nq:
            raise SystemExit(f'FAIL engine_view: display model nq {self.m.nq} != {flies} flies x {nq}')
        self.d, self.control = mujoco.MjData(self.m), control
        self.view = mujoco.viewer.launch_passive(self.m, self.d)
        with self.view.lock():
            self.view.cam.distance, self.view.cam.azimuth, self.view.cam.elevation = 30, 135, -35
        self.root = tk.Tk()
        self.root.title('Engine controls')
        self.root.attributes('-topmost', True)
        self.root.protocol('WM_DELETE_WINDOW', lambda: control.apply('stop'))
        self.status = tk.StringVar(value='starting')
        ttk.Label(self.root, textvariable=self.status).grid(row=0, column=0, columnspan=6, padx=10, pady=8, sticky='w')
        self.pause = ttk.Button(self.root, text='Pause', command=self.toggle_pause)
        self.pause.grid(row=1, column=0, padx=4, pady=4, sticky='ew')
        ttk.Button(self.root, text='Step 10', command=lambda: self.safe('step', steps=10)).grid(row=1, column=1, padx=4, sticky='ew')
        ttk.Button(self.root, text='Stop', command=lambda: control.apply('stop')).grid(row=1, column=2, padx=4, sticky='ew')
        self.vars = {}
        for f in range(flies):
            ttk.Label(self.root, text=f'fly {f}  {roles[f]}').grid(row=2 + f, column=0, padx=8, sticky='w')
            for c, k in enumerate(k for k in INPUTS if k in control.names):
                v = self.vars[(f, k)] = tk.IntVar(value=0)
                ttk.Checkbutton(self.root, text=k, variable=v,
                                command=lambda f=f, k=k, v=v: self.safe('drive', edit={'fly': f, 'input': k, 'on': bool(v.get())})
                                ).grid(row=2 + f, column=1 + c, padx=2)
        ttk.Label(self.root, text='Mouse in the 3D window: orbit, pan, zoom. Toggles hold that input for the fly.').grid(
            row=3 + flies, column=0, columnspan=6, padx=8, pady=8, sticky='w')
        control.idle = self.idle

    def safe(self, cmd, edit=None, steps=1):
        try:
            self.control.apply(cmd, edit, steps)
        except ValueError as exc:
            self.status.set(str(exc))

    def toggle_pause(self):
        self.safe('resume' if self.control.paused else 'pause')
        self.pause.configure(text='Resume' if self.control.paused else 'Pause')

    def idle(self):
        self.status.set(f'{"paused" if self.control.paused else "running"} · t {self.control.i * self.control.dt:.2f} s simulated')
        self.root.update()
        self.view.sync()

    def frame(self, q):
        """q: (flies, nq) engine state; False once the viewer window is closed."""
        if not self.view.is_running():
            return False
        with self.view.lock():
            self.d.qpos[:] = np.asarray(q).ravel()
            mujoco.mj_kinematics(self.m, self.d)
        self.idle()
        return True

    def close(self):
        self.root.destroy()
        self.view.close()
