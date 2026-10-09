"""RuntimeControl for an engine runner (docs/plans/CONTROL_PANEL.md): the panel's socket reaches play's loop.

The runner calls serve(i, rates) once per control step from its rates_at (play.loop), on the host, between steps:
requests are acknowledged at that boundary, as neural_single does. The session registers in RuntimeControl's
registry, so `panelctl state` lists it beside fly sessions.

  status · pause · resume · step (paused, n steps, acknowledged when done) · stop    (neural_single's semantics)
  drive  {"fly": f, "input": name, "on": bool} or {..., "hz": x}   held input for fly f (an input of keymap.toml: w, a, d,
         space, threat, ...) at the runner's own rate for it, or at x Hz; on false / hz < 0 clears it, back to the script
  camera {"azimuth": rad, "elevation": rad, "zoom": x}   orbit of the follow camera (DIR/orbit.bin), any subset
"""
import json
import time

import numpy as np

from agent.runtime_control import RuntimeControl


class EngineControl:
    def __init__(self, out, profile, names, hz, flies, orbit_file, dt):
        self.rc = RuntimeControl(out, profile, 'engine')
        self.log = open(out / 'control.log', 'w')  # every applied action with its snapshot, one JSON line each (the GUI check reads it)
        self.names, self.hz, self.flies, self.orbit_file, self.dt = list(names), list(hz), flies, orbit_file, dt
        self.paused, self.stopping, self.left, self.i = False, False, 0, 0
        self.held = {}  # (fly, input index) -> hz
        self.orbit = np.array([1, -np.pi / 4, 0.45, 4], np.float32)  # on, azimuth, elevation, zoom
        self.idle = lambda: None
        self.drive = []  # latest DN drive per fly [L, R] (the runner's on_step), for the GUI's sparklines
        orbit_file.with_name('control.sid').write_text(self.rc.session_id)  # the engine's GUI (ctl.mojo) reads it

    def snapshot(self):
        return {'status': 'APPLIED', 'paused': self.paused, 'step': self.i, 'time_s': self.i * self.dt, 'flies': self.flies, 'inputs': self.names,
                'held': [{'fly': f, 'input': self.names[j], 'hz': hz} for (f, j), hz in self.held.items()],
                'camera': {'azimuth': float(self.orbit[1]), 'elevation': float(self.orbit[2]), 'zoom': float(self.orbit[3])},
                'drive': self.drive}

    def set_camera(self, azimuth=None, elevation=None, zoom=None):
        for k, v in ((1, azimuth), (2, elevation), (3, zoom)):
            if v is not None:
                self.orbit[k] = float(v)
        tmp = self.orbit_file.with_suffix('.tmp')
        self.orbit.tofile(tmp)
        tmp.replace(self.orbit_file)

    def apply(self, cmd, edit=None, steps=1):
        """One action, from the socket or in process (engine_view's buttons); raises ValueError when refused."""
        edit = edit or {}
        if cmd == 'pause':
            self.paused = True
        elif cmd == 'resume':
            self.paused = False
        elif cmd == 'stop':
            self.stopping = True
        elif cmd == 'step':
            if not self.paused:
                raise ValueError('pause before stepping')
            self.left = steps
        elif cmd == 'drive':
            f, j = int(edit['fly']), self.names.index(edit['input'])
            if not 0 <= f < self.flies:
                raise ValueError(f'fly must be in [0, {self.flies})')
            hz = (float(self.hz[j]) if edit['on'] else -1.0) if 'on' in edit else float(edit['hz'])
            if hz < 0:
                self.held.pop((f, j), None)
            else:
                self.held[(f, j)] = hz
        elif cmd == 'camera':
            self.set_camera(edit.get('azimuth'), edit.get('elevation'), edit.get('zoom'))
        elif cmd != 'status':
            raise ValueError(f'{cmd} is not an engine action')

    def handle(self, conn, req):
        cmd = req['action']
        if self.stopping and cmd not in ('status', 'stop'):
            return self.rc.reply(conn, dict(self.snapshot(), status='REJECTED', error='runner is stopping'))
        if self.rc.waiting_step and cmd not in ('status', 'stop'):
            return self.rc.reply(conn, dict(self.snapshot(), status='REJECTED', error='step in progress'))
        try:
            self.apply(cmd, req.get('edit'), req.get('steps', 1))
        except (ValueError, KeyError, TypeError) as exc:
            return self.rc.reply(conn, dict(self.snapshot(), status='REJECTED', error=f'{cmd}: {exc}'))
        if cmd == 'step':
            self.rc.waiting_step = (conn, req)  # acknowledged when the steps are done
            self.log.write(json.dumps({'action': 'step', 'from': self.i, 'steps': self.left}) + '\n')
            return None
        snap = dict(self.snapshot(), action=cmd, request_id=req['request_id'])
        self.log.write(json.dumps(dict(snap, drive=None)) + '\n')
        self.log.flush()
        self.rc.reply(conn, snap)

    def serve(self, i, rates):
        """rates (flies, inputs) from the script, with held drives on top; None = stop. Blocks while paused."""
        self.i = i
        if self.left:
            self.left -= 1
            if not self.left:
                self.paused = True
                self.rc.finish_step(self.snapshot())
        while True:
            for conn, req in self.rc.poll():
                self.handle(conn, req)
            if self.stopping:
                return None
            if not self.paused or self.left:
                break
            self.idle()  # a window in process (engine_view) stays live while paused
            time.sleep(0.03)
        for (f, j), hz in self.held.items():
            rates[f, j] = hz
        return rates

    def close(self, error=None):
        self.rc.close(error)
