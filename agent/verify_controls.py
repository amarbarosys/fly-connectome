"""Click the actual Tk reset button through X11 XTest, then verify runner state.

The live viewer and controls are real; GPU neural/rendering is the default.
Only this process's test-owned controls are clicked.
"""
from __future__ import annotations

import argparse
import ctypes
from ctypes.util import find_library
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from unittest.mock import patch

import numpy as np
from agent import neural_single as runner
from agent.fly_controls import FlyControls
from agent.launch_neural_flies import desktop_environment
from agent.verify_episode_replay import tree_equal


class Mouse:
    def __init__(self):
        self.x = ctypes.CDLL(find_library('X11'))
        self.xt = ctypes.CDLL(find_library('Xtst'))
        self.x.XOpenDisplay.argtypes = [ctypes.c_char_p]
        self.x.XOpenDisplay.restype = ctypes.c_void_p
        self.x.XFlush.argtypes = [ctypes.c_void_p]
        self.x.XCloseDisplay.argtypes = [ctypes.c_void_p]
        self.x.XWarpPointer.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong,
                                       ctypes.c_int, ctypes.c_int, ctypes.c_uint, ctypes.c_uint,
                                       ctypes.c_int, ctypes.c_int]
        self.xt.XTestFakeMotionEvent.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                                ctypes.c_int, ctypes.c_int, ctypes.c_ulong]
        self.xt.XTestFakeButtonEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint,
                                                ctypes.c_int, ctypes.c_ulong]
        self.display = self.x.XOpenDisplay(os.environ['DISPLAY'].encode())
        if not self.display:
            raise RuntimeError('cannot open XWayland display')

    def click(self, point, widget=None):
        if widget:
            window, width, height = widget
            self.x.XWarpPointer(self.display, 0, window, 0, 0, 0, 0, width//2, height//2)
        else:
            self.xt.XTestFakeMotionEvent(self.display, -1, *point, 0)
        self.xt.XTestFakeButtonEvent(self.display, 1, 1, 0)
        self.xt.XTestFakeButtonEvent(self.display, 1, 0, 0)
        self.x.XFlush(self.display)

    def close(self):
        self.x.XCloseDisplay(self.display)


def verify(output, backend='gpu'):
    output.mkdir(parents=True, exist_ok=False)
    environment = desktop_environment()
    if backend == 'cpu':
        environment.update(MUJOCO_GL='glfw', LIBGL_ALWAYS_SOFTWARE='1', GALLIUM_DRIVER='llvmpipe')
    else:
        environment.update(MUJOCO_GL='glfw', LIBGL_ALWAYS_SOFTWARE='0', GALLIUM_DRIVER='radeonsi')
    os.environ.update(environment)
    # This module is launched with the same software-GL settings before imports.
    checks = []
    errors = []
    ready = threading.Event()
    reset_done = threading.Event()
    controls_holder = []
    model_holder = []
    initial = {}
    neural_live = []
    original_reset = runner.reset_body_and_brain
    original_model_init = runner.Readout.__init__
    original_worker_step = runner.Worker.step

    def observed_step(brain, *args, **kwargs):
        result = original_worker_step(brain, *args, **kwargs)
        neural_live[:] = [result]
        return result

    def readout_init(model, *args, **kwargs):
        original_model_init(model, *args, **kwargs)
        model_holder.append(model)

    class ObservedControls(FlyControls):
        def update(self, text=None):
            super().update(text)
            # KWin can move the client after its first map; query live geometry.
            initial['buttons'] = {name: [button.winfo_rootx()+button.winfo_width()//2,
                                        button.winfo_rooty()+button.winfo_height()//2]
                                  for name, button in self.buttons.items()}
            initial['button_windows'] = {name: [button.winfo_id(), button.winfo_width(), button.winfo_height()]
                                         for name, button in self.buttons.items()}

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            controls_holder.append(self)
            initial['window_id'] = self.root.winfo_id()
            initial['buttons'] = json.loads((output/'run/control-buttons.json').read_text())
            ready.set()

    def observed_reset(arena, brain, body, decision, now):
        before_position = arena.position.copy()
        weights = model_holder[0].weights.copy()
        before_neural_nonzero = (bool(np.any(brain.v) or np.any(brain.ever)) if backend == 'cpu'
                                else bool(neural_live and np.any(neural_live[0]['features'])))
        result = original_reset(arena, brain, body, decision, now)
        if backend == 'gpu':
            zero = brain.step(np.zeros((2, 8), np.float32))
            neural_zero = not np.any(zero['features']) and not np.any(zero['counts'])
        else:
            neural_zero = all(not np.any(getattr(brain, k)) for k in ('v', 'spikes', 'trace', 'ever'))
        check = {'physical_click': True, 'time_s': now,
                 'before_position_mm': before_position.tolist(),
                 'after_position_mm': arena.position.tolist(),
                 'full_body_controller_state_matches_initial': tree_equal(body, arena.get_state()),
                 'neural_was_nonzero': before_neural_nonzero,
                 'neural_arrays_zero': neural_zero if backend == 'cpu' else None,
                 'neural_zero_observables': bool(neural_zero),
                 'neural_reset_scope': 'all mutable CPU arrays' if backend == 'cpu' else 'zero-input exchange features and spike/recruitment counts in both GPU lanes; full voltage array not exported',
                 'readout_weights_preserved': bool(np.array_equal(weights, model_holder[0].weights)),
                 'decision_idle': decision.current == 'IDLE',
                 'physics_steps': arena.physics_steps}
        checks.append(check)
        reset_done.set()
        return result

    def drive():
        mouse = None
        try:
            if not ready.wait(15):
                raise RuntimeError('controls did not appear')
            mouse = Mouse()
            # Wait for real movement/neural activation, then pause through the UI.
            trace = output/'run/telemetry.jsonl'
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if trace.exists() and len(trace.read_text().splitlines()) >= 30:
                    break
                time.sleep(.1)
            else:
                raise RuntimeError('runner did not advance')
            buttons = initial['buttons']
            widgets = initial['button_windows']
            initial['clicked_buttons'] = buttons
            mouse.click(buttons['Pause / resume'], widgets['Pause / resume'])
            events = output/'run/events.jsonl'
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if events.exists() and any(json.loads(line)['control_code'] == 32 for line in events.read_text().splitlines()):
                    break
                time.sleep(.05)
            else:
                raise RuntimeError('physical pause click not received')
            subprocess.run(['import', '-window', str(initial['window_id']), str(output/'controls.png')],
                           env=environment, check=True, timeout=10)
            mouse.click(buttons['Reset body / brain'], widgets['Reset body / brain'])
            if not reset_done.wait(10):
                raise RuntimeError('physical reset click not received')
            mouse.click(buttons['Pause / resume'], widgets['Pause / resume'])
            # Confirm new post-reset physics actually runs, then stop by the UI.
            target = len(trace.read_text().splitlines()) + 10
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline and len(trace.read_text().splitlines()) < target:
                time.sleep(.1)
            if len(trace.read_text().splitlines()) < target:
                raise RuntimeError('runner did not resume after reset')
            initial['post_reset_exchanges'] = 10
            mouse.click(buttons['Stop fly'], widgets['Stop fly'])
        except BaseException as exc:
            errors.append(repr(exc))
            if mouse and initial.get('buttons'):
                mouse.click(initial['buttons']['Stop fly'], initial['button_windows']['Stop fly'])
        finally:
            if mouse:
                mouse.close()

    driver = threading.Thread(target=drive, daemon=True)
    driver.start()
    probe = subprocess.check_output(['glxinfo', '-B'], env=environment, text=True)
    (output/'renderer.txt').write_text(probe)
    if backend == 'gpu':
        assert '7900 XTX' in probe and 'llvmpipe' not in probe
    with patch('agent.fly_controls.FlyControls', ObservedControls), \
         patch.object(runner, 'reset_body_and_brain', observed_reset), \
         patch.object(runner.Worker, 'step', observed_step), \
         patch.object(runner.Readout, '__init__', readout_init):
        runner.run(output/'run', backend=backend, seconds=2., viewer=True, audit=False)
    driver.join(timeout=5)
    (output/'interaction.json').write_text(json.dumps({'errors': errors, 'checks': checks,
                                                     'geometry': initial}, indent=2)+'\n')
    assert not errors, errors
    assert len(checks) == 1, checks
    check = checks[0]
    assert check['neural_was_nonzero'] and check['neural_zero_observables']
    assert check['full_body_controller_state_matches_initial'] and check['readout_weights_preserved']
    assert check['decision_idle'] and check['physics_steps'] == 0
    assert np.linalg.norm(np.array(check['before_position_mm'])-check['after_position_mm']) > .01
    assert initial.get('post_reset_exchanges') == 10
    report = {'status': 'PASS', 'backend': 'CPU neural + llvmpipe viewer' if backend == 'cpu' else 'GPU neural + RX 7900 XTX viewer; CPU body physics',
              'controls': 'actual Tk buttons via XWayland XTest mouse injection',
              'reset_check': check, 'post_reset_exchanges': 10,
              'sources': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (Path(__file__), Path(runner.__file__))}}
    (output/'metrics.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', choices=['cpu', 'gpu'], default='gpu')
    args = parser.parse_args()
    verify(args.output, args.backend)
