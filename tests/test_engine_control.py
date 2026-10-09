import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent import panelctl
from agent import runtime_control as rcm
from agent.body_engine.engine_control import EngineControl

NAMES = ['w', 'a', 'd', 'threat']


@pytest.fixture
def runner(tmp_path, monkeypatch):
    """A fake engine runner: serve() once per control step on a thread, as play.loop's rates_at does; serving waits while
    `serving` is cleared (play's HOLD: Python busy, nobody accepts)."""
    monkeypatch.setattr(rcm, 'REGISTRY', tmp_path / 'reg')
    ec = EngineControl(tmp_path, 'test', NAMES, [100.0, 80.0, 80.0, 200.0], 3, tmp_path / 'orbit.bin', 1e-3)
    seen, done = [], threading.Event()
    ec.serving = threading.Event()
    ec.serving.set()

    def run():
        i = 0
        while ec.serving.wait() and (r := ec.serve(i, np.zeros((3, len(NAMES)), np.float32))) is not None:
            seen.append(r.copy())
            i += 1
            time.sleep(0.002)
        done.set()

    threading.Thread(target=run, daemon=True).start()
    yield ec, seen, done
    ec.close()


def build_ctl(tmp_path):
    import subprocess
    exe = tmp_path / 'ctl'
    subprocess.run([str(Path.home() / 'iTools/bin/mojo-cached'), 'build', '-I', str(Path.home() / 'Projects/mojo/mojo-imkit/src'),
                    str(ROOT / 'agent/body_engine/ctl.mojo'), '-o', str(exe)], check=True, capture_output=True)
    return exe


def ask(ec, action, **kw):
    return rcm.request_control(ec.rc.session_id, action, timeout=5, **kw)


def test_pause_step_resume_stop(runner):
    ec, seen, done = runner
    assert ask(ec, 'pause')['paused'] is True
    time.sleep(0.1)
    n = len(seen)
    time.sleep(0.2)
    assert len(seen) == n  # paused: no control steps
    assert ask(ec, 'step', steps=3)['paused'] is True and len(seen) == n + 3
    ask(ec, 'resume')
    with pytest.raises(RuntimeError, match='pause before stepping'):
        ask(ec, 'step', steps=1)
    ask(ec, 'stop')
    assert done.wait(2)


def test_drive_and_camera(runner):
    ec, seen, done = runner
    assert ask(ec, 'drive', edit={'fly': 2, 'input': 'a', 'hz': 150.0})['held'] == [{'fly': 2, 'input': 'a', 'hz': 150.0}]
    time.sleep(0.05)
    assert seen[-1][2, NAMES.index('a')] == 150.0 and seen[-1][1].sum() == 0
    ask(ec, 'drive', edit={'fly': 2, 'input': 'a', 'hz': -1})
    assert ask(ec, 'status')['held'] == []
    assert ask(ec, 'drive', edit={'fly': 0, 'input': 'threat', 'on': True})['held'] == [{'fly': 0, 'input': 'threat', 'hz': 200.0}]
    assert ask(ec, 'drive', edit={'fly': 0, 'input': 'threat', 'on': False})['held'] == []
    with pytest.raises(RuntimeError, match='fly must be'):
        ask(ec, 'drive', edit={'fly': 9, 'input': 'w', 'hz': 1})
    ask(ec, 'camera', edit={'zoom': 6.5})
    assert np.fromfile(ec.orbit_file, np.float32)[3] == 6.5
    with pytest.raises(RuntimeError, match='edit object goes with'):
        ask(ec, 'drive')
    ask(ec, 'stop')
    assert done.wait(2)


def test_session_listed_by_panelctl(runner, monkeypatch):
    ec, seen, done = runner
    monkeypatch.setattr(panelctl, 'REGISTRY', rcm.REGISTRY)
    s = [r for r in panelctl.sessions() if r['session_id'] == ec.rc.session_id]
    assert s and s[0]['backend'] == 'engine' and s[0]['paused'] is False and s[0]['flies'] == 3 and s[0]['inputs'] == NAMES
    ask(ec, 'stop')
    assert done.wait(2)


def test_mojo_client_drives_the_runner(runner, tmp_path):
    """agent/body_engine/ctl.mojo (the engine GUI's client) through the same socket: pause, a drive toggle, step 3."""
    import subprocess
    ec, seen, done = runner
    exe = build_ctl(tmp_path)
    ask = lambda *a: subprocess.run([str(exe), str(tmp_path) + '/', *a], capture_output=True, text=True, timeout=60).stdout
    assert '"paused": true' in ask('pause')
    assert '{"fly": 1, "input": "w", "hz": 100.0}' in ask('drive', '{"fly": 1, "input": "w", "on": true}')
    n = len(seen)
    assert '"status": "APPLIED"' in ask('step') and len(seen) == n + 1
    ask('stop')
    assert done.wait(2)


def test_mojo_client_keeps_an_action_while_the_runner_is_busy(runner, tmp_path):
    """The GUI's click must not be lost when the listener's backlog is full (play's HOLD: nobody accepts): ctl.mojo
    keeps the action and sends it once the runner serves again (a status may be dropped, an action never)."""
    import socket
    import subprocess
    ec, seen, done = runner
    exe = build_ctl(tmp_path)
    ec.serving.clear()
    time.sleep(0.05)
    fill = []
    for _ in range(64):  # fill the listen backlog until connect would block
        s = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        s.setblocking(False)
        try:
            s.connect(ec.rc.address)
        except BlockingIOError:
            s.close()
            break
        fill.append(s)
    p = subprocess.Popen([str(exe), str(tmp_path) + '/', 'pause'], stdout=subprocess.PIPE, text=True)
    time.sleep(0.5)
    for s in fill:  # the stale connections go away, then the runner serves again
        s.close()
    ec.serving.set()
    out = p.communicate(timeout=30)[0]
    assert '"paused": true' in out, f'action lost while the runner was busy: {out!r}'
    ask(ec, 'stop')
    assert done.wait(2)
