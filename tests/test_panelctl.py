import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent import panelctl


def test_state_shape_and_dead_sessions_dropped(tmp_path, monkeypatch):
    monkeypatch.setattr(panelctl, 'REGISTRY', tmp_path)
    monkeypatch.setattr(panelctl, 'jobs', lambda: [])
    base = {'profile': 'natural', 'backend': 'cpu', 'output': '/x', 'created_unix': 1.0, 'lifecycle': 'running'}
    (tmp_path / 'a.json').write_text(json.dumps(dict(base, session_id='a' * 32, pid=os.getpid())))
    (tmp_path / 'b.json').write_text(json.dumps(dict(base, session_id='b' * 32, pid=2 ** 22 + 7)))
    s = panelctl.state()
    assert [r['session_id'] for r in s['fly']['sessions']] == ['a' * 32]
    assert s['organisms']['spider'] == 'no runner'


def test_hold_line_refuses_start(tmp_path, monkeypatch):
    board = tmp_path / 'whiteboard.md'
    board.write_text('# b\n- GPU HOLD: bench\n')
    monkeypatch.setattr(panelctl, 'BOARD', board)
    assert panelctl.hold() == 'GPU HOLD: bench'
    assert panelctl.act({'do': 'start'}) == {'ok': False, 'out': 'refused: GPU HOLD: bench'}
    board.write_text('# b\nPAUSED earlier for the GPU hold\n')
    assert panelctl.hold() is None


def test_cli_rejects_bad_action():
    r = subprocess.run([sys.executable, '-m', 'agent.panelctl', 'act', '{"do": "control", "action": "explode"}'],
                       cwd=ROOT, text=True, capture_output=True)
    assert r.returncode == 1 and 'unsupported action' in json.loads(r.stdout)['out']
