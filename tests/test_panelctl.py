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


def test_port_queue_rows_and_receipt(tmp_path, monkeypatch):
    q = tmp_path / 'port_queue.tsv'
    q.write_text('# h\nhfield-cone\tmj_instantiateContact\tengine_core_constraint.c\tcone_hfield\topen\tnote\n')
    port = tmp_path / 'port'; port.mkdir()
    (port / 'cone_hfield.receipt.json').write_text(json.dumps({'result': 'PASS', 'mutants_caught': 1, 'mutants_missed': 0, 'us_per_launch': '5', 'when': 'w', 'head': 'h', 'log': 'l'}))
    monkeypatch.setattr(panelctl, 'QUEUE', q)
    monkeypatch.setattr(panelctl, 'PORT', port)
    rows = panelctl.port_queue()
    assert rows[0]['stage'] == 'cone_hfield' and rows[0]['receipt']['result'] == 'PASS' and rows[0]['fields'] is False
    try:
        panelctl.act({'do': 'port', 'step': 'land', 'name': 'hfield-cone'})
        assert False
    except ValueError as e:
        assert 'reviewed git commit' in str(e)
    try:
        panelctl.act({'do': 'port', 'step': 'gate', 'name': 'nope'})
        assert False
    except ValueError as e:
        assert 'unknown queue row' in str(e)
