"""One JSON bridge for the control panel (docs/plans/CONTROL_PANEL.md).

    python -m agent.panelctl state           -> everything the panel draws, one object
    python -m agent.panelctl act '<json>'    -> one action, its answer

Actions: {"do": "preflight"|"start", "profile": "natural"|"diagnostic"}, {"do": "stop"},
{"do": "control", "session_id": ..., "action": <RuntimeControl action>, "steps": n}.
Launches go through flyctl, so through gpu-wait (CLAUDE.md §17); this file never touches the GPU.
"""
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.runtime_control import ACTIONS, REGISTRY, request_control

GPU_DB = Path.home() / '.local/state/gpuwaitingroom/jobs.db'
BOARD = Path.home() / 'Brain' / ROOT.name / 'whiteboard.md'
HOLD = re.compile(r'^\s*(?:[-*]\s*)?(GPU HOLD\b.*)$', re.M)
LIVE_STATES = ('waiting', 'starting', 'running', 'reclaiming')
JOB_COLUMNS = "id, state, json_extract(spec, '$.label') AS label, priority, started_at, submitted_at"


def alive(pid):
    try: os.kill(pid, 0)
    except ProcessLookupError: return False
    except PermissionError: return True
    return True


def sessions():
    out = []
    for path in REGISTRY.glob('*.json'):
        r = json.loads(path.read_text())
        if r.get('lifecycle') == 'running' and alive(r['pid']):
            out.append({k: r.get(k) for k in ('session_id', 'pid', 'profile', 'backend', 'output', 'created_unix')})
    return sorted(out, key=lambda r: r['created_unix'])


def jobs():
    # Read-only query of gpu-waitd's own DB: the CLI costs 330 ms, the daemon's list op 80 ms, this 1 ms.
    try:
        with sqlite3.connect(f'file:{GPU_DB}?mode=ro', uri=True, timeout=1) as db:
            db.row_factory = sqlite3.Row
            rows = db.execute(f"SELECT {JOB_COLUMNS} FROM jobs WHERE state IN ({','.join('?' * len(LIVE_STATES))})"
                              ' ORDER BY submitted_at', LIVE_STATES).fetchall()
    except sqlite3.Error as exc: return {'error': f'gpu-waitd db: {exc}'}
    return [dict(r) for r in rows]


def hold():
    try: head = '\n'.join(BOARD.read_text().split('\n')[:60])
    except OSError: return None
    m = HOLD.search(head)
    return m.group(1).strip() if m else None


def state():
    return {'fly': {'sessions': sessions()}, 'gpu': {'jobs': jobs(), 'hold': hold()},
            'organisms': {'fly': 'live', 'spider': 'no runner', 'worm': 'no runner'}}


def flyctl(*args):
    r = subprocess.run([sys.executable, str(ROOT / 'agent/flyctl.py'), *args], cwd=ROOT,
                       text=True, capture_output=True, timeout=600)
    return {'ok': r.returncode == 0, 'out': (r.stdout or r.stderr).strip()[-2000:]}


def act(req):
    do = req.get('do')
    if do in ('preflight', 'start'):
        if do == 'start' and hold(): return {'ok': False, 'out': 'refused: ' + hold()}
        profile = req.get('profile', 'natural')
        if profile not in ('natural', 'diagnostic'): raise ValueError('profile must be natural or diagnostic')
        return flyctl(do, '--profile', profile)
    if do == 'stop': return flyctl('stop')
    if do == 'control':
        if req.get('action') not in ACTIONS: raise ValueError('unsupported action')
        return {'ok': True, 'out': request_control(req.get('session_id'), req['action'], req.get('steps', 1))}
    raise ValueError(f'unknown do: {do!r}')


def main(argv):
    try:
        if argv[:1] == ['state']: answer = state()
        elif argv[:1] == ['act'] and len(argv) == 2: answer = act(json.loads(argv[1]))
        else: raise ValueError('usage: panelctl state | act <json>')
    except (ValueError, RuntimeError, OSError, TimeoutError) as exc:
        answer = {'ok': False, 'out': f'{type(exc).__name__}: {exc}'}
    print(json.dumps(answer))
    return 0 if answer.get('ok', True) else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
