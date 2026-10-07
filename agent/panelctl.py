"""One JSON bridge for the control panel (docs/plans/CONTROL_PANEL.md).

    python -m agent.panelctl state           -> everything the panel draws, one object
    python -m agent.panelctl act '<json>'    -> one action, its answer

Actions: {"do": "preflight"|"start", "profile": "natural"|"diagnostic"}, {"do": "stop"},
{"do": "control", "session_id": ..., "action": <RuntimeControl action>, "steps": n},
{"do": "port", "step": "source"|"brief"|"gate", "name": <queue row>}: the body-engine port lane (~/iTools/bin/mj-port); gate is
detached (it runs under gpu-wait for up to an hour) and lands its receipt in .work/agent/body_engine/port/.
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
            s = {k: r.get(k) for k in ('session_id', 'pid', 'profile', 'backend', 'output', 'created_unix')}
            try:
                st = request_control(s['session_id'], 'status', timeout=2)
                s.update(paused=st.get('paused'), time_s=st.get('time_s'), world_food=st.get('world_food', []))
            except (OSError, ValueError, RuntimeError, TimeoutError) as exc: s['status_error'] = str(exc)[:200]
            out.append(s)
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


QUEUE = ROOT / 'agent/body_engine/port_queue.tsv'
PORT = ROOT / '.work/agent/body_engine/port'
MJ_PORT = Path.home() / 'iTools/bin/mj-port'


def port_queue():
    """The port queue with each row's receipt (mj-port list, read directly: one file and a glob, no subprocess)."""
    rows = []
    try: lines = QUEUE.read_text().splitlines()
    except OSError: return rows
    for line in lines:
        if not line.strip() or line.startswith('#'): continue
        name, fn, file, stage, status, note = (line.split('\t') + [''] * 6)[:6]
        r = {'name': name, 'function': fn, 'file': file, 'stage': stage, 'status': status, 'note': note, 'receipt': None,
             'fields': (PORT / f'{fn}.fields.md').exists(), 'brief': (ROOT / f'.work/briefs/port-{fn}.md').exists(),
             'running': (PORT / f'{stage}.running').exists()}
        try:
            j = json.loads((PORT / f'{stage}.receipt.json').read_text())
            r['receipt'] = {k: j.get(k) for k in ('result', 'mutants_caught', 'mutants_missed', 'us_per_launch', 'when', 'head')}
        except (OSError, ValueError): pass
        rows.append(r)
    return rows


def port_act(req):
    step, name = req.get('step'), req.get('name')
    row = next((r for r in port_queue() if r['name'] == name), None)
    if row is None: raise ValueError(f'unknown queue row {name!r}')
    if step in ('source', 'brief'):
        args = [str(MJ_PORT), 'source', row['function'], row['file']] if step == 'source' else [str(MJ_PORT), 'brief', row['function'], row['stage']]
        r = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, timeout=120)
        return {'ok': r.returncode == 0, 'out': (r.stdout or r.stderr).strip()[-2000:]}
    if step == 'gate':
        if hold(): return {'ok': False, 'out': 'refused: ' + hold()}
        flag = PORT / f"{row['stage']}.running"; PORT.mkdir(parents=True, exist_ok=True); flag.touch()
        subprocess.Popen(['bash', '-c', f'{MJ_PORT} gate {row["stage"]}; rm -f {flag}'], cwd=ROOT, start_new_session=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return {'ok': True, 'out': f"gate {row['stage']} started (gpu-wait); receipt lands in {PORT}"}
    raise ValueError('port step must be source, brief or gate (land is a reviewed git commit: mj-port land STAGE "subject")')


def state():
    return {'fly': {'sessions': sessions()}, 'gpu': {'jobs': jobs(), 'hold': hold()},
            'organisms': {'fly': 'live', 'spider': 'no runner', 'worm': 'no runner'}, 'port': port_queue()}


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
    if do == 'port': return port_act(req)
    if do == 'control':
        if req.get('action') not in ACTIONS: raise ValueError('unsupported action')
        return {'ok': True, 'out': request_control(req.get('session_id'), req['action'], req.get('steps', 1), edit=req.get('edit'))}
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
