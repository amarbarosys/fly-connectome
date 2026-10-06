"""Live check for world_edit (docs/plans/CONTROL_PANEL.md phase 3), CPU only, through the panel bridge.

Two paused CPU fly runs, same seed. Both step 5 exchanges, run B gets a world_edit that moves its
first food patch, both step 5 more. PASS when the edit is acknowledged at the boundary, logged in
B's events, the first 5 telemetry rows of A and B are identical (same seed, same world), and B's
inputs after the edit differ from A's (the fly sensed the moved food).

    .venv/bin/python tools/panel_world_edit_check.py
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARENA_PY = ROOT / '.work/fly-research/arena-venv/bin/python'
OUT = ROOT / '.work/panel-world-edit'


def bridge(*args):
    r = subprocess.run([sys.executable, '-m', 'agent.panelctl', *args], cwd=ROOT, text=True, capture_output=True, timeout=180)
    answer = json.loads(r.stdout)
    if r.returncode: raise SystemExit(f'FAIL bridge {args[:1]}: {answer}')
    return answer


def launch(name):
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    log = (out / 'runner.log').open('w')
    proc = subprocess.Popen([str(ARENA_PY), '-m', 'agent.neural_single', '--backend', 'cpu', '--headless', '--skip-audit',
                             '--profile', 'natural', '--start-paused', '--output', str(out)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(600):
        if (out / 'control-session.json').exists() and 'SINGLE_NEURAL_FLY_READY' in (out / 'runner.log').read_text(): break
        if proc.poll() is not None: raise SystemExit(f'FAIL launch {name}: exited, see {out}/runner.log')
        time.sleep(.5)
    else: raise SystemExit(f'FAIL launch {name}: not ready in 300 s, see {out}/runner.log')
    return proc, json.loads((out / 'control-session.json').read_text())['session_id'], out


def control(sid, action, **kw):
    return bridge('act', json.dumps({'do': 'control', 'session_id': sid, 'action': action, **kw}))['out']


def rows(out):
    return [json.loads(line)['inputs'] for line in (out / 'telemetry.jsonl').read_text().splitlines()]


def main():
    runs = {name: launch(name) for name in ('a', 'b')}
    try:
        for name, (_, sid, _) in runs.items(): control(sid, 'step', steps=5)
        _, sid_b, out_b = runs['b']
        food = control(sid_b, 'status')['world_food_amounts']
        first = sorted(food)[0]
        ack = control(sid_b, 'world_edit', edit={'food_id': first, 'x': 0.0, 'y': 3.0})
        print('world_edit ack:', json.dumps(ack['world_edit']))
        for name, (_, sid, _) in runs.items(): control(sid, 'step', steps=5)
        a, b = rows(runs['a'][2]), rows(out_b)
        logged = any('world_edit' in json.loads(l) for l in (out_b / 'events.jsonl').read_text().splitlines())
        checks = [('edit acknowledged', ack.get('status') == 'APPLIED' and ack['world_edit']['after'] == {'x': 0.0, 'y': 3.0}),
                  ('edit logged in events.jsonl', logged),
                  ('same seed, same first 5 rows', len(a) >= 10 and a[:5] == b[:5]),
                  ('inputs differ after the edit', a[5:10] != b[5:10])]
        for label, ok in checks: print(('PASS ' if ok else 'FAIL ') + label)
        failed = [label for label, ok in checks if not ok]
        if failed: raise SystemExit(f'FAIL {len(failed)}/{len(checks)} {failed}; runs in {OUT}')
    finally:
        for proc, sid, _ in runs.values():
            try: control(sid, 'stop')
            except SystemExit: proc.terminate()
            proc.wait(timeout=60)


if __name__ == '__main__':
    main()
