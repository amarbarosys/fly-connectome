"""Live check of the engine runner's control socket, the way the panel reaches it: a real brain_watch run (5 FlyWire-brain
flies, headless) driven only through `python -m agent.panelctl state|act <json>`.

  gpu-wait run --shared --vram 6 --timeout 1200 -- .venv/bin/python -m agent.body_engine.control_live_check

Gates: the session is listed with its flies and inputs; pause holds the step counter for 2 s; step 50 advances it by exactly
50; drive on/off shows in held; camera writes the orbit file; resume moves again; stop ends the run with exit 0.
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def bridge(*args):
    r = subprocess.run([sys.executable, '-m', 'agent.panelctl', *args], cwd=ROOT, text=True, capture_output=True, timeout=180)
    return json.loads(r.stdout)


def act(sid, action, **kw):
    a = bridge('act', json.dumps({'do': 'control', 'session_id': sid, 'action': action, **kw}))
    if not a.get('ok'):
        raise SystemExit(f'FAIL {action}: {a}')
    return a['out']


def main():
    run = subprocess.Popen([sys.executable, '-m', 'agent.body_engine.brain_watch', 'world'], cwd=ROOT, text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env={**os.environ, 'WATCH_HEADLESS': '1'})
    sid = None
    for line in run.stdout:
        if m := re.search(r'control session ([0-9a-f]{32})', line):
            sid = m.group(1)
            break
    if not sid:
        raise SystemExit(f'FAIL no control session line (exit {run.wait()})')
    gates = {}
    while act(sid, 'status')['step'] < 100:  # past play's calibration hold
        time.sleep(1)
    s = [x for x in bridge('state')['fly']['sessions'] if x['session_id'] == sid]
    gates['listed'] = bool(s) and s[0]['backend'] == 'engine' and s[0]['flies'] == 5 and 'w' in s[0]['inputs']
    a = act(sid, 'pause')['step']
    time.sleep(2)
    b = act(sid, 'status')['step']
    gates['pause holds'] = a == b
    c = act(sid, 'step', steps=50)['step']
    gates['step 50'] = c == b + 50
    held = act(sid, 'drive', edit={'fly': 0, 'input': 'w', 'on': True})['held']
    gates['drive on'] = held == [{'fly': 0, 'input': 'w', 'hz': held[0]['hz']}] and held[0]['hz'] > 0 if held else False
    gates['drive off'] = act(sid, 'drive', edit={'fly': 0, 'input': 'w', 'on': False})['held'] == []
    act(sid, 'camera', edit={'zoom': 7.0, 'azimuth': 0.5})
    orbit = sorted(ROOT.glob('.work/agent/body_engine/play/*-f5/orbit.bin'), key=lambda p: p.stat().st_mtime)[-1]
    gates['camera'] = np.allclose(np.fromfile(orbit, np.float32)[[1, 3]], [0.5, 7.0])
    d = act(sid, 'resume')['step']
    time.sleep(2)
    gates['resume moves'] = act(sid, 'status')['step'] > d
    act(sid, 'stop')
    gates['stop exits 0'] = run.wait(timeout=60) == 0
    for k, v in gates.items():
        print(('PASS' if v else 'FAIL'), k)
    bad = [k for k, v in gates.items() if not v]
    if bad:
        raise SystemExit(f'FAIL {len(bad)}/{len(gates)} control gates: {" ".join(bad)}')
    print(f'PASS control_live_check: {len(gates)} gates through panelctl')


if __name__ == '__main__':
    main()
