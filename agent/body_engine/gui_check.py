"""The engine GUI checked the way a user meets it: brain_watch with Kinestride's window (WATCH_VIEW=gui, 5 brain flies on
the world), driven only by mouse clicks pushed through SDL's queue (GUI_SCRIPT, live.mojo), judged by what EngineControl
applied (its control.log): pause -> paused; step 10 -> exactly 10 control steps; fly 1's 'w' toggle -> held; resume ->
running; stop -> the run ends with exit 0.

  gpu-wait run --shared --vram 6 --timeout 900 --env WAYLAND_DISPLAY=$WAYLAND_DISPLAY --env XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR -- .venv/bin/python -m agent.body_engine.gui_check
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / '.work/agent/body_engine/brain-watch'
CLICKS = (('pause', 3000, 54, 74), ('step', 3300, 98, 74), ('w fly 1', 3600, 978, 216), ('resume', 3900, 54, 74), ('stop', 6000, 142, 74))


def main():
    script = ROOT / '.work/gui/clicks.script'
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(''.join(f'{it} 1025 1 {x} {y} 0 0\n{it + 1} 1026 1 {x} {y} 0 0\n' for _, it, x, y in CLICKS))
    env = {**os.environ, 'WATCH_VIEW': 'gui', 'GUI_SCRIPT': str(script), 'GUI_SHOT': str(OUT / 'gui.ppm')}  # the window's picture at the end
    r = subprocess.run([sys.executable, '-m', 'agent.body_engine.brain_watch', 'world'], cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    log = [json.loads(line) for line in (OUT / 'control.log').read_text().splitlines()]
    acts = [a for a in log if a.get('action') != 'status']
    names = [a['action'] for a in acts]
    print('applied:', ' '.join(names))
    get = lambda name: next((a for a in acts if a['action'] == name), None)
    pause, step, drive, resume = get('pause'), get('step'), get('drive'), get('resume')
    gates = {
        'pause -> paused': bool(pause) and pause['paused'],
        'step 10 exact': bool(step and drive) and drive['step'] == step['from'] + 10,
        "fly 1 'w' held": bool(drive) and {'fly': 1, 'input': 'w'} in [{k: h[k] for k in ('fly', 'input')} for h in drive['held']],
        'resume -> running': bool(resume) and not resume['paused'],
        'stop -> exit 0': 'stop' in names and r.returncode == 0,
    }
    for k, v in gates.items():
        print(('PASS' if v else 'FAIL'), k)
    bad = [k for k, v in gates.items() if not v]
    if bad:
        sys.exit(f'FAIL {len(bad)}/{len(gates)} gui gates: {bad}; brain_watch exit {r.returncode}: {r.stderr[-400:]}')
    print(f'PASS gui_check: {len(gates)} gates by mouse clicks only')


if __name__ == '__main__':
    main()
