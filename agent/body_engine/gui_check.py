"""The engine GUI checked the way a user meets it: brain_watch with Kinestride's window (WATCH_VIEW=gui, 5 brain flies on
the world), driven only by mouse clicks pushed through SDL's queue (GUI_SCRIPT, live.mojo), judged by what EngineControl
applied (its control.log): pause -> paused; step 10 -> exactly 10 control steps; fly 1's 'w' toggle -> held; resume ->
running; stop -> the run ends with exit 0.

  .venv/bin/python -m agent.body_engine.gui_check      (gui_run builds on the CPU, then runs under gpu-wait; ~2 min window)
"""
import sys

from agent.body_engine.gui_run import LOG, ROOT, run

CLICKS = (('pause', 3000, 54, 74), ('step', 3300, 98, 74), ('w fly 1', 3600, 978, 216), ('resume', 3900, 54, 74), ('stop', 6000, 142, 74))


def main():
    script = ROOT / '.work/gui/clicks.script'
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(''.join(f'{it} 1025 1 {x} {y} 0 0\n{it + 1} 1026 1 {x} {y} 0 0\n' for _, it, x, y in CLICKS))
    rc, log, png = run(script, shot=True, timeout=900)  # the window's picture just before the stop, opened
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
        'stop -> exit 0': 'stop' in names and rc == 0,
    }
    for k, v in gates.items():
        print(('PASS' if v else 'FAIL'), k)
    bad = [k for k, v in gates.items() if not v]
    if bad:
        sys.exit(f'FAIL {len(bad)}/{len(gates)} gui gates: {bad}; brain_watch exit {rc}, log {LOG}')
    print(f'PASS gui_check: {len(gates)} gates by mouse clicks only; picture {png}')


if __name__ == '__main__':
    main()
