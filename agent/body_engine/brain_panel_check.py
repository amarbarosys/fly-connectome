"""The Brain panel checked the way a user meets it (gui_run: brain_watch, 5 brain flies, Kinestride's window): a click
on fly 1's card title (fly 1 holds 'w', which fires DNp09) selects it; the panel must then report DNp09 -> walk forward
for fly 1 (the engine's 'brain fly' lines in DIR/play.stderr), the worker's tap must be on column 1, and the window's
picture is saved and opened for a look. Frame rate read from the run log.

  .venv/bin/python -m agent.body_engine.brain_panel_check      (~2 min window; gui_run builds on the CPU first)
"""
import re
import sys

import numpy as np

from agent.body_engine.gui_run import LOG, OUT, ROOT, run

CLICKS = (('select fly 1', 2000, 980, 186), ('stop', 4200, 142, 74))


def main():
    script = ROOT / '.work/gui/brain.script'
    script.write_text(''.join(f'{it} 1025 1 {x} {y} 0 0\n{it + 1} 1026 1 {x} {y} 0 0\n' for _, it, x, y in CLICKS))
    rc, acts, png = run(script, shot=True, timeout=900)
    err = (ROOT / '.work/agent/body_engine/play/fly-world1-f5/play.stderr').read_text(errors='replace')
    events = [l for l in err.splitlines() if l.startswith('brain fly')]
    print('\n'.join(events[-12:]))
    tap = OUT / 'brain.tap'
    head = np.fromfile(tap, np.int32, 4) if tap.is_file() else np.zeros(4, np.int32)
    fps = [float(m) for m in re.findall(r'([\d.]+) fps', LOG.read_text())]
    print(f'tap: seq {head[0]} column {head[1]} tick {head[2]} N {head[3]}; fps {fps[-6:]}')
    gates = {
        'fly 1 selected -> tap column 1': head[1] == 1,
        'fly 1: DNp09 -> walk forward': any(l.startswith('brain fly 1') and 'DNp09  ->  walk forward' in l for l in events),
        'picture saved': png is not None,
        'stop -> exit 0': rc == 0,
    }
    for k, v in gates.items():
        print(('PASS' if v else 'FAIL'), k)
    bad = [k for k, v in gates.items() if not v]
    if bad:
        sys.exit(f'FAIL {len(bad)}/{len(gates)} brain panel gates: {bad}; log {LOG}')
    print(f'PASS brain_panel_check; picture {png}')


if __name__ == '__main__':
    main()
