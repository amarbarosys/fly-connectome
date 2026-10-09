"""Watch several flies, each driven by its own copy of the FlyWire brain (play's Brain, one column per fly), in the engine's
window. No keyboard: a script fires the same neurons play's keys do, per fly, and the brain does the rest (descending
neurons -> drive -> gait -> body; its own drives can outvote the script, as in play).

  fly 0 idle (no input) · 1 W (walk) · 2 A (turn left) · 3 D (turn right) · 4 W, with a looming threat on both eyes every 4 s

  gpu-wait run --shared --vram 6 --priority 90 --timeout 3600 -- .venv/bin/python -m agent.body_engine.brain_watch [world [SEED]]

Flies start 5 mm apart on a line; the camera follows fly 0 (the idle one) at WATCH_ZOOM (default 4).
Controls: the run owns a RuntimeControl (engine_control.py), so /panel's engine tab or `python -m agent.panelctl act ...`
pause, step, stop it, hold inputs per fly (drive) and orbit the camera.
A frame every WATCH_RENDER control steps (default 5). Close the mpv window to stop. WATCH_HEADLESS=1: no window (timing).
"""
import os
import subprocess
import sys
import time
import tomllib

import numpy as np

from agent.body_engine import fly_link as fl, spectate_check as sc
from agent.body_engine.engine_control import EngineControl
from agent.body_engine.play import KEYMAP, OUT, Brain, loop, setup

ROLES = ('idle', 'w', 'a', 'd', 'w+threat')
SPACING = 5.0  # mm between start positions
ZOOM = float(os.environ.get('WATCH_ZOOM', 4))
RENDER = int(os.environ.get('WATCH_RENDER', 5))  # control steps (1 ms each) per frame: play's 24 gave ~8 fps at 5 flies
HEADLESS = os.environ.get('WATCH_HEADLESS') == '1'  # frames to /dev/null, no window: timing runs (the engine prints ms per control step)
GRAB = int(os.environ.get('WATCH_GRAB', 0))  # > 0: no window, keep frame GRAB as OUT/grab.rgba, then stop


def main(*args):
    seed = next((int(x) for x in args[1:] if x.isdigit()), 1) if args[:1] == ('world',) else None
    km = tomllib.loads(KEYMAP.read_text())
    F = len(ROLES)
    brain = Brain(F, km)
    out = OUT.parent / 'brain-watch'
    out.mkdir(parents=True, exist_ok=True)
    fifo = out / 'frames.fifo'
    fifo.unlink(missing_ok=True)
    os.mkfifo(fifo)
    frame = sc.W * sc.H * 4
    grab = ['dd', f'if={fifo}', f'of={out / "grab.rgba"}', f'bs={frame}', 'count=1', f'skip={GRAB}', 'iflag=fullblock', 'status=none']
    mpv = subprocess.Popen(grab) if GRAB else subprocess.Popen(['cat', str(fifo)], stdout=subprocess.DEVNULL) if HEADLESS else subprocess.Popen(['mpv', '--demuxer=rawvideo', f'--demuxer-rawvideo-w={sc.W}', f'--demuxer-rawvideo-h={sc.H}',
                            '--demuxer-rawvideo-mp-format=rgba', '--profile=low-latency', '--untimed', '--title=brain flies', str(fifo)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def q0(d):  # the settled standing pose, root moved along y: 0, -5, +5, -10, +10 mm
        q = np.fromfile(d / 'qpos.bin', np.float32).astype(float)
        qs = np.tile(q, (F, 1))
        for f in range(F):
            qs[f, 1] += SPACING * ((f + 1) // 2) * (1 if f % 2 == 0 else -1)
        return qs

    d, m, eng = setup(F, render=RENDER, out=fifo, seed=seed, q0=q0)
    edir = d.parent / f'{d.name}-f{F}'  # the engine's own dir for F > 1 (fly_link.Engine): it reads orbit.bin from there
    gaits = [fl.Gait(m, d, fl.K) for _ in ROLES]
    ix = {k: brain.names.index(k) for k in ('w', 'a', 'd', 'threat')}
    every, dur = round(4.0 / (fl.K * fl.DT)), round(km['threat']['dur_s'] / (fl.K * fl.DT))
    control = EngineControl(out, 'brain-watch', brain.names, brain.hz, F, edir / 'orbit.bin', fl.K * fl.DT)
    control.set_camera(-np.pi / 4, 0.45, ZOOM)
    print('brain flies: ' + ', '.join(f'{f} {r}' for f, r in enumerate(ROLES)) + f'; control session {control.rc.session_id}; close the window to stop', flush=True)

    clock = [time.perf_counter(), 0]

    def rates_at(i, game):
        if mpv.poll() is not None:
            return None
        if i and i % 500 == 0:  # wall ms per 1 ms control step over the last 500 (brain + gait + physics + render)
            t = time.perf_counter()
            print(f'step {i}: {(t - clock[0]) / (i - clock[1]) * 1e3:.2f} ms per control step, {1e3 / RENDER / ((t - clock[0]) / (i - clock[1]) * 1e3):.1f} fps', flush=True)
            clock[:] = t, i
        r = np.zeros((F, len(brain.names)), np.float32)
        for col, key in ((1, 'w'), (2, 'a'), (3, 'd'), (4, 'w')):
            r[col, ix[key]] = brain.hz[ix[key]]
        if i % every < dur:
            r[4, ix['threat']] = brain.hz[ix['threat']]
        return control.serve(i, r)  # panel requests at this boundary; held drives on top; None = stop

    try:
        loop(brain, eng, gaits, rates_at, 0)
    except (BrokenPipeError, SystemExit):  # the engine ended: the window was closed (its frame pipe broke), else show why
        eng.p.wait()
        try:
            mpv.wait(timeout=5)  # a closed window: mpv may still be shutting down when the engine's frame write fails
        except subprocess.TimeoutExpired:
            pass
        if mpv.poll() is None and eng.p.returncode != -13:  # -13: SIGPIPE, the frame reader went away
            sys.exit(f'FAIL engine exited {eng.p.returncode}: {eng.stderr()[-1500:]}')
        print('window closed', flush=True)
        return
    finally:
        control.close()
        brain.brain.close()
        mpv.terminate()
    eng.close()


if __name__ == '__main__':
    main(*sys.argv[1:])
