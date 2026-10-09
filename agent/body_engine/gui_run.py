"""One command for an engine GUI session (brain_watch WATCH_VIEW=gui, 5 brain flies on the world): the engine is built
on the CPU first, then run under gpu-wait with the display passed through; optionally driven by an SDL event script
(GUI_SCRIPT rows: iteration kind a x y rx ry; real mouse ignored), the window's picture saved (GUI_SHOT) as PNG and
opened; prints the actions EngineControl applied (control.log). Without --script the window is yours until you close it.
gui_check drives its clicks through run().

  .venv/bin/python -m agent.body_engine.gui_run [--script FILE] [--shot] [--seed N] [--timeout S] [--env K=V ...]
"""
import argparse
import collections
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / '.work/agent/body_engine/brain-watch'
LOG = ROOT / '.work/gui/gui_run.log'
F = 5  # brain_watch.ROLES


def prebuild(seed):
    """The binary brain_watch's engine will ask for (play.engine_flags, fly_link.Engine's build), built now on the CPU."""
    os.environ['PLAY_GUI'] = '1'
    from agent.body_engine import fly_link as fl, play, spectate_check as sc
    d = play.OUT / f'fly-world{seed}-f{F}'
    if not (d / 'sizes.txt').is_file():
        print(f'gui_run: {d} not exported yet; the first run builds inside the GPU job')
        return None
    b = sc.build(d, 'fly', play.engine_flags(True), fl.K, brain=1, flies=F)
    print('gui_run: built', b.name)
    return b


def run(script=None, shot=False, seed=1, timeout=3600, env=()):
    """-> (brain_watch exit code, EngineControl's applied actions, PNG path or None)."""
    pre = prebuild(seed)
    t0 = time.time()
    ppm = OUT / 'gui.ppm'
    ppm.unlink(missing_ok=True)
    (OUT / 'control.log').unlink(missing_ok=True)
    e = {'WAYLAND_DISPLAY': os.environ.get('WAYLAND_DISPLAY', ''), 'XDG_RUNTIME_DIR': os.environ.get('XDG_RUNTIME_DIR', ''), 'WATCH_VIEW': 'gui',
         **({'GUI_SCRIPT': str(Path(script).resolve())} if script else {}), **({'GUI_SHOT': str(ppm)} if shot else {}), **dict(kv.split('=', 1) for kv in env)}
    cmd = ['gpu-wait', 'run', '--shared', '--vram', '6', '--priority', '90', '--timeout', str(timeout),
           *[x for k, v in e.items() for x in ('--env', f'{k}={v}')], '--', sys.executable, '-m', 'agent.body_engine.brain_watch', 'world', str(seed)]
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, 'w') as f:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
    if pre:
        late = [b.name for b in pre.parent.glob('fly_k*') if b.stat().st_mtime > t0]
        if late:
            print(f'gui_run: WARN the run built {late} inside the GPU job (prebuilt {pre.name}): an input changed after the prebuild')
    acts = [json.loads(x) for x in (OUT / 'control.log').read_text().splitlines()] if (OUT / 'control.log').is_file() else []
    png = None
    if shot and ppm.is_file():
        png = ppm.with_suffix('.png')
        subprocess.run(['magick', str(ppm), str(png)], check=True)
        subprocess.Popen(['setsid', 'nohup', 'xdg-open', str(png)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    return rc, acts, png


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--script', type=Path)
    p.add_argument('--shot', action='store_true')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--timeout', type=int, default=3600)
    p.add_argument('--env', action='append', default=[])
    a = p.parse_args()
    rc, acts, png = run(a.script, a.shot, a.seed, a.timeout, a.env)
    print('applied:', dict(collections.Counter(x.get('action') for x in acts)))
    if png:
        print('picture:', png)
    if a.shot and not png:
        sys.exit(f'FAIL gui_run: no picture (brain_watch exit {rc}, log {LOG})')
    if rc != 0:
        sys.exit(f'FAIL gui_run: brain_watch exit {rc}, log {LOG}')
    print('PASS gui_run: exit 0')


if __name__ == '__main__':
    main()
