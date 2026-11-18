"""Queue the neural two-fly viewer with a bounded exclusive GPU lease."""
from pathlib import Path
import argparse
import datetime
import json
import os
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'.work/neural-flies'


def desktop_environment():
    env=os.environ.copy()
    for pid in subprocess.check_output(['pgrep','-u',str(os.getuid()),'plasmashell'],text=True).split():
        for item in Path(f'/proc/{pid}/environ').read_bytes().split(b'\0'):
            if b'=' in item:
                k,v=item.split(b'=',1)
                if k.decode() in {'DISPLAY','XAUTHORITY','WAYLAND_DISPLAY','XDG_RUNTIME_DIR','DBUS_SESSION_BUS_ADDRESS'}:
                    env[k.decode()]=v.decode()
    env.update(MUJOCO_GL='glfw',LIBGL_ALWAYS_SOFTWARE='0',PYTHONPATH=str(ROOT),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    env.pop('PYTHONHOME',None)
    env.pop('MOJO_PYTHON_LIBRARY',None)
    return env


def run_inside():
    env=desktop_environment()
    probe=subprocess.check_output(['glxinfo','-B'],env=env,text=True)
    (WORK/'renderer.txt').write_text(probe)
    if '7900 XTX' not in probe or 'llvmpipe' in probe:
        raise SystemExit('Hardware-renderer gate failed')
    os.environ.update(env)
    sys.path.insert(0,str(ROOT))
    from agent.verify_embodiment import verify
    verify()
    cmd=[str(ROOT/'.work/fly-research/arena-venv/bin/python'),'-m','agent.neural_duel',
         '--output',str(WORK/'live'), '--seconds','0']
    (WORK/'viewer-process.json').write_text(json.dumps({'pid':os.getpid(),'command':cmd,
        'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n')
    os.chdir(ROOT)
    os.execvpe(cmd[0],cmd,env)


def submit(after=None):
    cmd=[os.path.expanduser('~/.local/bin/gpu-wait'),'submit','--priority','10','--vram','1.0',
         '--timeout','3600','--label','two-neural-flies-exclusive','--cwd',str(ROOT),'--json']
    if after:cmd+=['--after',after,'--after-success']
    cmd+=['--',str(ROOT/'.work/fly-research/arena-venv/bin/python'),str(Path(__file__).resolve()),'--inside']
    result=subprocess.run(cmd,capture_output=True,text=True)
    (WORK/'viewer-submit.json').write_text(result.stdout)
    print(result.stdout.strip())
    if result.returncode:print(result.stderr,file=sys.stderr)
    raise SystemExit(result.returncode)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inside',action='store_true');p.add_argument('--after');a=p.parse_args()
    run_inside() if a.inside else submit(a.after)
