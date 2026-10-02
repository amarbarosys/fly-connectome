"""Run inside an exclusive gpu-wait lease; TensorFlow policy stays on CPU."""
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.launch_neural_flies import desktop_environment
env=desktop_environment()
out=ROOT/'.work/neural-flies/basic-flight';out.mkdir(parents=True,exist_ok=True)
renderer=subprocess.check_output(['glxinfo','-B'],env=env,text=True)
(out/'renderer.txt').write_text(renderer)
if '7900 XTX' not in renderer or 'llvmpipe' in renderer:raise SystemExit('hardware rendering gate failed')
cmd=[str(ROOT/'.work/neural-flies/flight-venv/bin/python'),str(ROOT/'agent/flight_live.py')]
os.chdir(ROOT);os.execvpe(cmd[0],cmd,env)
