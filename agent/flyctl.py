"""Host-local launch lifecycle; CPU preflight precedes exclusive GPU admission."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import uuid
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agent.launch_neural_flies import desktop_environment
WORK=ROOT/'.work/neural-flies';STATE=WORK/'flyctl.json'
GPU=str(Path.home()/'.local/bin/gpu-wait');PY=ROOT/'.work/fly-research/arena-venv/bin/python'

def call(args):
    r=subprocess.run(args,text=True,capture_output=True)
    if r.returncode:raise RuntimeError(r.stderr or r.stdout)
    return r.stdout

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['start','preflight','status','logs','stop'])
    p.add_argument('--profile',choices=['natural','diagnostic'],default='diagnostic')
    p.add_argument('--timeout',type=int,default=3600);a=p.parse_args()
    if a.timeout<=0:p.error('timeout must be positive')
    WORK.mkdir(exist_ok=True,parents=True)
    if a.action in ('status','logs','stop'):
        if not STATE.exists():raise RuntimeError('No managed fly job; launch with fly-live start')
        job=json.loads(STATE.read_text())['job_id']
        args=[GPU,{'status':'status','logs':'logs','stop':'cancel'}[a.action],job]
        args+=['--tail','12'] if a.action=='logs' else ['--json']
        print(call(args).strip());return
    jobs=json.loads(call([GPU,'list','--json']))['jobs']
    active=[j for j in jobs if j.get('state') in ('waiting','starting','running') and j.get('label') in ('one-neural-fly-exclusive','two-neural-flies-exclusive')]
    if active and a.action=='start':raise RuntimeError('Fly already queued/running: '+','.join(j['id'] for j in active))
    env=dict(desktop_environment(),LIBGL_ALWAYS_SOFTWARE='1')
    run_id=uuid.uuid4().hex
    preflight=WORK/'preflights'/run_id
    preflight.mkdir(parents=True,exist_ok=False)
    cmd=[str(PY),'-m','agent.neural_single','--profile',a.profile,'--backend','cpu','--headless','--skip-audit','--seconds','.02','--output',str(preflight)]
    with (preflight/'preflight.log').open('w') as log:
        r=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    if r.returncode:raise RuntimeError('CPU preflight failed: '+str(preflight/'preflight.log'))
    print('CPU_PREFLIGHT_PASS '+str(preflight))
    if a.action=='preflight':return
    result=call([GPU,'submit','--priority','10','--vram','1','--timeout',str(a.timeout),
                 '--label','one-neural-fly-exclusive','--cwd',str(ROOT),'--json','--',str(PY),str(ROOT/'agent/launch_single_fly.py'),'--profile',a.profile,'--run-id',run_id])
    parsed=json.loads(result);parsed['profile']=a.profile;parsed['run_id']=run_id;parsed['output']=str(WORK/'single-live/runs'/run_id);STATE.write_text(json.dumps(parsed,indent=2)+'\n');print(result.strip())
if __name__=='__main__':
    try:main()
    except (RuntimeError,OSError,ValueError) as exc:print('flyctl:',exc,file=sys.stderr);sys.exit(1)
