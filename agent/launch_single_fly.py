"""Hardware-rendered single fly, invoked inside the GPU waiting room."""
import argparse
import json
import re
import uuid
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agent.launch_neural_flies import desktop_environment
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--profile',choices=['natural','diagnostic'],default='diagnostic')
    parser.add_argument('--run-id',default=uuid.uuid4().hex)
    args=parser.parse_args()
    if not re.fullmatch('[0-9a-f]{32}',args.run_id):parser.error('invalid run-id')
    env=desktop_environment()
    base=ROOT/'.work/neural-flies/single-live'
    out=base/'runs'/args.run_id;out.mkdir(parents=True,exist_ok=False)
    (base/'latest.json').write_text(json.dumps({'run_id':args.run_id,'output':str(out),'profile':args.profile},indent=2)+'\n')
    renderer=subprocess.check_output(['glxinfo','-B'],env=env,text=True)
    (out/'renderer.txt').write_text(renderer)
    if '7900 XTX' not in renderer or 'llvmpipe' in renderer:raise SystemExit('hardware renderer gate failed')
    import numpy as np
    from agent.neural_connectome import Worker,CPUBrain,digest
    brain=Worker();oracle=CPUBrain()
    try:
        inputs=np.zeros((2,8),np.float32)
        actual=brain.step(inputs,direct=list(range(32)))
        expected=oracle.step(inputs,direct=list(range(32)))
        error=float(np.max(np.abs(np.asarray(actual['features'])-np.asarray(expected['features']))))
        assert error<1e-5
        assert actual['counts']==expected['counts']
        assert actual['counts'][0][1]>=32 and actual['counts'][1][1]==0
        (out/'direct-gate.json').write_text(json.dumps({'status':'PASS','cpu_gpu_max_abs_error':error,
            'targeted':32,'counts':actual['counts'],'worker_sha256':digest(ROOT/'.work/neural-flies/neural_worker')},indent=2)+'\n')
    finally:brain.close()
    cmd=[str(ROOT/'.work/fly-research/arena-venv/bin/python'),'-m','agent.neural_single','--profile',args.profile,'--skip-audit','--output',str(out)]
    os.chdir(ROOT);os.execvpe(cmd[0],cmd,env)
