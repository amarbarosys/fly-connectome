"""Actual single-runner save/stop/restart/continuation gate, CPU or queued GPU."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from agent.launch_neural_flies import desktop_environment
from agent.neural_connectome import ROOT, digest
from agent.runtime_control import request_control
from agent.live_checkpoint import tree_hash, load
from fly_arena.checkpoint import load_checkpoint, save_checkpoint


def start(out, backend, profile, checkpoint=None):
    out.mkdir()
    env=dict(desktop_environment(),LIBGL_ALWAYS_SOFTWARE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    command=[sys.executable,'-m','agent.neural_single','--backend',backend,'--profile',profile,
             '--headless','--skip-audit','--start-paused','--output',str(out)]
    if checkpoint: command+=['--restore-checkpoint',str(checkpoint)]
    with (out/'runner.log').open('w') as log:
        proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    try:
        deadline=time.monotonic()+60
        while time.monotonic()<deadline:
            if (out/'control-session.json').exists():
                sid=json.loads((out/'control-session.json').read_text())['session_id']
                request_control(sid,'status')
                return proc,sid
            if proc.poll() is not None: raise RuntimeError((out/'runner.log').read_text())
            time.sleep(.05)
        raise TimeoutError('runner readiness timeout')
    except BaseException:
        stop_process(proc)
        raise


def stop_process(proc):
    if proc.poll() is None:
        proc.terminate()
        try:proc.wait(timeout=10)
        except subprocess.TimeoutExpired:proc.kill();proc.wait()


def verify(out, backend, profile, stage='both'):
    if stage!='restore':out.mkdir(parents=True,exist_ok=False)
    owned=[]
    started=time.perf_counter()
    try:
        if stage!='restore':
            proc,sid=start(out/'original',backend,profile);owned.append(proc)
            request_control(sid,'step',20)
            prefix=request_control(sid,'checkpoint')
            request_control(sid,'step',8)
            expected=request_control(sid,'checkpoint')
            request_control(sid,'stop');proc.wait(timeout=10);assert proc.returncode==0
            rows=[json.loads(row) for row in (out/'original/telemetry.jsonl').read_text().splitlines()][20:]
            record={'status':'PASS','backend':backend,'profile':profile,'prefix':prefix,'expected':expected,
                    'rows':rows,'wall_seconds':time.perf_counter()-started}
            (out/'stage-record.json').write_text(json.dumps(record,indent=2)+'\n')
            if stage=='record':
                print(json.dumps({key:value for key,value in record.items() if key!='rows'},indent=2),flush=True)
                return record
        else:
            record=json.loads((out/'stage-record.json').read_text())
            assert record['status']=='PASS' and record['backend']==backend and record['profile']==profile
            prefix=record['prefix'];expected=record['expected'];rows=record['rows']
        prefix_path=Path(prefix['checkpoint'])
        envelope,identity=load_checkpoint(prefix_path)
        state=envelope['payload']
        assert state['vision_previous'] is not None
        if profile=='diagnostic': assert np.any(state['neural']['ever'])
        else:
            assert state['continuation']['direct_batches']==0
            assert not state['continuation']['sweep']
        proc,sid=start(out/'restored',backend,profile,prefix_path);owned.append(proc)
        status=request_control(sid,'status');assert status['paused'] and status['exchange']==20
        request_control(sid,'step',8)
        actual=request_control(sid,'checkpoint')
        request_control(sid,'stop');proc.wait(timeout=10);assert proc.returncode==0
        replay=[json.loads(row) for row in (out/'restored/telemetry.jsonl').read_text().splitlines()]
        assert rows==replay,'restored visual/neural/body telemetry differs'
        expected_state,_=load_checkpoint(Path(expected['checkpoint']),identity)
        actual_state,_=load_checkpoint(Path(actual['checkpoint']),identity)
        assert tree_hash(expected_state)==tree_hash(actual_state),'final continuation state differs'
        # Integrity and compatibility rejection use the same public loader, no live mutations.
        corrupted=dict(envelope,payload_sha256='0'*64)
        save_checkpoint(out/'corrupted.npz',corrupted,identity)
        try:load(out/'corrupted.npz',identity)
        except ValueError:pass
        else:raise AssertionError('corrupt checkpoint accepted')
        try:load(prefix_path,dict(identity,profile='other'))
        except ValueError:pass
        else:raise AssertionError('incompatible profile accepted')
        result={'status':'PASS','backend':backend,'profile':profile,'exchanges':8,
            'prefix_exchanges':20,'fresh_runner_process':True,'exact_telemetry':True,
            'exact_full_state':True,'visual_history_preserved':True,
            'nonzero_neural_state':bool(np.any(state['neural']['ever'])),
            'integrity_and_identity_rejection':True,'prefix':prefix,'expected':expected,'actual':actual,
            'wall_seconds':time.perf_counter()-started,'record_stage_wall_seconds':record['wall_seconds'],
            'separate_invocations':stage=='restore',
            'source_sha256':digest(Path(__file__)),
            'scope':'actual headless single-fly loop; eye images, CPU full body, decoder, interventions and all neural arrays',
            'unverified':['native viewer camera/UI continuity','biological memory','training efficacy']}
        (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2),flush=True)
        return result
    finally:
        for proc in owned:stop_process(proc)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--backend',choices=('cpu','gpu'),default='cpu')
    parser.add_argument('--profile',choices=('natural','diagnostic'),default='diagnostic')
    parser.add_argument('--stage',choices=('both','record','restore'),default='both')
    args=parser.parse_args()
    verify(args.out,args.backend,args.profile,args.stage)
