"""CPU-only split-reuse, artifact preservation and concurrent consumption checks."""
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import uuid
from agent.experiment_ledger import consume_final, ROOT


def competing(out,ledger,queue):
    try:
        consume_final(out,'fixture:race',[21,22],'b'*64,ledger=ledger)
        queue.put('claimed')
    except ValueError:queue.put('rejected')


def verify():
    out=ROOT/'.work/agent/control-audit'/('ledger-check-'+uuid.uuid4().hex)
    out.mkdir(parents=True);ledger=out/'ledger.jsonl'
    def reject(function):
        try:function()
        except ValueError:return
        raise AssertionError('expected rejection')
    first=consume_final(out/'first','fixture:A',[10,11],'a'*64,ledger=ledger)
    reject(lambda:consume_final(out/'second','fixture:A',[11,12],'a'*64,ledger=ledger))
    reject(lambda:consume_final(out/'first','fixture:A',[30,31],'a'*64,ledger=ledger))
    reject(lambda:consume_final(out/'bad','fixture:A',[10,10],'a'*64,ledger=ledger))
    reject(lambda:consume_final(out/'bad','fixture:A',[-1],'a'*64,ledger=ledger))
    second=consume_final(out/'second','fixture:A',[12,13],'b'*64,ledger=ledger)
    assert first['experiment_id']!=second['experiment_id']
    historical=consume_final(out/'historical','fixture:A',[10,11],'c'*64,ledger=ledger,historical=True)
    assert historical['historical_overlap_frames']==2 and historical['status']=='historical_consumed'
    assert consume_final(out/'historical','fixture:A',[10,11],'c'*64,ledger=ledger,historical=True)==historical
    ctx=mp.get_context('spawn');queue=ctx.Queue()
    processes=[ctx.Process(target=competing,args=(out/f'race-{i}',ledger,queue)) for i in range(2)]
    try:
        for proc in processes:proc.start()
        for proc in processes:proc.join(timeout=10);assert proc.exitcode==0
        assert sorted(queue.get(timeout=2) for _ in processes)==['claimed','rejected']
    finally:
        for proc in processes:
            if proc.is_alive():proc.terminate();proc.join(timeout=3)
        queue.close()
    from agent.calcium_validation import prepare,record,evaluate
    saved=ROOT/'.work/agent/calcium-validation-source-v2'
    files=[saved/p for p in ('protocol.npz','simulation.npz','evaluation.json','frozen-observer.npz')]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    reject(lambda:prepare(saved));reject(lambda:record(saved));reject(lambda:evaluate(saved))
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p)] for p in files)
    from agent.launch_neural_flies import desktop_environment
    import subprocess
    natural=ROOT/'.work/agent/control-audit/natural-profile'
    before=(natural/'report.json').read_bytes()
    duplicate=subprocess.run([str(ROOT/'.work/fly-research/arena-venv/bin/python'),'-m','agent.neural_single',
        '--backend','cpu','--profile','natural','--headless','--seconds','.01','--output',str(natural)],
        cwd=ROOT,env=dict(desktop_environment(),LIBGL_ALWAYS_SOFTWARE='1'),capture_output=True,text=True)
    assert duplicate.returncode!=0 and 'existing run receipts' in duplicate.stderr
    assert (natural/'report.json').read_bytes()==before
    # Exercise final-freeze -> ledger -> scoring with synthetic labels only.
    import numpy as np
    import functools
    import agent.calcium_validation as calcium
    fixture=out/'synthetic-evaluation';fixture.mkdir()
    rng=np.random.default_rng(4);n=144;trial=np.repeat(np.arange(6),24);within=np.tile(np.arange(24),6)
    np.savez(fixture/'protocol.npz',source_indices=np.arange(n),trial=trial,calcium=rng.normal(size=(n,3)),
        train=(trial<3)&(within>=8)&(within<15),selection=(trial<3)&(within>=19),
        final=(trial>=3)&(within>=8),heading=np.sin(np.arange(n)),proxy_contrast=np.zeros(n),
        roi_indices=np.arange(3),source_stimulus=False)
    np.savez(fixture/'simulation.npz',source_indices=np.arange(n),spike_rate=rng.random((n,47)),
        timestamps=np.arange(n)*.1,trial=trial,epg_root_ids=np.arange(47))
    (fixture/'protocol.json').write_text(json.dumps({'source_nwb':'synthetic-mechanics.nwb',
        'data_scope':'synthetic mechanics fixture; no biological observations'}))
    original=calcium.consume_final
    try:
        calcium.consume_final=functools.partial(original,ledger=out/'synthetic-ledger.jsonl')
        evaluation=calcium.evaluate(fixture)
        assert evaluation['status']=='PASS' and evaluation['data_scope'].startswith('synthetic')
        consumption=json.loads((fixture/'evaluation-consumption.json').read_text())
        assert consumption['frozen_observer_sha256']==hashlib.sha256((fixture/'frozen-observer.npz').read_bytes()).hexdigest()
        reject(lambda:calcium.evaluate(fixture))
    finally:calcium.consume_final=original
    recording=out/'synthetic-recording';recording.mkdir()
    np.savez(recording/'protocol.npz',source_indices=[0],source_timestamps=[.01],heading=[0.],trial=[0],
        proxy_contrast=[1.],contrast_code=[3],source_stimulus=True,cue_heading_sign=-1,neural_tick_s=.0025)
    recorded=subprocess.run([str(ROOT/'.work/fly-research/arena-venv/bin/python'),'-m','agent.calcium_validation',
        'record','--output',str(recording)],cwd=ROOT,env=dict(desktop_environment(),LIBGL_ALWAYS_SOFTWARE='1'),
        text=True,capture_output=True,check=True)
    (out/'record-stage.log').write_text(recorded.stdout+recorded.stderr)
    assert json.loads((recording/'recording.json').read_text())['status']=='PASS'
    result={'status':'PASS','output':str(out),'exact_final_overlap_rejected':True,
        'fresh_disjoint_frames_accepted':True,'historical_import_idempotent':True,
        'concurrent_same_frames':'one claim, one rejection','completed_calcium_artifacts_unchanged':True,
        'live_receipt_overwrite_rejected':True,'real_final_targets_scored_or_fitted':False,'synthetic_freeze_consume_score_pipeline':'PASS','synthetic_full_body_record_stage':'PASS',
        'scope':'mechanics guards, not independent biological data validation'}
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    (ROOT/'.work/agent/control-audit/ledger-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':verify()
