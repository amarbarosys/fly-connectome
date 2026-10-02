"""Append-only final-frame consumption ledger. Fresh frames do not mean independent flies."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
LEDGER=ROOT/'.work/agent/experiment-ledger/final-consumption.jsonl'


def consume_final(out,dataset_id,frames,frozen_sha256,*,ledger=LEDGER,historical=False):
    out=Path(out).resolve();ledger=Path(ledger)
    if not dataset_id or not isinstance(dataset_id,str):raise ValueError('dataset_id required')
    if not re.fullmatch('[0-9a-f]{64}',frozen_sha256):raise ValueError('frozen observer hash required')
    if not frames or any(type(i) is not int or i<0 for i in frames) or len(set(frames))!=len(frames):
        raise ValueError('unique nonnegative source frame indices required')
    ledger.parent.mkdir(parents=True,exist_ok=True)
    with ledger.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        entries=[json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
        same=[e for e in entries if e['output']==str(out)]
        if same:
            if historical and same[0]['frozen_observer_sha256']==frozen_sha256:return same[0]
            raise ValueError('final evaluation already consumed for this output; use recorded receipt')
        old=set(i for e in entries if e['dataset_id']==dataset_id for i in e['final_source_frames'])
        overlap=old.intersection(frames)
        if overlap and not historical:
            raise ValueError(f'{len(overlap)} final frames already consumed; use a predeclared fresh split')
        entry={'experiment_id':uuid.uuid4().hex,'recorded_unix':time.time(),'output':str(out),
            'dataset_id':dataset_id,'ledger':str(ledger),'final_source_frames':sorted(frames),
            'frozen_observer_sha256':frozen_sha256,
            'status':'historical_consumed' if historical else 'consumed_before_final_scoring',
            'historical_overlap_frames':len(overlap) if historical else 0,
            'freshness_limit':'exact frame reuse guard; not independent animals or full-recording decontamination'}
        with ledger.open('a') as stream:
            stream.write(json.dumps(entry)+'\n');stream.flush();os.fsync(stream.fileno())
        return entry


def import_saved(out):
    """Record completed historical evaluations without recomputing or claiming fresh data."""
    import numpy as np
    out=Path(out).resolve()
    if not out.is_relative_to(ROOT):raise ValueError('saved evaluation must be inside project')
    p=json.loads((out/'protocol.json').read_text());e=json.loads((out/'evaluation.json').read_text())
    if hashlib.sha256((out/'frozen-observer.npz').read_bytes()).hexdigest()!=e['frozen_observer_sha256']:
        raise ValueError('historical frozen observer hash mismatch')
    if hashlib.sha256((out/'protocol.npz').read_bytes()).hexdigest()!=e['protocol_sha256']:
        raise ValueError('historical protocol hash mismatch')
    with np.load(out/'protocol.npz') as arrays:
        frames=arrays['source_indices'][arrays['final']].astype(int).tolist()
    return consume_final(out,'nwb:'+Path(p['source_nwb']).name,frames,e['frozen_observer_sha256'],historical=True)
