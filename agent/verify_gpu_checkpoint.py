"""CPU buffer preflight and full-array GPU neural restart gate."""
import argparse
import io
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from agent import neural_io
from agent.neural_connectome import ROOT, Worker, CPUBrain, digest


def preflight():
    base=ROOT/'.work/neural-flies/worker-state'
    base.mkdir(parents=True,exist_ok=True)
    state={name:np.zeros((139255,2),bool if name=='ever' else np.float32)
           for name in ('v','spikes','trace','ever')}
    state['v'][3,0]=.4;state['spikes'][7,1]=1;state['ever'][7,1]=True;state['trace'][7,1]=.15
    neural_io.validate_state(state,2)
    buffers={name:np.ascontiguousarray(value.T,dtype=np.float32) for name,value in state.items()}
    with tempfile.TemporaryDirectory(dir=base) as directory:
        path=Path(directory)/'state.npz'
        neural_io.state_buffers({'op':'export_state','path':str(path)},
            *(buffers[name].ctypes.data for name in ('v','spikes','trace','ever')),2)
        with patch('sys.stdin',io.StringIO(json.dumps({'op':'import_state','path':str(path)})+'\n')):
            request=neural_io.receive(2)
        for value in buffers.values():value[:]=0
        neural_io.state_buffers(request,*(buffers[name].ctypes.data for name in ('v','spikes','trace','ever')),2)
        assert all(np.array_equal(buffers[name].T,state[name]) for name in state)
        wrong=dict(state,v=np.zeros((1,2),np.float32))
        try:neural_io.validate_state(wrong,2)
        except ValueError:pass
        else:raise AssertionError('invalid state accepted')
        try:neural_io.state_path(ROOT/'outside-state.npz')
        except ValueError:pass
        else:raise AssertionError('outside transfer path accepted')
    return {'status':'PASS','host_transfer_roundtrip':True,'invalid_shape_and_path_rejected':True}


def verify(out):
    out.mkdir(parents=True,exist_ok=False)
    cpu_preflight=preflight()
    worker=None
    try:
        worker=Worker()
        inputs=np.array([[.2,.7,.1,.2,.1,.2,.15,.5],[0]*8],np.float32)
        gpu=worker.step(inputs);cpu=CPUBrain().step(inputs)
        parity=float(np.max(np.abs(np.asarray(gpu['features'])-np.asarray(cpu['features']))))
        assert parity<2e-4 and np.allclose(gpu['counts'],cpu['counts'],atol=1)
        assert gpu['counts'][1]==[0,0]
        inputs[1]=[.7,.1,.3,.2,.5,.1,.4,.3]
        for _ in range(7):worker.step(inputs)
        saved=worker.export_state()
        assert np.any(saved['ever'][:,0]) and np.any(saved['ever'][:,1])
        targets=worker.step(inputs)['inactive'][:16]
        # Return to the saved state before producing the continuation reference.
        worker.import_state(saved)
        expected=worker.step(inputs,direct=targets)
        final=worker.export_state()
        worker.close();worker=None
        worker=Worker()
        worker.import_state(saved)
        restored=worker.export_state()
        assert all(np.array_equal(saved[name],restored[name]) for name in saved)
        actual=worker.step(inputs,direct=targets)
        continued=worker.export_state()
        assert expected==actual
        assert all(np.array_equal(final[name],continued[name]) for name in final)
        before=worker.export_state()
        try:worker.import_state(dict(saved,v=np.zeros((1,2),np.float32)))
        except ValueError:pass
        else:raise AssertionError('invalid import accepted')
        assert all(np.array_equal(before[name],worker.export_state()[name]) for name in before)
        # Reset must clear the host recruitment mirror used for direct-target filtering.
        recruited=int(np.flatnonzero(before['ever'][:,0])[0])
        worker.reset();worker.step(np.zeros((2,8),np.float32),direct=[recruited])
        assert worker.export_state()['ever'][recruited,0]
        np.savez_compressed(out/'saved-neural.npz',**saved)
        result={'status':'PASS','cpu_preflight':cpu_preflight,'cpu_gpu_feature_max_abs_error':parity,
            'both_lanes_nonzero':True,'fresh_gpu_worker':True,'all_arrays_exact_roundtrip':True,
            'continued_arrays_features_counts_inactive_exact':True,'invalid_import_preserves_state':True,
            'reset_host_recruitment_mirror':True,'worker_sha256':digest(Path(worker.proc.args[0])),
            'saved_state_sha256':digest(out/'saved-neural.npz'),'scope':'GPU neural engineering continuity, no biology'}
        (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2),flush=True)
    finally:
        if worker:worker.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry',action='store_true');parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    if args.dry:print(json.dumps(preflight()))
    elif args.out:verify(args.out)
    else:parser.error('--out required for GPU run')
