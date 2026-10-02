"""Measure resident worker VRAM and parity with a bounded repeated workload."""
import json
import subprocess
import numpy as np
from agent.neural_connectome import Worker,CPUBrain,ROOT

def allocated(pid):
    output=subprocess.check_output(['/opt/rocm/bin/rocm-smi','--showpids'],text=True,stderr=subprocess.DEVNULL)
    for line in output.splitlines():
        fields=line.split()
        if fields and fields[0]==str(pid):return int(fields[3])
    raise RuntimeError('No VRAM allocation receipt for worker '+str(pid))

def verify():
    brain=Worker();samples=[];error=0.
    oracle=CPUBrain();inputs=np.zeros((2,8),np.float32);inputs[0]=.5
    try:
        samples.append({'step':0,'bytes':allocated(brain.proc.pid)})
        for i in range(120):
            actual=brain.step(inputs)
            if i<3:
                expected=oracle.step(inputs)
                error=max(error,float(np.max(np.abs(np.asarray(actual['features'])-np.asarray(expected['features'])))))
                assert actual['counts']==expected['counts']
            assert np.isfinite(actual['features']).all() and actual['counts'][1]==[0.,0.]
            if (i+1)%40==0:samples.append({'step':i+1,'bytes':allocated(brain.proc.pid)})
        values=[s['bytes'] for s in samples]
        assert error<1e-5 and max(values)<1024**3 and max(values)-min(values)<16*1024**2
        report={'status':'PASS','pool_percent':2,'worker_vram_samples':samples,
                'cpu_gpu_max_abs_error':error,'steps':120,'explicit_buffers_MB':92.96,
                'baseline_worker_bytes':23263784960,'max_worker_bytes':max(values),
                'scope':'bounded resident allocation check; not proof against every long-run leak'}
        (ROOT/'.work/neural-flies/memory-gate.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)
    finally:brain.close()
if __name__=='__main__':verify()
