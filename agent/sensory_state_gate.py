"""CPU mechanics falsifier for dim input and initial-state dependence; no biological fit."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from agent.calcium_validation import cue_images, epg_population
from agent.neural_connectome import CPUBrain, ROOT, ASSETS, digest
from agent.vision import EyeVision


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    cases=[('cold_zero',False,False,0.,True),('cold_dim',False,False,1/15,False),
           ('cold_bright',False,False,1.,False),('warm_bright_dim',True,False,1/15,False),
           ('warm_bright_reset_dim',True,True,1/15,False),('warm_bright_zero',True,False,0.,True)]
    protocol={'cases':cases,'prelude_frames':8,'test_frames':12,'updates_per_frame':40,
        'nominal_update_seconds':.0025,'image_width_per_eye':48,'stripe_degrees':7.5,
        'heading_radians':'0.6*sin(frame*0.8), identical test sequence across cases',
        'scope':'synthetic input and state mechanics only; no physical body, calcium target, fitting or parameter change',
        'source_hashes':{str(p.relative_to(ROOT)):digest(p) for p in
            [ROOT/'agent/sensory_state_gate.py',ROOT/'agent/neural_connectome.py',ROOT/'agent/vision.py',
             ROOT/'agent/calcium_validation.py',ASSETS/'ports.npz',ROOT/'data/weights.bin',
             ROOT/'data/row_ptr.bin',ROOT/'data/col_idx.bin']}}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    _,epg=epg_population();brain=CPUBrain(flies=1);rows=[];summaries={};start=time.monotonic()
    for name,warm,reset,contrast,blind in cases:
        brain.reset();vision=EyeVision();prelude_spikes=0
        def frame(index,intensity,disabled):
            visual,_,info=vision.step(cue_images(.6*np.sin(index*.8),intensity,width=48,source_stimulus=True),blind=disabled)
            sensory=np.zeros((1,8),np.float32);sensory[0,6:]=visual
            total=epg_spikes=0
            for _ in range(40):
                brain.step(sensory,ticks=1)
                total+=int(brain.spikes.sum());epg_spikes+=int(brain.spikes[epg].sum())
                if not all(np.isfinite(x).all() for x in (brain.v,brain.trace)):
                    raise AssertionError('nonfinite state')
            drive=(brain.ports@sensory.T)*np.float32(.65)
            return {'visual':visual.tolist(),'max_drive':float(drive.max()),
                'no_spike_leaky_ceiling':float(drive.max())/.15,'total_spikes':total,
                'epg_spikes':epg_spikes,'voltage_max':float(brain.v.max()),
                'voltage_min':float(brain.v.min()),'ever_fired':int(brain.ever.sum()),
                'descending_feature_norm':float(np.linalg.norm(np.r_[brain.v[brain.descending,0],brain.trace[brain.descending,0]])),
                'motion':info['motion'],'image_sha256':info['image_sha256']}
        if warm:
            for i in range(8):prelude_spikes+=frame(i,1.,False)['total_spikes']
        if reset:brain.reset()  # Keep image history to isolate neural reset from vision reset.
        initial_ever=int(brain.ever.sum());initial_voltage_max=float(brain.v.max())
        results=[]
        for i in range(12):
            row=dict(frame(i,contrast,blind),case=name,frame=i);results.append(row);rows.append(row)
        summaries[name]={'prelude_spikes':prelude_spikes,'initial_ever_fired':initial_ever,
            'initial_voltage_max':initial_voltage_max,'test_spikes':sum(r['total_spikes'] for r in results),
            'test_epg_spikes':sum(r['epg_spikes'] for r in results),
            'max_no_spike_leaky_ceiling':max(r['no_spike_leaky_ceiling'] for r in results),
            'last_frame_spikes':results[-1]['total_spikes'],'last_frame_epg_spikes':results[-1]['epg_spikes'],
            'last_voltage_max':results[-1]['voltage_max']}
        print(name,json.dumps(summaries[name]),flush=True)
    assert summaries['cold_zero']['test_spikes']==0
    assert summaries['cold_dim']['max_no_spike_leaky_ceiling']<1
    assert summaries['cold_dim']['test_spikes']==0
    assert summaries['cold_bright']['test_spikes']>0
    assert summaries['warm_bright_dim']['prelude_spikes']>0
    assert summaries['warm_bright_reset_dim']['initial_ever_fired']==0
    report={'status':'PASS','mechanics_gate':'PASS','biological_calibration':'UNVERIFIED',
        'parameters_changed':False,'heldout_calcium_targets_used':False,'backend':'full-connectome CPU oracle',
        'neural_updates':40*(12*len(cases)+8*sum(c[1] for c in cases)),
        'nominal_test_seconds_per_case':1.2,'wall_seconds':time.monotonic()-start,'cases':summaries,
        'interpretation':'Cold dim silence is a sufficient reset-state drive bound. Warm versus reset conditions isolate initial-state and image-history sensitivity; firing is not biological validation.'}
    with (out/'telemetry.jsonl').open('w') as stream:
        for row in rows:stream.write(json.dumps(row,allow_nan=False)+'\n')
    (out/'verification.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(run(args.output)))
