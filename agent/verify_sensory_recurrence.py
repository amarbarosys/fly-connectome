"""Paired fixed-state recurrence ablation and visual spatial-information falsifier."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from agent.calcium_validation import cue_images, epg_population
from agent.neural_connectome import CPUBrain, ROOT, ASSETS, digest
from agent.vision import EyeVision


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    protocol={'test_updates':480,'warm_frames':8,'warm_updates_per_frame':40,
        'model_update_interval_s_assumption':.0025,'threshold':1.,'decay':.85,
        'conditions':'identical saved warm state; original/zero recurrent edges crossed with dim/zero drive',
        'scope':'synthetic mechanics and input distinguishability, not calcium fitting or biological validation',
        'source_hashes':{str(p.relative_to(ROOT)):digest(p) for p in [ROOT/'agent/verify_sensory_recurrence.py',
            ROOT/'agent/neural_connectome.py',ROOT/'agent/vision.py',ROOT/'agent/calcium_validation.py',
            ASSETS/'ports.npz',ROOT/'data/weights.bin',ROOT/'data/row_ptr.bin',ROOT/'data/col_idx.bin']}}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    start=time.monotonic();brain=CPUBrain(flies=1);vision=EyeVision();_,epg=epg_population()
    for i in range(8):
        visual,_,_=vision.step(cue_images(.6*np.sin(i*.8),1.,width=48,source_stimulus=True))
        inputs=np.zeros((1,8),np.float32);inputs[0,6:]=visual
        brain.step(inputs,ticks=40)
    checkpoint={name:getattr(brain,name).copy() for name in ('v','spikes','trace','ever')}
    assert checkpoint['spikes'].sum()>0
    np.savez_compressed(out/'warm-state.npz',**checkpoint)
    state_hash=hashlib.sha256(b''.join(v.tobytes() for v in checkpoint.values())).hexdigest()
    graph=brain.graph.copy();summaries={};rows=[]
    for connected in (True,False):
        for condition in ('dim','zero'):
            brain.graph=graph.copy()
            if not connected:brain.graph.data[:]=0
            for name,value in checkpoint.items():setattr(brain,name,value.copy())
            assert hashlib.sha256(b''.join(getattr(brain,k).tobytes() for k in checkpoint)).hexdigest()==state_hash
            key=('connected_' if connected else 'disconnected_')+condition
            visual,_,_=EyeVision().step(cue_images(0.,1/15,width=48,source_stimulus=True))
            inputs=np.zeros((1,8),np.float32)
            if condition=='dim':inputs[0,6:]=visual
            results=[]
            for tick in range(480):
                brain.step(inputs,ticks=1)
                assert np.isfinite(brain.v).all() and np.isfinite(brain.trace).all()
                row={'case':key,'tick':tick+1,'spikes':int(brain.spikes.sum()),
                    'epg_spikes':int(brain.spikes[epg].sum()),'voltage_max':float(brain.v.max())}
                rows.append(row);results.append(row)
            summaries[key]={'initial_state_sha256':state_hash,'spike_events':sum(r['spikes'] for r in results),
                'epg_spike_events':sum(r['epg_spikes'] for r in results),
                'last_40_updates_spike_events':sum(r['spikes'] for r in results[-40:]),
                'last_40_updates_epg_spike_events':sum(r['epg_spikes'] for r in results[-40:]),
                'final_voltage_max':results[-1]['voltage_max']}
    bearings=[]
    for cell in (5,10,20):
        angle=(cell+.5)*np.pi/48
        image=cue_images(angle,1.,width=48,source_stimulus=True)
        drive,threat,info=EyeVision().step(image)
        bearings.append({'bearing_rad':float(angle),'visual_drive':drive.tolist(),
            'threat':threat.tolist(),'image_sha256':info['image_sha256']})
    images_differ=len({r['image_sha256'] for r in bearings})==len(bearings)
    same_static_drive=all(np.array_equal(bearings[0]['visual_drive'],r['visual_drive']) for r in bearings[1:])
    assert images_differ and same_static_drive
    assert summaries['disconnected_zero']['spike_events']==0
    assert summaries['disconnected_dim']['last_40_updates_spike_events']==0
    assert summaries['connected_zero']['last_40_updates_spike_events']>0
    assert digest(ROOT/'data/weights.bin')==protocol['source_hashes']['data/weights.bin']
    report={'status':'PASS','mechanics_gate':'PASS','biological_validity':'UNVERIFIED',
        'checkpoint_sha256':state_hash,'cases':summaries,'static_bearing_examples':bearings,
        'distinct_retinal_images_identical_static_drive':same_static_drive,
        'connected_dim_to_zero_total_spike_ratio':summaries['connected_dim']['spike_events']/summaries['connected_zero']['spike_events'],
        'graph_file_unchanged':True,'active_model_parameters_changed':False,'calcium_targets_used':False,
        'wall_seconds':time.monotonic()-start,
        'conclusions':['In this fixture, recurrent edges sustain warm-state firing without sensory input.',
            'Three distinct static cue bearings within one hemisphere collapse to identical coarse visual ports.',
            'Warm firing alone does not establish dim-cue responsiveness or directional memory.']}
    with (out/'telemetry.jsonl').open('w') as stream:
        for row in rows:stream.write(json.dumps(row)+'\n')
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(run(args.output)))
