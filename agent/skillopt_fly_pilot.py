"""Bounded SkillOpt patch-engine bridge with recorded embodied neural activity.

One proposal is supplied by the current assistant, not an autonomous optimizer
backend. CPU shadow records full neural state; GPU outputs are checked each tick.
"""
import argparse
import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from agent.neural_connectome import CPUBrain, Worker, ASSETS, ANNOTATIONS, Readout, digest
from agent.neural_single import EthologyArena, Environment, Organism, encode, step_signed
from agent.odor_memory_gate import ABSENT, GAITS, scenes
from agent.parallel_learning_trial import fit
from fly_arena.behavior import BehaviorDecision

ROOT=Path('.work/agent/skillopt-pilot')
BINARY=Path('.work/agent/ten-fly/bin/worker-c1-optimized')


def prepare():
    spec=importlib.util.spec_from_file_location('archived_skillopt_patch',ROOT/'skillopt_skill.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    initial='# Fly training supervisor\n\ntraining_examples: 2\n\nFit the decoder only from recorded physical ingestion rewards.\n'
    patch={'edits':[{'op':'replace','target':'training_examples: 2','content':'training_examples: 10'}]}
    candidate,report=module.apply_patch_with_report(initial,patch)
    assert report[0]['status']=='applied_replace'
    (ROOT/'initial_skill.md').write_text(initial);(ROOT/'candidate_skill.md').write_text(candidate)
    (ROOT/'patch.json').write_text(json.dumps({'proposal_source':'current assistant','patch':patch,'application':report},indent=2)+'\n')
    data=np.load('.work/agent/spectator-learning/learning/seed-11-c1/samples.npz')
    for name,text in [('baseline',initial),('candidate',candidate)]:
        line=next(line for line in text.splitlines() if line.startswith('training_examples:'))
        count=int(line.split(':')[1]);assert count in (2,10)
        fit(data['features'][:count],data['rewards'][:count],ROOT/f'{name}.npz')
    protocol={'wall_runtime_seconds':150,'selection':scenes(range(12000,12003)),
        'untouched_final_test':scenes(range(13000,13003)),
        'proposal':'replay ten existing reward-labelled training layouts instead of two',
        'optimizer':'archived SkillOpt patch engine; current assistant supplied one edit',
        'gate':'strict higher success on completed matched selection pairs, at least two pairs; no partial episodes',
        'success_ingested':.05,'dt_s':.01,'physics_dt_s':.0001,'neural_ticks_per_exchange':4,
        'direct_stimulation':False,'graph_weights_modified':False,
        'neuron_recording':'full CPU shadow states; GPU feature and count parity each exchange, not full GPU-state readback',
        'baseline_sha256':digest(ROOT/'baseline.npz'),'candidate_sha256':digest(ROOT/'candidate.npz'),
        'training_samples_sha256':digest(Path('.work/agent/spectator-learning/learning/seed-11-c1/samples.npz'))}
    (ROOT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    before,after=Readout(ROOT/'baseline.npz'),Readout(ROOT/'candidate.npz')
    descending=np.load(ASSETS/'ports.npz')['descending']
    changed=np.any(before.weights[:-1]!=after.weights[:-1],axis=1)
    affected=np.unique(descending[np.flatnonzero(changed)%len(descending)])
    np.save(ROOT/'decoder_feature_neurons.npy',affected)
    print(json.dumps({'status':'PASS','patch_engine':'actual archived source','decoder_feature_neurons':len(affected)}))


def run(preflight=False):
    protocol=json.loads((ROOT/'protocol.json').read_text())
    output=ROOT/('cpu-preflight' if preflight else 'session')
    output.mkdir(exist_ok=False)
    brain=CPUBrain(flies=1)
    worker=None if preflight else Worker(binary=BINARY,flies=1)
    selection=protocol['selection'];scene=selection[0]
    arena=EthologyArena(seed=0,food_patches=[scene['food']],eye_size=16)
    arena.eye_renderer.close();initial=arena.get_state()
    import mujoco
    renderer=mujoco.Renderer(arena.sim.mj_model,width=640,height=480)
    camera=mujoco.MjvCamera();camera.distance=7;camera.azimuth=125;camera.elevation=-45;camera.lookat[:]=arena.position
    encoder=subprocess.Popen(['/usr/bin/ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo',
        '-pixel_format','rgb24','-video_size','640x480','-framerate','4','-i','pipe:0','-an',
        '-c:v','libx264','-threads','2','-preset','veryfast','-pix_fmt','yuv420p',str(output/'session.mp4')],
        stdin=subprocess.PIPE,stderr=(output/'video.log').open('w'))
    ports=brain.ports;union_fired=np.zeros(139255,bool);union_voltage=np.zeros(139255,bool);union_input=np.zeros(139255,bool)
    packed_spikes=[];packed_ever=[];packed_voltage=[];input_history=[];snapshots=[];episode_summaries=[];errors=[]
    start=time.monotonic();deadline=start+(2 if preflight else 150);next_frame=start;frame_count=0;exchanges=0
    graph_hash=digest(Path('data/weights.bin'))
    try:
        with (output/'trajectory.jsonl').open('w',buffering=1) as trace:
            for index in range(100):
                scene=selection[index%len(selection)]
                arms=['baseline','candidate'] if index%2==0 else ['candidate','baseline']
                for arm in arms:
                    if time.monotonic()>=deadline:break
                    food=scene['food'];state=copy.deepcopy(initial);state['food_patches']=[food]
                    geom=arena.sim.mj_model.geom('food_0');geom.pos[:]=[food['x'],food['y'],food['surface_z']-.005]
                    geom.size[:2]=[food['radius'],.005];arena.set_state(state)
                    environment=Environment([food],seed=scene['seed']);organism=Organism({})
                    model=Readout(ROOT/f'{arm}.npz');brain.reset()
                    if worker:worker.reset()
                    minimum=1.;steps=0;gait=[0.,0.]
                    for step in range(10 if preflight else 138):
                        if not preflight and time.monotonic()>=deadline:break
                        inputs=encode(environment.sense(arena,organism),ABSENT)
                        reference=brain.step(np.asarray(inputs)[None])
                        if worker:
                            gpu=worker.step(np.asarray(inputs)[None])
                            error=float(np.max(np.abs(np.asarray(gpu['features'])-reference['features'])))
                            assert error<=1e-5 and gpu['counts']==reference['counts']
                            errors.append(error)
                        if step==7:
                            prediction=model.predict(np.asarray(reference['features']))[0]
                            gait=GAITS['left'] if prediction[1]>prediction[0] else GAITS['right']
                        action='IDLE' if step<8 else 'WALK' if step<58 else 'FEED'
                        events=step_signed(arena,BehaviorDecision(action),gait if action=='WALK' else [0.,0.])
                        environment.apply_physical_events(events,organism,.01);organism.advance(.01,action)
                        assert np.isfinite(arena.sim.mj_data.qvel).all();minimum=min(minimum,float(arena.upright));assert minimum>=.85
                        fired=brain.spikes[:,0]!=0;voltage=np.abs(brain.v[:,0])>1e-7;driven=(ports@np.asarray(inputs))>0
                        union_fired|=brain.ever[:,0];union_voltage|=voltage;union_input|=driven
                        packed_spikes.append(np.packbits(fired));packed_ever.append(np.packbits(brain.ever[:,0]))
                        packed_voltage.append(np.packbits(voltage));input_history.append(inputs)
                        if exchanges%25==0:snapshots.append((exchanges,brain.v[:,0].copy(),brain.trace[:,0].copy()))
                        now=time.monotonic()
                        if now>=next_frame:
                            renderer.update_scene(arena.sim.mj_data,camera=camera);frame=renderer.render()
                            while next_frame<=now:
                                encoder.stdin.write(frame.tobytes());frame_count+=1;next_frame+=.25
                        trace.write(json.dumps({'exchange':exchanges,'wall_s':now-start,'pair_id':index,'scene_id':scene['id'],'arm':arm,
                            'step':step,'action':action,'inputs':inputs.tolist(),'fired':int(fired.sum()),
                            'recruited_episode':int(brain.ever[:,0].sum()),'position':arena.position.tolist(),
                            'upright':float(arena.upright),'ingested':organism.state.ingested_total,
                            'video_frames_written':frame_count})+'\n')
                        exchanges+=1;steps+=1
                    episode_summaries.append({'pair_id':index,'scene_id':scene['id'],'arm':arm,'steps':steps,
                        'complete':steps==(10 if preflight else 138),'ingested':organism.state.ingested_total,'minimum_upright':minimum})
                    print(json.dumps(episode_summaries[-1]),flush=True)
                    if preflight:break
                if preflight or time.monotonic()>=deadline:break
        elapsed=time.monotonic()-start
        np.savez_compressed(output/'neural-history.npz',spikes_packbits=packed_spikes,ever_packbits=packed_ever,
                            voltage_nonzero_packbits=packed_voltage,
                            inputs=input_history,neurons=139255,snapshot_exchange=[x[0] for x in snapshots],
                            voltage_snapshots=[x[1] for x in snapshots],trace_snapshots=[x[2] for x in snapshots])
        roots=np.load('data/proofread_root_ids_783.npy')
        rows=list(csv.DictReader(ANNOTATIONS.open(),delimiter='\t'));labels={int(row['root_id']):row for row in rows}
        affected=np.load(ROOT/'decoder_feature_neurons.npy');decoder=set(map(int,affected))
        with (output/'touched-neurons.csv').open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['index','flywire_root_id','input_driven','fired','nonzero_voltage',
                                                     'decoder_feature_changed','cell_class','super_class','side'])
            for i in np.flatnonzero(union_fired|union_voltage|union_input|np.isin(np.arange(139255),affected)):
                annotation=labels.get(int(roots[i]),{})
                writer.writerow([i,int(roots[i]),int(union_input[i]),int(union_fired[i]),int(union_voltage[i]),int(i in decoder),
                                 annotation.get('cell_class',''),annotation.get('super_class',''),annotation.get('side','')])
        pairs=[]
        for pair_id in sorted({row['pair_id'] for row in episode_summaries}):
            pair=[row for row in episode_summaries if row['pair_id']==pair_id and row['complete']]
            if len(pair)==2:pairs.append({row['arm']:int(row['ingested']>=.05) for row in pair})
        accepted=len(pairs)>=2 and sum(x['candidate'] for x in pairs)>sum(x['baseline'] for x in pairs)
        if not preflight:
            (ROOT/'best_skill.md').write_text((ROOT/('candidate_skill.md' if accepted else 'initial_skill.md')).read_text())
        assert digest(Path('data/weights.bin'))==graph_hash
        report={'status':'PASS','wall_runtime_seconds':elapsed,'exchanges':exchanges,'episodes':episode_summaries,
            'completed_matched_pairs':len(pairs),'baseline_successes':sum(x['baseline'] for x in pairs),
            'candidate_successes':sum(x['candidate'] for x in pairs),'candidate_accepted':accepted,
            'input_driven_neurons':int(union_input.sum()),'fired_neurons':int(union_fired.sum()),
            'nonzero_voltage_neurons':int(union_voltage.sum()),'decoder_feature_neurons':len(affected),
            'gpu_feature_error':max(errors) if errors else None,'direct_stimulation':False,'graph_weights_unchanged':True,
            'video_frames':frame_count,'scope':'assistant-proposed SkillOpt patch bridge, CPU-shadow neuron records, one-decision selection pilot'}
        (output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    finally:
        encoder.stdin.close();code=encoder.wait(timeout=20);assert code==0,code
        renderer.close();arena.close()
        if worker:worker.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('prepare','preflight','run'));a=p.parse_args()
    prepare() if a.stage=='prepare' else run(a.stage=='preflight')
