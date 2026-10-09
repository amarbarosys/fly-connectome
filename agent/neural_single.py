"""One physical fly, sensory activation sweep and resident GPU neural state.

The second lane of the existing GPU worker remains zero and has no body.
Sensory encodings and decoder motor primitives are experimental.
"""
import argparse
import hashlib
import json
from pathlib import Path
from queue import SimpleQueue
from types import SimpleNamespace
import signal
import time
import numpy as np
from agent.neural_connectome import ROOT, ASSETS, CHANNELS, CPUBrain, Worker, Readout
from agent.neural_duel import MotorDecision, encode, _json_default
from agent.runtime_control import RuntimeControl
from agent import world_edit
from agent.desire import Desire
from agent.vision import EyeVision
from fly_arena.ethology_arena import EthologyArena
from fly_arena.environment import Environment
from fly_arena.organism import Organism
from fly_duel.duel import encounter_scene


def activation_audit(brain, out):
    """Reset each intervention, plus zero-input and combined-input controls."""
    rows=[]
    for channel in ['null',*CHANNELS,'combined']:
        brain.reset()
        inputs=np.zeros((2,8),np.float32)
        if channel=='combined':inputs[0]=.8
        elif channel!='null':inputs[0,CHANNELS.index(channel)]=.8
        for _ in range(16):result=brain.step(inputs)
        features=np.asarray(result['features'])
        row={'channel':channel,'input':inputs[0].tolist(),
             'active':int(result['counts'][0][0]),'recruited':int(result['counts'][0][1]),
             'descending_feature_nonzero':int(np.count_nonzero(features[0])),
             'descending_feature_norm':float(np.linalg.norm(features[0])),
             'unused_lane_recruited':int(result['counts'][1][1])}
        assert row['unused_lane_recruited']==0
        if channel=='null':assert row['recruited']==row['descending_feature_nonzero']==0
        else:assert row['recruited']>0
        rows.append(row)
    brain.reset()
    report={'status':'PASS','neural_ticks_per_intervention':64,'rows':rows,
            'scope':'all eight existing sensory ports exercised; not all-neuron recruitment or biological validation'}
    (out/'activation-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def step_signed(arena, decision, gait):
    """Use the installed controller's signed frequencies to back up and pivot."""
    from fly_arena.sensors import PhysicalEvents
    gait=np.asarray(gait,float)
    if gait.shape!=(2,) or not np.isfinite(gait).all():raise ValueError('invalid signed gait')
    arena.control=np.clip(gait,-.75,.75)
    arena.motor.select(decision)
    events=PhysicalEvents();arena.max_groom_contact_force=0.
    for _ in range(round(.01/arena.sim.timestep)):
        arena.update_looming(arena.physics_steps*arena.sim.timestep)
        if arena.physics_steps%arena.controller_stride==0:arena.motor.update(arena.control)
        arena.sim.step();arena.physics_steps+=1;arena.motor.elapsed+=arena.sim.timestep
        arena._accumulate_events(events)
    if not np.isfinite(arena.sim.mj_data.qpos).all():raise RuntimeError('nonfinite physics')
    return events


def wall_observation(arena):
    """Read physical wall contacts, not arena coordinates or hidden targets."""
    import mujoco
    model,data=arena.sim.mj_model,arena.sim.mj_data
    normal=np.zeros(2);contacts=0
    for contact in data.contact[:data.ncon]:
        names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,int(g)) or '' for g in contact.geom]
        if names[0].startswith('wall_'):
            normal+=np.asarray(contact.frame[:2]);contacts+=1
        elif names[1].startswith('wall_'):
            normal-=np.asarray(contact.frame[:2]);contacts+=1
    if not contacts:return False,1.
    heading=arena.heading
    away=np.arctan2(normal[1],normal[0])
    error=np.arctan2(np.sin(away-heading),np.cos(away-heading))
    # Contact with a wall while already facing away is not a forward blockage.
    return bool(abs(error)>.6),float(np.sign(error) or 1.)


def reset_body_and_brain(arena, brain, initial_body, decision, time_s):
    """Reset the body/controller and neural state, preserving world and learning."""
    arena.set_state(initial_body)
    brain.reset()
    decision.__dict__.update(vars(MotorDecision()))
    decision.since = time_s
    current = arena.get_state()
    return {'body_integration_matches_initial': bool(np.array_equal(current['integration'], initial_body['integration'])),
            'physics_steps': arena.physics_steps, 'controller_reset': True,
            'decision': decision.current, 'neural_reset': True,
            'position_mm': arena.position.tolist()}


def run(out, backend='gpu', seconds=0., viewer=True, audit=True, profile='diagnostic', start_paused=False, restore_checkpoint=None):
    if profile not in ('natural','diagnostic'):raise ValueError('invalid launch profile')
    import mujoco
    import mujoco.viewer
    out=Path(out)
    if any((out/name).exists() for name in ('launch.json','report.json','telemetry.jsonl','events.jsonl')):
        raise ValueError('existing run receipts; choose a fresh output directory')
    out.mkdir(parents=True,exist_ok=True)
    brain=arena=view=controls=remote=None
    stopping=False;paused=start_paused;pending=SimpleQueue();exchange=0;step_remaining=0
    def stop(*_):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    t=0.;pulse_end=0.;pulse=np.zeros(8,np.float32);sweep_start=0.;sweep=profile=='diagnostic'
    reward_updates=0;resets=0;last=None;action='IDLE';error=None
    direct_mode=profile=='diagnostic';attempted=set();direct_batches=0;blind=False
    try:
        brain=Worker() if backend=='gpu' else CPUBrain()
        if audit and profile=='diagnostic' and restore_checkpoint is None:activation_audit(brain,out)
        cfg,_,_=encounter_scene(resources='scarce')
        environment=Environment(cfg['food'],cfg.get('environment',{}),seed=1)
        organism=Organism(cfg.get('organism',{}),initial=cfg.get('initial',{}))
        patches=[{'id':p.id,'x':p.x,'y':p.y,'radius':p.radius,'surface_z':p.surface_z} for p in environment.food]
        arena=EthologyArena(seed=1,food_patches=patches)
        model=Readout();decision=MotorDecision();desire=Desire();vision=EyeVision(spatial=True)
        vision.register_cameras(arena)
        initial_body=arena.get_state()
        absent=SimpleNamespace(body_contact=False,detected=False,bearing=0.,distance_mm=None)
        from agent import live_checkpoint
        checkpoint_identity=live_checkpoint.compatibility(arena,brain,backend,profile)
        if restore_checkpoint is not None:
            saved=live_checkpoint.load(restore_checkpoint,checkpoint_identity)
            initial_body,continuation=live_checkpoint.restore(saved,arena,brain,environment,organism,model,decision,desire,vision)
            t=continuation['t'];exchange=continuation['exchange'];pulse_end=continuation['pulse_end']
            pulse=continuation['pulse'];sweep_start=continuation['sweep_start'];sweep=continuation['sweep']
            reward_updates=continuation['reward_updates'];resets=continuation['resets'];last=continuation['last']
            action=continuation['action'];direct_mode=continuation['direct_mode'];attempted=set(continuation['attempted'])
            direct_batches=continuation['direct_batches'];blind=continuation['blind'];paused=True
            (out/'restore.json').write_text(json.dumps({'checkpoint':str(Path(restore_checkpoint).resolve()),
                'checkpoint_sha256':live_checkpoint.digest(restore_checkpoint),'exchange':exchange,'time_s':t,
                'paused':True,'status':'PASS'},indent=2)+'\n')
        if viewer:
            view=mujoco.viewer.launch_passive(arena.sim.mj_model,arena.sim.mj_data,
                                             show_left_ui=False,show_right_ui=False)
            with view.lock():
                view.cam.distance=7;view.cam.azimuth=125;view.cam.elevation=-45
                view.cam.lookat[:]=arena.position
            from agent.fly_controls import FlyControls
            controls=FlyControls(pending.put,CHANNELS,out)
        (out/'launch.json').write_text(json.dumps({'physical_flies':1,'neural_lanes':2,
            'profile':profile,'automatic_stimulation':profile=='diagnostic',
            'unused_lane':'zero inputs, no physical body','backend':backend,'flight':False,
            'learning':'motor decoder only, graph fixed','controls':'Fly controls window: buttons; native viewer keys affect rendering only'},indent=2)+'\n')
        remote=RuntimeControl(out,profile,backend)
        def snapshot():
            return {'status':'APPLIED','time_s':t,'exchange':exchange,'paused':paused,
                'profile':profile,'backend':backend,'stopping':stopping,'pending_step_exchanges':step_remaining,
                'automatic_sweep':sweep,'direct_stimulation':direct_mode,
                'operator_resets':resets,'position_mm':arena.position.tolist(),
                'physics_steps':arena.physics_steps,
                'counts':last['counts'][0] if last else [0,0],
                'decoder_sha256':hashlib.sha256(model.weights.tobytes()).hexdigest(),
                'hunger':organism.hunger,'ingested_total':organism.state.ingested_total,
                'world_food_amounts':{p.id:p.amount for p in environment.food},'world_food':[{'id':p.id,'x':p.x,'y':p.y,'amount':p.amount} for p in environment.food],'world_event_cursor':environment.event_cursor}
        print('SINGLE_NEURAL_FLY_READY',flush=True)
        with (out/'telemetry.jsonl').open('w',buffering=1) as trace,(out/'events.jsonl').open('w',buffering=1) as events:
            while not stopping and (seconds==0 or t<seconds):
                if view and not view.is_running():break
                if controls:controls.update()
                while not pending.empty():
                    key=pending.get();event={'time_s':t,'control_code':key,'source':'button_panel'}
                    if key==999:stopping=True
                    elif key==32:paused=not paused
                    elif 49<=key<=56:
                        pulse[:]=0;pulse[key-49]=.8;pulse_end=t+.4;sweep=False
                        event['channel']=CHANNELS[key-49]
                    elif key==65:sweep=True;sweep_start=t
                    elif key==68:direct_mode=not direct_mode
                    elif key==86:blind=not blind;vision.previous=None
                    elif key in (82,80) and last is not None:
                        norm=model.reinforce(last['features'][0],action,1. if key==82 else -1.)
                        reward_updates+=int(norm>0);event['decoder_delta_norm']=norm
                    elif key==78:
                        event['reset_receipt']=reset_body_and_brain(arena,brain,initial_body,decision,t)
                        last=None;vision.previous=None;resets+=1
                    events.write(json.dumps(event)+'\n')
                for connection,request in remote.poll():
                    command=request['action'];extra={}
                    if stopping and command not in ('status','stop'):
                        remote.reply(connection,dict(snapshot(),status='REJECTED',error='runner is stopping'))
                        continue
                    if remote.waiting_step and command not in ('status','stop'):
                        remote.reply(connection,dict(snapshot(),status='REJECTED',error='step in progress'))
                        continue
                    if command in ('drive','camera'):
                        remote.reply(connection,dict(snapshot(),status='REJECTED',error=f'{command}: engine runners only'))
                        continue
                    if command=='pause':paused=True
                    elif command=='resume':paused=False
                    elif command=='stop':stopping=True
                    elif command=='reset_body_brain':
                        extra['reset_receipt']=reset_body_and_brain(arena,brain,initial_body,decision,t)
                        last=None;vision.previous=None;resets+=1
                        pulse[:]=0;pulse_end=t;attempted.clear()
                        extra['reset_scope']='body/controller/neural state; world, organism, decoder and session time preserved'
                        if backend=='cpu':
                            extra['reset_receipt']['cpu_neural_arrays_zero']=all(not np.any(x) for x in (brain.v,brain.spikes,brain.trace,brain.ever))
                        else:
                            zero=brain.step(np.zeros((2,8),np.float32))
                            extra['reset_receipt']['gpu_zero_observables']=not np.any(zero['features']) and not np.any(zero['counts'])
                            extra['reset_receipt']['gpu_reset_probe_ticks']=4
                            extra['reset_receipt']['gpu_reset_scope']='zero-input features/counts in both lanes; full arrays not exported'
                    elif command=='world_edit':
                        try:extra['world_edit']=world_edit.apply(request['edit'],environment,arena)
                        except (ValueError,TypeError,KeyError) as exc:
                            remote.reply(connection,dict(snapshot(),status='REJECTED',error=f'world_edit: {exc}'))
                            continue
                    elif command=='step':
                        if not paused:
                            remote.reply(connection,dict(snapshot(),status='REJECTED',error='pause before stepping'))
                            continue
                        step_remaining=request['steps'];remote.waiting_step=(connection,request)
                    elif command=='checkpoint':
                        if not paused:
                            remote.reply(connection,dict(snapshot(),status='REJECTED',error='pause before checkpointing'))
                            continue
                        continuation={'t':t,'exchange':exchange,'pulse_end':pulse_end,'pulse':pulse,
                            'sweep_start':sweep_start,'sweep':sweep,'reward_updates':reward_updates,'resets':resets,
                            'last':last,'action':action,'direct_mode':direct_mode,'attempted':sorted(attempted),
                            'direct_batches':direct_batches,'blind':blind}
                        state=live_checkpoint.capture(arena,brain,environment,organism,model,decision,desire,vision,initial_body,continuation)
                        extra.update(live_checkpoint.save(out/'checkpoints'/(request['request_id']+'.npz'),state,checkpoint_identity))
                    answer=dict(snapshot(),action=command,request_id=request['request_id'],**extra)
                    events.write(json.dumps(dict(answer,source='local_control',acknowledgement='deferred' if command=='step' else 'boundary'))+'\n')
                    if command!='step':remote.reply(connection,answer)
                if stopping:break
                if paused and not step_remaining:
                    if view:view.sync()
                    time.sleep(.03);continue
                eyes=arena.eyes()
                visual_drive,visual_threat,visual_info=vision.step(eyes,blind=blind)
                frame=environment.sense(arena,organism,eyes)
                frame.looming_left,frame.looming_right=map(float,visual_threat)
                inputs=encode(frame,absent)
                # Natural images enter mapped lamina columns, never side-wide ports.
                # Explicit diagnostic/manual channel injections still use legacy ports.
                inputs[6:8]=0
                label='environment'
                if sweep:
                    index=int((t-sweep_start)/.6)
                    if index<8:
                        inputs[index]=max(inputs[index],.8);label=CHANNELS[index]+' [injected]'
                    else:sweep=False
                if t<pulse_end:inputs=np.maximum(inputs,pulse)
                observed_inputs=inputs.copy()
                wall_contact,wall_turn=wall_observation(arena)
                inputs,motivation=desire.evaluate(organism.hunger,inputs,arena.position,
                    frame.mouth_contact,frame.upright,wall_contact=wall_contact,wall_turn=wall_turn)
                packed=np.zeros((2,8),np.float32);packed[0]=inputs
                targets=[]
                if direct_mode and not sweep and last is not None:
                    targets=[i for i in last.get('inactive',[]) if i not in attempted][:256]
                    if targets:
                        attempted.update(targets);direct_batches+=1
                        events.write(json.dumps({'time_s':t,'direct_inactive_targets':targets,'amplitude':1.,'neural_ticks':4})+'\n')
                visual=np.zeros((2,len(vision.spatial_drive)),np.float32);visual[0]=vision.spatial_drive
                last=brain.step(packed,direct=targets,after=max(attempted)+1 if attempted else 0,visual=visual)
                features=np.asarray(last['features'][0]);raw_decoder=model.predict(features[None])[0]
                decision.signal=bool(np.max(np.abs(features))>1e-8)
                decision.output=desire.modulate(raw_decoder,decision.signal)
                chosen=decision.decide(frame,organism,None)
                # Single arena has no combat primitive or rival body.
                if chosen.action=='FENCE':
                    from dataclasses import replace
                    chosen=replace(chosen,action='WALK')
                action=chosen.action
                gait=np.clip(decision.output[:2],-.75,.75) if action=='WALK' else np.zeros(2)
                before=organism.state.ingested_total
                physical=step_signed(arena,chosen,gait)
                environment.apply_physical_events(physical,organism,.01)
                organism.advance(.01,action,movement_proxy=frame.movement)
                food_reward=organism.state.ingested_total-before
                if food_reward>0:
                    reward_updates+=int(model.reinforce(features,action,min(1.,food_reward*10))>0)
                t+=.01;exchange+=1
                row={'time_s':t,'action':action,'stimulus':label,'inputs':inputs.tolist(),
                     'position_mm':arena.position.tolist(),'upright':arena.upright,
                     'vision':visual_info,'desire':motivation,'observed_inputs':observed_inputs.tolist(),'raw_decoder':raw_decoder.tolist(),
                     'counts':last['counts'][0],'unused_lane_counts':last['counts'][1],
                     'descending_feature_nonzero':int(np.count_nonzero(features)),
                     'decoder':decision.output.tolist(),'profile':profile,'reward_updates':reward_updates,'operator_resets':resets,'direct_targets_this_step':len(targets),'direct_targets_attempted':len(attempted)}
                trace.write(json.dumps(row,allow_nan=False,default=_json_default)+'\n')
                if step_remaining:
                    step_remaining-=1
                    if not step_remaining:
                        paused=True
                        answer=snapshot()
                        events.write(json.dumps(dict(answer,source='local_control',action='step_complete'))+'\n')
                        remote.finish_step(answer)
                if view:
                    with view.lock():view.cam.lookat[:]=arena.position
                    text=f'ONE FLY / GPU brain, CPU body / {t:.2f}s / {action}\n'
                    text+=f'DESIRE: {motivation["goal"]} / strength {motivation["strength"]:.2f} / hunger {motivation["hunger"]:.2f}\n'
                    text+=f'VISION: {"blind" if blind else "eye cameras active"} / visual {visual_drive.round(2).tolist()}\n'
                    text+=f'TURN: {motivation["turn_reason"] or "none"} / wall {motivation["wall_contact"]}\n'
                    text+=f'active {int(last["counts"][0][0])} / recruited {int(last["counts"][0][1])}/139255\n'
                    text+=f'{label} | Use the separate Fly controls buttons\n'
                    text+=f'Direct inactive stimulation {direct_mode} / attempted {len(attempted)}\n'
                    text+=f'Decoder updates {reward_updates} | Experimental perturbation; graph fixed; coarse image-driven vision'
                    if controls:controls.update(f'{action} / {motivation["goal"]} / hunger {motivation["hunger"]:.2f}\nReward updates {reward_updates} / direct pulses {direct_mode}')
                    view.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100,mujoco.mjtGridPos.mjGRID_TOPLEFT,text,''));view.sync()
    except BaseException as exc:
        error=repr(exc);raise
    finally:
        if controls:controls.close()
        if view:view.close()
        if arena:arena.close()
        if isinstance(brain,Worker):brain.close()
        if remote:remote.close(error)
        if 'model' in locals():model.save(out/'readout.npz')
        (out/'report.json').write_text(json.dumps({'error':error,'seconds':t,'physical_flies':1,
            'profile':profile,'exchange':exchange,'session_id':remote.session_id if remote else None,'reward_updates':reward_updates,'operator_resets':resets,'direct_targets_attempted':len(attempted),'direct_batches':direct_batches},indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--backend',choices=['cpu','gpu'],default='gpu');p.add_argument('--seconds',type=float,default=0.)
    p.add_argument('--profile',choices=['natural','diagnostic'],default='diagnostic')
    p.add_argument('--start-paused',action='store_true')
    p.add_argument('--restore-checkpoint',type=Path)
    p.add_argument('--headless',action='store_true');p.add_argument('--skip-audit',action='store_true');a=p.parse_args()
    if not np.isfinite(a.seconds) or a.seconds<0:p.error('seconds must be finite and nonnegative')
    run(a.output,a.backend,a.seconds,not a.headless,not a.skip_audit,a.profile,a.start_paused,a.restore_checkpoint)
