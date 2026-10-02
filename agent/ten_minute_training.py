"""Timed ten-arena reward-labelled steering training with atomic checkpoints."""
import argparse
import importlib.util
import json
from multiprocessing.connection import Client
from pathlib import Path
import time

import numpy as np

from agent.neural_connectome import CPUBrain, Worker, digest
from agent.odor_memory_gate import scenes
from agent.parallel_learning_trial import experience, fit

ROOT=Path('.work/agent/ten-minute-training')
SOCKET=ROOT/'physics.sock'
BINARY=Path('.work/agent/ten-fly/bin/worker-c10-optimized')


def run(preflight=False):
    output=ROOT/('cpu-preflight' if preflight else 'session')
    output.mkdir(exist_ok=False)
    prior=Path('.work/agent/spectator-learning/learning/seed-11-c10/samples.npz')
    data=np.load(prior);features=list(data['features']);rewards=list(data['rewards'])
    spec=importlib.util.spec_from_file_location('archived_skillopt_patch',Path('.work/agent/skillopt-pilot/skillopt_skill.py'))
    skillopt=importlib.util.module_from_spec(spec);spec.loader.exec_module(skillopt)
    skill='# Shared fly training supervisor\n\ntraining_examples: 10\n\nAccept only strict improvement on selection worlds; preserve final-test worlds.\n'
    (output/'best_skill.md').write_text(skill)
    current,_=fit(np.asarray(features),np.asarray(rewards),output/'initial.npz')
    current.save(output/'latest.npz')
    brain=CPUBrain(flies=1) if preflight else Worker(binary=BINARY,flies=10)
    connection=Client(str(SOCKET),family='AF_UNIX')
    start=time.monotonic();deadline=start+(600 if not preflight else 60)
    summaries=[];accepted_updates=0
    (output/'ready.json').write_text(json.dumps({'started_unix':time.time(),'target_seconds':600,
        'mode':'reward-labelled two-action steering with continuous neural exchanges',
        'training_seed_start':14000,'selection_seed_start':15000,'reserved_final_test_seed_start':13000,
        'skillopt':'actual archived patch engine, fixed bounded replay-budget proposal; autonomous LLM backend not used'})+'\n')
    print('TEN_MINUTE_TRAINING_READY',flush=True)
    try:
        with (output/'batches.jsonl').open('w',buffering=1) as log:
            batch=0
            while time.monotonic()<deadline:
                configs=scenes(range(14000+batch*5,14005+batch*5))[:1 if preflight else 10]
                f,r,inputs,physical=experience(connection,brain,configs,continuous=True)
                features.extend(f);rewards.extend(r)
                path=output/f'checkpoint-{batch:03d}.npz'
                model,useful=fit(np.asarray(features),np.asarray(rewards),path)
                prior_budget=next(line for line in skill.splitlines() if line.startswith('training_examples:'))
                patch={'edits':[{'op':'replace','target':prior_budget,'content':f'training_examples: {len(features)}'}]}
                candidate_skill,application=skillopt.apply_patch_with_report(skill,patch)
                assert application[0]['status']=='applied_replace'
                validation=scenes(range(15000+batch*5,15005+batch*5))[:1 if preflight else 10]
                vf,vr,vi,vphysical=experience(connection,brain,validation,continuous=True)
                def score(decoder):
                    predictions=decoder.predict(vf)
                    choices=(predictions[:,1]<=predictions[:,0]).astype(int)
                    return int(np.count_nonzero(vr[np.arange(len(vr)),choices]>=.05))
                old_score,new_score=score(current),score(model)
                accepted=new_score>old_score
                if accepted:
                    current=model;skill=candidate_skill;accepted_updates+=1
                    temporary=output/'latest.pending.npz';current.save(temporary);temporary.replace(output/'latest.npz')
                    (output/'best_skill.md').write_text(skill)
                (output/f'skillopt-{batch:03d}.json').write_text(json.dumps({'patch':patch,'application':application,
                    'candidate_skill':candidate_skill,'accepted':accepted,'old_successes':old_score,
                    'candidate_successes':new_score,'selection_worlds':validation,
                    'optimizer':'fixed bounded proposal schedule, not autonomous LLM reflection'},indent=2)+'\n')
                np.savez(output/f'selection-{batch:03d}.npz',features=vf,rewards=vr,inputs=vi)
                (output/f'selection-physical-{batch:03d}.json').write_text(json.dumps(vphysical)+'\n')
                np.savez(output/f'experience-{batch:03d}.npz',features=f,rewards=r,inputs=inputs)
                (output/f'physical-{batch:03d}.json').write_text(json.dumps(physical)+'\n')
                row={'batch':batch,'elapsed_s':time.monotonic()-start,'new_scenes':configs,
                    'training_samples':len(features),'informative_pairs':useful,
                    'checkpoint_sha256':digest(path),'reward_pairs':r.tolist(),
                    'skillopt_candidate_accepted':accepted,'selection_old_successes':old_score,
                    'selection_candidate_successes':new_score,'selection_worlds':validation,
                    'physical_body_exchanges':len(configs)*268,
                    'neural_body_exchanges':len(configs)*284}
                log.write(json.dumps(row)+'\n');summaries.append(row)
                print(json.dumps({'batch':batch,'elapsed_s':row['elapsed_s'],'samples':len(features)}),flush=True)
                batch+=1
                if preflight:break
                # Avoid beginning another full cohort near the deadline.
                duration=(time.monotonic()-start)/batch
                if deadline-time.monotonic()<duration:break
        report={'status':'PASS','wall_seconds':time.monotonic()-start,'target_seconds':600,
            'completed_batches':len(summaries),'training_samples':len(features),
            'fresh_reward_labelled_worlds':sum(len(x['new_scenes']) for x in summaries),
            'skillopt_review_rounds':len(summaries),'accepted_decoder_updates':accepted_updates,
            'final_test_used':False,'saved_decoder':str(output/'latest.npz'),
            'scope':'bounded contextual-bandit steering, prescribed physical primitives, fixed anatomical graph; no final efficacy claim'}
        (output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    finally:
        connection.close()
        if not preflight:brain.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    run(args.preflight)
