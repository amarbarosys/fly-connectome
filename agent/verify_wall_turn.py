"""Bounded CPU physics reproduction facing an arena wall."""
import json
import numpy as np
import mujoco
from agent.neural_single import wall_observation,EthologyArena,step_signed
from agent.desire import Desire
from fly_arena.behavior import BehaviorDecision
from agent.neural_connectome import ROOT

def verify():
    arena=EthologyArena(seed=1);desire=Desire();records=[]
    try:
        arena.sim.mj_data.qpos[:3]=[19.2,0,.8]
        arena.sim.mj_data.qpos[3:7]=[1,0,0,0]
        arena.sim.mj_data.qvel[:]=0
        mujoco.mj_forward(arena.sim.mj_model,arena.sim.mj_data)
        initial_heading=arena.heading
        for _ in range(200):
            contact,direction=wall_observation(arena)
            _,state=desire.evaluate(.8,np.zeros(8),arena.position,False,arena.upright,
                                   wall_contact=contact,wall_turn=direction)
            drive=desire.modulate([.4,.4,0,0,0],True)
            step_signed(arena,BehaviorDecision('WALK'),drive[:2])
            records.append({'position_mm':arena.position.tolist(),'heading':arena.heading,
                            'upright':arena.upright,'desire':state,'drive':drive.tolist()})
        turned=[r for r in records if r['desire']['unstick_turn']]
        delta=float(np.arctan2(np.sin(arena.heading-initial_heading),np.cos(arena.heading-initial_heading)))
        result={'contact_frames':sum(r['desire']['wall_contact'] for r in records),
                'turn_frames':len(turned),'heading_change_rad':delta,'final_position_mm':arena.position.tolist(),
                'min_upright':min(r['upright'] for r in records),'finite':bool(np.isfinite(arena.sim.mj_data.qpos).all())}
        print(json.dumps(result),flush=True)
        assert result['contact_frames']>0 and result['turn_frames']>0
        assert abs(delta)>.3 and result['finite'] and result['min_upright']>.8
        result['status']='PASS'
        (ROOT/'.work/neural-flies/wall-turn-gate.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
    finally:arena.close()
if __name__=='__main__':verify()
