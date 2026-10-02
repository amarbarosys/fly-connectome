"""Check the actual neural/body boundary with intact and null sensory input."""
import json
from pathlib import Path
import time
import numpy as np
from agent.neural_connectome import Worker, Readout, ROOT
from agent.neural_duel import NeuralDuel


def verify():
    results = {}
    started = time.perf_counter()
    for control in ('intact','null','disconnected'):
        brain = Worker(disconnected=control=='disconnected')
        duel = None
        try:
            duel = NeuralDuel(brain,Readout(),control=control)
            initial = np.array([c.unit.position.copy() for c in duel.contestants])
            features, commands, finite = [], [], True
            for _ in range(60):
                duel.step()
                features.append(duel.last_neural['features'])
                commands.append(duel.last_outputs.copy())
                finite &= bool(np.isfinite(duel.arena.sim.mj_data.qpos).all())
            result = {'simulated_seconds':duel.elapsed_s,'finite_physics':finite,
                      'feature_nonzero':int(np.count_nonzero(features)),
                      'motor_command_max_abs':float(np.max(np.abs(commands))),
                      'displacement_mm':np.linalg.norm(np.array([c.unit.position for c in duel.contestants])-initial,axis=1).tolist(),
                      'actions':[c.action_durations for c in duel.contestants],
                      'neural_counts':duel.last_neural['counts']}
            assert finite
            if control == 'intact':
                assert result['feature_nonzero']>0
                assert result['motor_command_max_abs']>0
                assert any('WALK' in c.action_durations or 'FEED' in c.action_durations or 'FENCE' in c.action_durations for c in duel.contestants)
            else:
                assert result['feature_nonzero']==0
                assert result['motor_command_max_abs']==0
                assert all(set(c.action_durations)=={'IDLE'} for c in duel.contestants)
            results[control] = result
            print(control,json.dumps(result),flush=True)
        finally:
            if duel:duel.arena.close()
            brain.close()
    results['status']='PASS'
    results['wall_seconds']=time.perf_counter()-started
    results['scope']='short software causal intervention, not navigation competence or biological validity'
    path=ROOT/'.work/neural-flies/embodiment-gate.json'
    path.write_text(json.dumps(results,indent=2)+'\n')
    return results

if __name__=='__main__':verify()
