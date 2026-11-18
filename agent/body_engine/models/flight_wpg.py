"""Flybody's wing beat pattern generator as a table, plus the episode start it pairs with, for the engine's flight
(flight.py). flight_imitation (seed 1) reset, then WingBeatPatternGenerator.step at the base beat frequency (policy action
0) for N control steps: target wing angles (N, 6: yaw, roll, pitch left then right, the order of task._wing_joints).
Flybody's FlightImitation.before_step turns them into force control: ctrl[wing] = target - qpos[wing].
Run with the flight venv: .work/neural-flies/flight-venv/bin/python agent/body_engine/models/flight_wpg.py OUT_DIR [N]
(writes OUT_DIR/wpg.npz: target, qpos0, qvel0, wing_qadr, wing_act, names)"""
import sys
from pathlib import Path

import numpy as np
from flybody.fly_envs import flight_imitation

ROOT = str(Path(__file__).resolve().parents[3])
out, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2500
env = flight_imitation(wpg_pattern_path=f'{ROOT}/.work/neural-flies/flight-data/wing_pattern_fmech.npy',
                       terminal_com_dist=float('inf'), random_state=np.random.RandomState(1))
env.reset()
t, p = env.task, env.physics
wing = p.bind(t._wing_joints)
act = [p.model.name2id(a.full_identifier, 'actuator') for a in t._walker.mjcf_model.find_all('actuator')
       if a.name.startswith('wing_')]
target = np.array([t._wbpg.step(ctrl_freq=t._wbpg.base_beat_freq) for _ in range(n)])
np.savez(f'{out}/wpg.npz', target=target, qpos0=p.data.qpos.copy(), qvel0=p.data.qvel.copy(), wing_qadr=np.array(wing.qposadr),
         wing_act=np.array(act), names=np.array([j.name for j in t._wing_joints] + [p.model.id2name(i, 'actuator') for i in act]))
print(f'{out}/wpg.npz', target.shape, 'nq', p.model.nq, 'nu', p.model.nu, 'z0', p.data.qpos[2])
