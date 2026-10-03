"""Smoke check for the imported FlyGym body (CPU MuJoCo): does a CPG-driven fly walk, and how fast?

Run with the FlyGym venv (.work/flygym-venv, FlyGym 2.x from ~/Projects/imports, Apache-2.0):
    .work/flygym-venv/bin/python tools/flygym_smoke.py [seconds]
Pass: thorax moves forward more than 1 mm per simulated second. Prints the real-time factor.
"""
import sys
import time

import numpy as np
from flygym import Simulation
from flygym.anatomy import BodySegment
from flygym.compose import FlatGroundWorld
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain import (CPGController, LocomotionAction, PreprogrammedSteps, apply_locomotion_action,
                                         make_locomotion_fly, make_tripod_cpg_network)

seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
fly = make_locomotion_fly(name='smoke', add_adhesion=True, colorize=False)
world = FlatGroundWorld()
world.add_fly(fly, [0, 0, 0.5], Rotation3D('quat', [1, 0, 0, 0]))
sim = Simulation(world)
steps = PreprogrammedSteps()
dofs = fly.get_actuated_jointdofs_order('position')
controller = CPGController(cpg_network=make_tripod_cpg_network(timestep=sim.timestep, intrinsic_amplitude=1.0, coupling_strength=10.0,
                                                               convergence_coef=20.0, seed=0),
                           preprogrammed_steps=steps, output_dof_order=dofs)
sim.reset()
apply_locomotion_action(sim, fly.name, LocomotionAction(joint_angles=steps.default_pose_by_dof_order(dofs), adhesion_onoff=np.ones(6, dtype=bool)))
sim.warmup()
thorax = fly.get_bodysegs_order().index(BodySegment('c_thorax'))
start = sim.get_body_positions(fly.name)[thorax].copy()
n = int(seconds / sim.timestep)
t0 = time.perf_counter()
for _ in range(n):
    apply_locomotion_action(sim, fly.name, controller.step())
    sim.step()
wall = time.perf_counter() - t0
forward = float(sim.get_body_positions(fly.name)[thorax][0] - start[0])
ok = forward > seconds
print(f'{n} steps ({seconds:.2f} s simulated, dt {sim.timestep}) in {wall:.2f} s wall: real-time factor {seconds / wall:.2f}x, '
      f'forward {forward:.2f} mm  {"PASS" if ok else "FAIL"}')
sys.exit(0 if ok else 1)
