"""Basic pretrained Flybody flight; starts airborne, no connectome coupling yet."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['TF_NUM_INTRAOP_THREADS']='2'
os.environ['TF_NUM_INTEROP_THREADS']='2'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def run(headless=False, steps=0):
    import numpy as np
    import tensorflow as tf
    import tensorflow_probability as tfp
    from tensorflow_probability.python.internal import auto_composite_tensor as act
    import mujoco.viewer
    from flybody.fly_envs import flight_imitation
    data=ROOT/'.work/neural-flies/flight-data'
    out=ROOT/('.work/neural-flies/basic-flight-preflight' if headless else '.work/neural-flies/basic-flight');out.mkdir(parents=True,exist_ok=True)
    for cls in [tfp.distributions.Independent,tfp.distributions.Normal,tfp.distributions.TransformedDistribution,
                tfp.distributions.MultivariateNormalDiag,tfp.bijectors.Tanh,tfp.bijectors.Scale,
                tfp.bijectors.Shift,tfp.bijectors.Chain]:act.auto_composite_tensor(cls)
    policy=tf.saved_model.load(str(data/'flight'))
    env=flight_imitation(wpg_pattern_path=str(data/'wing_pattern_fmech.npy'),terminal_com_dist=float('inf'),random_state=np.random.RandomState(1))
    from flybody.tasks.synthetic_trajectories import constant_speed_trajectory
    qpos,qvel=constant_speed_trajectory(n_steps=3500,speed=20,init_pos=(0,0,1),body_rot_angle_y=-47.5,control_timestep=.0002)
    env.task._traj_generator.set_next_trajectory(qpos,qvel)
    obs=env.reset();spec=env.action_spec();view=None;count=0;episodes=0;error=None
    try:
        if not headless:
            view=mujoco.viewer.launch_passive(env.physics.model.ptr,env.physics.data.ptr,show_left_ui=False,show_right_ui=False)
            view.cam.distance=5;view.cam.elevation=-25;view.cam.azimuth=100
        print('BASIC_FLIGHT_READY: pretrained controller; starts airborne',flush=True)
        with (out/'trajectory.jsonl').open('w',buffering=1) as trace:
            while steps==0 or count<steps:
                if view and not view.is_running():break
                inputs={k:tf.convert_to_tensor(np.array(v,np.float32)[None]) for k,v in obs.observation.items()}
                action=policy(inputs).mean()[0].numpy()
                action=(action+1)*(spec.maximum-spec.minimum)/2+spec.minimum
                obs=env.step(np.clip(action,spec.minimum,spec.maximum));count+=1
                pose=env.physics.data.qpos[:7].copy()
                if not np.isfinite(env.physics.data.qpos).all():raise RuntimeError('Nonfinite flight state')
                trace.write(json.dumps({'step':count,'episode':episodes,'time_s':float(env.physics.data.time),'pose_cm':pose.tolist()})+'\n')
                if view and count%10==0:
                    with view.lock():view.cam.lookat[:]=pose[:3]
                    view.sync()
                if obs.last():obs=env.reset();episodes+=1
    except BaseException as exc:error=repr(exc);raise
    finally:
        if view:view.close()
        env.close()
        (out/'report.json').write_text(json.dumps({'error':error,'steps':count,'episodes':episodes,
            'controller':'pretrained Flybody flight policy','starts_airborne':True,'neural_control':False},indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--headless',action='store_true');p.add_argument('--steps',type=int,default=0);a=p.parse_args()
    run(a.headless,a.steps)
