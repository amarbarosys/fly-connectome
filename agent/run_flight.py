#!/usr/bin/env python3
"""
agent/run_flight.py
Reproducible baseline execution of deep reinforcement learning flight policy in Flybody MuJoCo.
Loads pretrained flight imitation SavedModel, runs flight maneuvers, logs kinematics,
and renders visual flight camera snapshots.
"""

import sys
import os
import json
import argparse
from pathlib import Path
import numpy as np

# Suppress TF logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow.python.framework import type_spec_registry
import tensorflow_probability as tfp
from PIL import Image

# Register missing TFP TypeSpec for legacy SavedModel compatibility
cls = type_spec_registry.lookup("tfp.distributions.Independent_ACTTypeSpec")
type_spec_registry._NAME_TO_TYPE_SPEC[
    "tensorflow_probability.python.distributions.independent.Independent_ACTTypeSpec"
] = cls

from flybody.fly_envs import flight_imitation

ROOT = Path(__file__).resolve().parents[1]
FLIGHT_DIR = ROOT / ".work" / "neural-flies" / "flight-data"
MODEL_PATH = FLIGHT_DIR / "flight"
REF_PATH = FLIGHT_DIR / "flight-dataset_saccade-evasion_augmented.hdf5"
WPG_PATH = FLIGHT_DIR / "wing_pattern_fmech.npy"
OUT_DIR = ROOT / ".work" / "neural-flies" / "flight-run"


def run_flight(steps: int = 200, render_cameras: bool = True):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading flight policy model from {MODEL_PATH}...")
    model = tf.saved_model.load(str(MODEL_PATH))

    print(f"Initializing Flybody flight imitation environment...")
    env = flight_imitation(
        ref_path=str(REF_PATH),
        wpg_pattern_path=str(WPG_PATH),
    )

    timestep = env.reset()
    physics = env.physics

    trajectory = []
    rewards = []
    altitudes = []
    velocities = []

    print(f"Stepping flight simulation for {steps} steps (dt={physics.timestep():.4f}s)...")
    for step_idx in range(steps):
        # Format batch observation for SavedModel
        batch_obs = {
            k: tf.convert_to_tensor(np.expand_dims(v, 0), dtype=tf.float32)
            for k, v in timestep.observation.items()
        }

        # Policy inference
        action_dist = model(batch_obs)
        action = action_dist.sample()[0].numpy()

        timestep = env.step(action)
        rewards.append(float(timestep.reward) if timestep.reward is not None else 0.0)

        # Extract kinematics from physics
        # Root position (x, y, z) in cm
        qpos = np.copy(physics.data.qpos[:7])
        qvel = np.copy(physics.data.qvel[:6])
        pos_cm = qpos[:3] * 100.0  # convert to cm
        altitudes.append(float(pos_cm[2]))
        speed_cm_s = float(np.linalg.norm(qvel[:3]) * 100.0)
        velocities.append(speed_cm_s)

        trajectory.append({
            "step": step_idx,
            "time_sec": float(physics.time()),
            "pos_cm": [float(p) for p in pos_cm],
            "speed_cm_s": speed_cm_s,
            "reward": float(timestep.reward) if timestep.reward is not None else 0.0,
            "discount": float(timestep.discount) if timestep.discount is not None else 1.0,
        })

    # Render keyframe camera views
    rendered_files = {}
    if render_cameras:
        print("Rendering visual flight camera snapshots...")
        for cam_name in ["walker/hero", "walker/side", "top_camera"]:
            clean_name = cam_name.replace("/", "_")
            pixels = physics.render(camera_id=cam_name, height=720, width=1280)
            img = Image.fromarray(pixels)
            img_path = OUT_DIR / f"{clean_name}.png"
            img.save(img_path)
            rendered_files[clean_name] = str(img_path)
            print(f"  -> Saved {img_path} ({pixels.shape[1]}x{pixels.shape[0]})")

    # Summary metrics
    traj_arr = np.array([t["pos_cm"] for t in trajectory])
    is_finite = bool(np.all(np.isfinite(traj_arr)))
    total_displacement = float(np.linalg.norm(traj_arr[-1] - traj_arr[0]))
    mean_altitude = float(np.mean(altitudes))
    mean_speed = float(np.mean(velocities))
    mean_reward = float(np.mean(rewards))

    summary = {
        "status": "PASS" if is_finite and mean_altitude > 0.8 else "FAIL",
        "steps": steps,
        "physics_time_sec": float(physics.time()),
        "is_finite": is_finite,
        "mean_altitude_cm": mean_altitude,
        "altitude_range_cm": [float(np.min(altitudes)), float(np.max(altitudes))],
        "total_displacement_cm": total_displacement,
        "mean_speed_cm_s": mean_speed,
        "mean_reward": mean_reward,
        "rendered_files": rendered_files,
    }

    # Save outputs
    json_path = OUT_DIR / "flight_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    np.save(OUT_DIR / "flight_poses.npy", traj_arr)

    print("\n========================================================")
    print(f" Flight Run Complete: {summary['status']}")
    print(f" Physics Time: {summary['physics_time_sec']:.3f} s ({steps} steps)")
    print(f" Mean Altitude: {summary['mean_altitude_cm']:.3f} cm [Min: {summary['altitude_range_cm'][0]:.3f}, Max: {summary['altitude_range_cm'][1]:.3f}]")
    print(f" Displacement:  {summary['total_displacement_cm']:.3f} cm | Mean Speed: {summary['mean_speed_cm_s']:.2f} cm/s")
    print(f" Mean Reward:    {summary['mean_reward']:.4f}")
    print(f" Summary JSON:  {json_path}")
    print("========================================================\n")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Flybody flight policy baseline")
    parser.add_argument("--steps", type=int, default=200, help="Number of flight policy steps")
    args = parser.parse_args()
    run_flight(steps=args.steps)
