"""Export the installed FlyGym preprogrammed gait with explicit MJCF joint mapping.

Source API: flygym_demo.complex_terrain.preprogrammed.PreprogrammedSteps.
The source angles are radians. FlyGym v2 flips right roll/yaw when it constructs
PreprogrammedSteps; the exported values retain that convention. The model is
provided explicitly so joint names and output order are tied to the engine MJCF.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flygym_demo.complex_terrain import PreprogrammedSteps, HybridTurningController
from agent.body_engine.model import standing

LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
DOFS = (
    ("coxa", "pitch"),
    ("coxa", "roll"),
    ("coxa", "yaw"),
    ("trochanterfemur", "pitch"),
    ("trochanterfemur", "roll"),
    ("tibia", "pitch"),
    ("tarsus1", "pitch"),
)


def joint_name(leg: str, segment: str, axis: str) -> str:
    if segment == "coxa":
        return f"fly/c_thorax-{leg}_coxa-{axis}"
    if segment == "trochanterfemur":
        return f"fly/{leg}_coxa-{leg}_trochanterfemur-{axis}"
    if segment == "tibia":
        return f"fly/{leg}_trochanterfemur-{leg}_tibia-{axis}"
    return f"fly/{leg}_tibia-{leg}_tarsus1-{axis}"



def rollout(mjcf: Path) -> None:
    """Run the mapped controller on body-v2 in MuJoCo as a morphology preflight."""
    from flygym_demo.complex_terrain.common import get_default_locomotion_dof_order
    from flygym_demo.complex_terrain.cpg_controller import CPGController, make_tripod_cpg_network
    from flygym_demo.complex_terrain.hybrid_controller import HybridControllerObservation
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    qpos, qvel = standing(model, 2)
    start_q, start_v = qpos[0].copy(), qvel[0].copy()
    thorax = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fly/c_thorax")
    root = int(model.body_rootid[thorax])
    geoms = np.flatnonzero(model.body_rootid[model.geom_bodyid] == root)
    data = mujoco.MjData(model)
    data.qpos[:] = start_q
    data.qvel[:] = start_v
    mujoco.mj_forward(model, data)
    xspan = data.geom_xpos[geoms, 0] + model.geom_rbound[geoms]
    xlow = data.geom_xpos[geoms, 0] - model.geom_rbound[geoms]
    body_length = float(max(xspan) - min(xlow))
    if body_length <= 0:
        raise ValueError("could not derive fly body length")
    order = get_default_locomotion_dof_order()
    steps = PreprogrammedSteps()
    action_dofs = (("coxa", "yaw"), ("coxa", "pitch"), ("coxa", "roll"),
                   ("trochanterfemur", "pitch"), ("trochanterfemur", "roll"),
                   ("tibia", "pitch"), ("tarsus1", "pitch"))
    output_to_joint = [joint_name(leg, segment, axis) for leg in LEGS for segment, axis in action_dofs]
    if len(order) != len(output_to_joint):
        raise ValueError(f"FlyGym actuator order size {len(order)} != mapped leg DOFs {len(output_to_joint)}")
    act_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n + "-position") for n in output_to_joint]
    adh_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"fly/{leg}_tarsus5-adhesion") for leg in LEGS]
    if min(act_ids + adh_ids) < 0:
        raise ValueError("body-v2 lacks a gait or adhesion actuator")
    held = np.where(model.actuator_trntype == mujoco.mjtTrn.mjTRN_BODY, 1.0,
                    start_q[model.jnt_qposadr[np.minimum(model.actuator_trnid[:, 0], model.njnt - 1)]])
    obs = HybridControllerObservation(thorax_z=0.0, tarsus5_z=np.zeros(6),
                                      stumbling_contact_forces=np.zeros((6, 3, 3)),
                                      fly_heading=np.array([1.0, 0.0, 0.0]))

    def run(signal: tuple[float, float]) -> tuple[float, float, float]:
        data.qpos[:] = start_q; data.qvel[:] = start_v
        mujoco.mj_forward(model, data)
        controller = HybridTurningController(timestep=float(model.opt.timestep), preprogrammed_steps=steps,
                                             output_dof_order=order, enable_adhesion=True)
        controller.reset(seed=4, init_phases=np.array([0, np.pi, 0, np.pi, 0, np.pi]),
                         init_magnitudes=np.zeros(6))
        initial = data.xpos[thorax].copy()
        r = data.xmat[thorax].reshape(3, 3)
        last_yaw = float(np.arctan2(r[1, 0], r[0, 0])); yaw_delta = 0.0
        max_tilt = 0.0
        for _ in range(round(1.0 / model.opt.timestep)):
            action = controller.step(np.asarray(signal), obs)
            ctrl = held.copy()
            ctrl[act_ids] = action.joint_angles
            ctrl[adh_ids] = action.adhesion_onoff.astype(np.float64)
            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data)
            r = data.xmat[thorax].reshape(3, 3)
            yaw = float(np.arctan2(r[1, 0], r[0, 0]))
            yaw_delta += float(np.arctan2(np.sin(yaw - last_yaw), np.cos(yaw - last_yaw)))
            last_yaw = yaw
            max_tilt = max(max_tilt, float(np.degrees(np.arccos(np.clip(r[2, 2], -1, 1)))))
        displacement = float(data.xpos[thorax, 0] - initial[0])
        return displacement, yaw_delta, max_tilt

    def run_cpg() -> tuple[float, float]:
        data.qpos[:] = start_q; data.qvel[:] = start_v
        mujoco.mj_forward(model, data)
        net = make_tripod_cpg_network(timestep=float(model.opt.timestep), intrinsic_amplitude=1.0, seed=0)
        controller = CPGController(net, steps, order)
        initial = data.xpos[thorax].copy()
        max_tilt = 0.0
        for _ in range(round(1.0 / model.opt.timestep)):
            action = controller.step()
            ctrl = held.copy(); ctrl[act_ids] = action.joint_angles
            ctrl[adh_ids] = action.adhesion_onoff.astype(np.float64)
            data.ctrl[:] = ctrl; mujoco.mj_step(model, data)
            r = data.xmat[thorax].reshape(3, 3)
            max_tilt = max(max_tilt, float(np.degrees(np.arccos(np.clip(r[2, 2], -1, 1)))))
        return float(data.xpos[thorax, 0] - initial[0]), max_tilt

    forward = run((1.0, 1.0))
    cpg = run_cpg()
    zero = run((0.0, 0.0))
    left = run((0.25, 1.0))
    right = run((1.0, 0.25))
    print(f"body-v2 MuJoCo 1s: body_length={body_length:.4g} model units; Hybrid={forward[0]:.4g}; CPG={cpg[0]:.4g}; "
          f"zero={zero[0]:.4g}; yaw(left/right)={left[1]:.4g}/{right[1]:.4g}; "
          f"tilt_max={forward[2]:.3g} deg")
    if forward[0] > 3 * body_length and forward[2] < 20 and abs(zero[0]) < forward[0] and cpg[0] > 3 * body_length and cpg[1] < 20:
        print("PASS gait locomotion reference")
    else:
        print("FAIL gait locomotion reference")
    if left[1] * right[1] < 0:
        print("PASS opposite turn yaw sign")
    else:
        print("FAIL opposite turn yaw sign")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mjcf", type=Path, nargs="?", help="body-v2/body.xml")
    parser.add_argument("out", type=Path, nargs="?", help="output .npz")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--check-export", type=Path, help="verify a prior export against the installed FlyGym oracle")
    parser.add_argument("--check-controller", action="store_true", help="compare mapped targets against HybridTurningController")
    parser.add_argument("--rollout", type=Path, help="one-second body-v2 MuJoCo morphology preflight")
    args = parser.parse_args()
    if args.rollout:
        rollout(args.rollout)
        return
    if args.check_controller:
        from flygym_demo.complex_terrain.common import get_default_locomotion_dof_order
        from flygym_demo.complex_terrain.hybrid_controller import HybridControllerObservation
        steps = PreprogrammedSteps()
        order = get_default_locomotion_dof_order()
        controller = HybridTurningController(timestep=1e-4, preprogrammed_steps=steps,
                                             output_dof_order=order, enable_adhesion=True)
        obs = HybridControllerObservation(thorax_z=0.0, tarsus5_z=np.zeros(6),
                                          stumbling_contact_forces=np.zeros((6, 3, 3)),
                                          fly_heading=np.array([1.0, 0.0, 0.0]))
        max_error = 0.0
        for left, right in ((1.0, 1.0), (0.25, 1.0), (1.0, 0.25), (-0.5, 0.75)):
            controller.reset(seed=4)
            for _ in range(200):
                action = controller.step(np.array([left, right]), obs)
                phases = controller.cpg_network.curr_phases
                magnitudes = controller.cpg_network.curr_magnitudes
                expected = steps.get_joint_angles_by_dof_order(phases, magnitudes, order)
                max_error = max(max_error, float(np.max(np.abs(action.joint_angles - expected))))
                adhesion_expected = np.array([controller._get_adhesion_onoff(leg, phases[i]) for i, leg in enumerate(LEGS)])
                if not np.array_equal(action.adhesion_onoff, adhesion_expected):
                    raise SystemExit("FAIL HybridTurningController adhesion mapping")
        if max_error > 1e-10:
            raise SystemExit(f"FAIL controller target mismatch max_abs={max_error:.9g}")
        print(f"PASS HybridTurningController target/adhesion mapping: max_abs={max_error:.1f}")
        return
    if args.check_export:
        data = np.load(args.check_export)
        meta = json.loads(args.check_export.with_suffix(".json").read_text())
        phases = data["phase"].astype(float)
        steps = PreprogrammedSteps()
        expected = np.stack([steps.get_joint_angles(leg, phases) for leg in LEGS]).astype(np.float32)
        errors = [float(np.max(np.abs(expected - data["targets"]))) ]
        for slot, item in enumerate(meta["servo_mapping"]):
            if item["leg"] is not None:
                li, di = LEGS.index(item["leg"]), item["dof_index"]
                errors.append(float(np.max(np.abs(expected[li, di] - data["servo_targets"][:, slot]))))
        adhesion_controller = HybridTurningController(timestep=1e-4, preprogrammed_steps=steps)
        expected_adhesion = np.array([[adhesion_controller._get_adhesion_onoff(leg, float(ph)) for leg in LEGS]
                                      for ph in phases], np.float32)
        errors.append(float(np.max(np.abs(expected_adhesion - data["adhesion"]))))
        if max(errors) > 1e-6:
            raise SystemExit(f"FAIL export mismatch max_abs={max(errors):.9g}")
        if data["servo_targets"].shape[1] != 44 or data["adhesion"].shape[1] != 6:
            raise SystemExit("FAIL actuator layout must be 44 servos + 6 adhesion controls")
        print(f"PASS FlyGym export: 256 phase samples; 44 servo slots; 6 adhesion controls; max_abs={max(errors):.3g}")
        return
    if args.mjcf is None or args.out is None:
        parser.error("mjcf and out are required unless --check-export is used")

    model = mujoco.MjModel.from_xml_path(str(args.mjcf))
    standing_qpos, _ = standing(model, 2)
    standing_qpos = standing_qpos[0]
    steps = PreprogrammedSteps()
    adhesion_controller = HybridTurningController(timestep=float(model.opt.timestep), preprogrammed_steps=steps)
    phase = np.linspace(0.0, 2 * np.pi, args.samples, endpoint=False)
    table = np.empty((len(LEGS), len(DOFS), args.samples), dtype=np.float32)
    neutral = np.empty((len(LEGS), len(DOFS)), dtype=np.float32)
    mapping = []
    for li, leg in enumerate(LEGS):
        table[li] = steps.get_joint_angles(leg, phase).astype(np.float32)
        neutral[li] = steps.get_joint_angles(leg, np.pi).astype(np.float32)
        for di, (segment, axis) in enumerate(DOFS):
            name = joint_name(leg, segment, axis)
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"MJCF lacks mapped joint {name}")
            mapping.append({"leg": leg, "dof_index": di, "segment": segment,
                            "axis": axis, "joint_name": name, "joint_id": int(jid),
                            "actuator_name": name + "-position",
                            "actuator_id": int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name + "-position")),
                            "unit": "radian",
                            "source_sign": -1 if leg[0] == "r" and axis in ("roll", "yaw") else 1})
    absent = [m["actuator_name"] for m in mapping
              if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, m["actuator_name"]) < 0]
    if absent:
        raise ValueError(f"MJCF lacks mapped actuators: {absent}")
    actuator_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                      for i in range(model.nu)]
    servo = sorted(mapping, key=lambda m: m["actuator_id"])
    for joint in ("fly/c_head-c_rostrum-pitch", "fly/c_rostrum-c_haustellum-pitch"):
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint + "-position")
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        if aid < 0 or jid < 0:
            raise ValueError(f"MJCF lacks held head servo {joint}")
        servo.append({"leg": None, "dof_index": None, "segment": "head", "axis": "pitch",
                      "joint_name": joint, "joint_id": int(jid), "actuator_name": joint + "-position",
                      "actuator_id": int(aid), "unit": "radian", "source_sign": 1,
                      "target": "live.py prepare standing qpos"})
    servo.sort(key=lambda m: m["actuator_id"])
    if len(servo) != 44:
        raise ValueError(f"expected 44 position servos (42 legs + 2 head), found {len(servo)}")
    servo_targets = np.empty((args.samples, len(servo)), np.float32)
    adhesion = np.empty((args.samples, len(LEGS)), np.float32)
    for sample, ph in enumerate(phase):
        adhesion[sample] = [float(adhesion_controller._get_adhesion_onoff(leg, float(ph))) for leg in LEGS]
    for slot, item in enumerate(servo):
        if item["leg"] is None:
            qadr = int(model.jnt_qposadr[item["joint_id"]])
            servo_targets[:, slot] = standing_qpos[qadr]
        else:
            li, di = LEGS.index(item["leg"]), item["dof_index"]
            servo_targets[:, slot] = table[li, di]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, phase=phase.astype(np.float32), targets=table,
                        neutral=neutral, servo_targets=servo_targets, adhesion=adhesion,
                        actuator_names=np.asarray(actuator_names))
    metadata = {"source": "FlyGym PreprogrammedSteps (installed FlyGym 2.x)",
                "source_api": "flygym_demo.complex_terrain.preprogrammed.PreprogrammedSteps",
                "mjcf": str(args.mjcf), "shape": list(table.shape),
                "phase_unit": "radian", "target_unit": "radian",
                "body_length_unit": "MJCF model unit (mm)",
                "neutral_phase": float(np.pi), "neutral_pose": "leg table sampled at phase pi; proboscis held at model.standing() qpos",
                "mapping": mapping, "servo_mapping": servo, "servo_target_count": 44,
                "adhesion_actuators": [n for n in actuator_names if n.endswith("-adhesion")],
                "adhesion_convention": "HybridTurningController stance gate: 1=enabled, 0=swing; six leg order lf,lm,lh,rf,rm,rh",
                "hybrid_swing_extension_rad": 0.7853981633974483,
                "unmapped_actuators": [n for n in actuator_names
                    if n not in {m["actuator_name"] for m in servo}]}
    args.out.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"exported {table.shape} targets, {len(mapping)} named joints, {model.nu} actuators")


if __name__ == "__main__":
    main()
