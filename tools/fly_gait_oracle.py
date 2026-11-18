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
from kinestride.model import standing

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



def _normal_name(name: str | None, root_alias: bool = False) -> str | None:
    if name is None:
        return None
    if root_alias and "/" not in name:
        return "@root"
    return name.split("/", 1)[-1]


def _obj_names(model, obj, count: int) -> list[str | None]:
    return [mujoco.mj_id2name(model, obj, i) for i in range(count)]


def diagnose_models(body_mjcf: Path, out: Path) -> None:
    """Record FlyGym's one-second CPG controls and compare/replay them on body-v2."""
    import json
    from flygym import Simulation
    from flygym.compose import FlatGroundWorld
    from flygym.utils.math import Rotation3D
    from flygym.anatomy import BodySegment
    from flygym_demo.complex_terrain import (CPGController, LocomotionAction, PreprogrammedSteps,
                                             apply_locomotion_action, make_locomotion_fly,
                                             make_tripod_cpg_network)

    out.mkdir(parents=True, exist_ok=True)
    fly = make_locomotion_fly(name="smoke", add_adhesion=True, colorize=False)
    world = FlatGroundWorld()
    world.add_fly(fly, [0, 0, 0.5], Rotation3D("quat", [1, 0, 0, 0]))
    sim = Simulation(world)
    source = sim.mj_model
    source_data = sim.mj_data
    steps = PreprogrammedSteps()
    dofs = fly.get_actuated_jointdofs_order("position")
    controller = CPGController(cpg_network=make_tripod_cpg_network(
        timestep=sim.timestep, intrinsic_amplitude=1.0, coupling_strength=10.0,
        convergence_coef=20.0, seed=0), preprogrammed_steps=steps, output_dof_order=dofs)
    sim.reset()
    apply_locomotion_action(sim, fly.name, LocomotionAction(
        joint_angles=steps.default_pose_by_dof_order(dofs), adhesion_onoff=np.ones(6, dtype=bool)))
    sim.warmup()
    start_q, start_v = source_data.qpos.copy(), source_data.qvel.copy()
    thorax_id = fly.get_bodysegs_order().index(BodySegment("c_thorax"))
    start_thorax = sim.get_body_positions(fly.name)[thorax_id].copy()
    nstep = round(1.0 / sim.timestep)
    ctrl_trace = np.empty((nstep, source.nu), np.float64)
    q_trace = np.empty((nstep + 1, source.nq), np.float64)
    q_trace[0] = start_q
    for t in range(nstep):
        apply_locomotion_action(sim, fly.name, controller.step())
        ctrl_trace[t] = source_data.ctrl
        sim.step()
        q_trace[t + 1] = source_data.qpos
    forward = float(sim.get_body_positions(fly.name)[thorax_id][0] - start_thorax[0])

    target = mujoco.MjModel.from_xml_path(str(body_mjcf))
    source_act = _obj_names(source, mujoco.mjtObj.mjOBJ_ACTUATOR, source.nu)
    target_act = _obj_names(target, mujoco.mjtObj.mjOBJ_ACTUATOR, target.nu)
    source_joints = _obj_names(source, mujoco.mjtObj.mjOBJ_JOINT, source.njnt)
    target_joints = _obj_names(target, mujoco.mjtObj.mjOBJ_JOINT, target.njnt)
    source_act_by_name = {_normal_name(n): i for i, n in enumerate(source_act)}
    target_act_by_name = {_normal_name(n): i for i, n in enumerate(target_act)}
    source_joint_by_name = {_normal_name(n, root_alias=True): i for i, n in enumerate(source_joints)}
    target_joint_by_name = {_normal_name(n, root_alias=True): i for i, n in enumerate(target_joints)}
    common_acts = sorted(source_act_by_name.keys() & target_act_by_name.keys())
    common_joints = sorted(source_joint_by_name.keys() & target_joint_by_name.keys())
    rows = ["# Compiled model comparison: FlyGym smoke vs body-v2", "",
            f"| Field | FlyGym | body-v2 |", "|---|---:|---:|",
            f"| nu / nq / njnt / ngeom / npair | {source.nu} / {source.nq} / {source.njnt} / {source.ngeom} / {source.npair} | {target.nu} / {target.nq} / {target.njnt} / {target.ngeom} / {target.npair} |",
            f"| timestep / integrator / noslip | {source.opt.timestep} / {int(source.opt.integrator)} / {source.opt.noslip_iterations} | {target.opt.timestep} / {int(target.opt.integrator)} / {target.opt.noslip_iterations} |",
            f"| gravity | {source.opt.gravity.tolist()} | {target.opt.gravity.tolist()} |",
            f"| warm-start root xyz | {start_q[:3].tolist()} | mapped from FlyGym root by freejoint |",
            f"| matched actuators / joints | {len(common_acts)} / {len(common_joints)} | {len(common_acts)} / {len(common_joints)} |",
            f"| one-second FlyGym forward displacement | {forward:.6f} model units | recorded as ctrl replay input |", "",
            "## Actuator order and parameters", "", "| source id | source actuator | target id | target actuator | gain max abs | bias max abs | ctrlrange max abs |", "|---:|---|---:|---|---:|---:|---:|"]
    for sn, si in source_act_by_name.items():
        if sn not in target_act_by_name:
            rows.append(f"| {si} | {source_act[si]} | - | missing | - | - | - |")
            continue
        ti = target_act_by_name[sn]
        gain = float(np.max(np.abs(source.actuator_gainprm[si] - target.actuator_gainprm[ti])))
        bias = float(np.max(np.abs(source.actuator_biasprm[si] - target.actuator_biasprm[ti])))
        ctrlrange = float(np.max(np.abs(source.actuator_ctrlrange[si] - target.actuator_ctrlrange[ti])))
        rows.append(f"| {si} | {source_act[si]} | {ti} | {target_act[ti]} | {gain:.6g} | {bias:.6g} | {ctrlrange:.6g} |")
    rows += ["", "## Joint order and qpos0", "", "| joint | FlyGym qposadr | body-v2 qposadr | qpos0 max abs |", "|---|---:|---:|---:|"]
    qdim = {int(mujoco.mjtJoint.mjJNT_FREE): 7, int(mujoco.mjtJoint.mjJNT_BALL): 4,
            int(mujoco.mjtJoint.mjJNT_SLIDE): 1, int(mujoco.mjtJoint.mjJNT_HINGE): 1}
    for jn in common_joints:
        si, ti = source_joint_by_name[jn], target_joint_by_name[jn]
        sa, ta = int(source.jnt_qposadr[si]), int(target.jnt_qposadr[ti])
        size = qdim[int(source.jnt_type[si])]
        err = float(np.max(np.abs(source.qpos0[sa:sa + size] - target.qpos0[ta:ta + size])))
        rows.append(f"| {jn} | {sa} | {ta} | {err:.6g} |")
    source_geom = {_normal_name(n): i for i, n in enumerate(_obj_names(source, mujoco.mjtObj.mjOBJ_GEOM, source.ngeom))}
    target_geom = {_normal_name(n): i for i, n in enumerate(_obj_names(target, mujoco.mjtObj.mjOBJ_GEOM, target.ngeom))}
    common_geoms = sorted(source_geom.keys() & target_geom.keys())
    fric_diff = [(n, float(np.max(np.abs(source.geom_friction[source_geom[n]] - target.geom_friction[target_geom[n]])))) for n in common_geoms]
    rows += ["", "## Geometry and explicit pairs", "",
             f"Matched named geoms: {len(common_geoms)}; missing from body-v2: {len(source_geom.keys() - target_geom.keys())}; extra in body-v2: {len(target_geom.keys() - source_geom.keys())}.",
             f"Matched geom friction max abs: {max((x[1] for x in fric_diff), default=0):.6g}.", "",
             "body-v2 pair names include arena walls/stone fixture pairs absent from FlatGroundWorld:", ""]
    for i, (g1, g2) in enumerate(zip(target.pair_geom1, target.pair_geom2)):
        n1 = mujoco.mj_id2name(target, mujoco.mjtObj.mjOBJ_GEOM, int(g1)) or str(g1)
        n2 = mujoco.mj_id2name(target, mujoco.mjtObj.mjOBJ_GEOM, int(g2)) or str(g2)
        pair_name = mujoco.mj_id2name(target, mujoco.mjtObj.mjOBJ_PAIR, i) or ""
        if any(tag in (n1 + " " + n2 + " " + pair_name).lower() for tag in ("wall", "stone", "arena", "ground_plane")):
            rows.append(f"- pair {i}: {pair_name}: {n1} x {n2}")
    (out / "private note model_diff").write_text("\n".join(rows) + "\n")

    # Transfer the exact warm state and controls by joint/actuator name, leaving body-v2's extra head servos at the
    # corresponding standing angle. Compare every common joint after each paired MuJoCo step.
    target_data = mujoco.MjData(target)
    target_data.qpos[:] = target.qpos0
    target_data.qvel[:] = 0
    for jn in common_joints:
        si, ti = source_joint_by_name[jn], target_joint_by_name[jn]
        qs, qt = int(source.jnt_qposadr[si]), int(target.jnt_qposadr[ti])
        vs, vt = int(source.jnt_dofadr[si]), int(target.jnt_dofadr[ti])
        nq = qdim[int(source.jnt_type[si])]
        nv = 6 if int(source.jnt_type[si]) == int(mujoco.mjtJoint.mjJNT_FREE) else 3 if int(source.jnt_type[si]) == int(mujoco.mjtJoint.mjJNT_BALL) else 1
        target_data.qpos[qt:qt + nq] = start_q[qs:qs + nq]
        target_data.qvel[vt:vt + nv] = start_v[vs:vs + nv]
    mujoco.mj_forward(target, target_data)
    source_replay = mujoco.MjData(source)
    source_replay.qpos[:] = start_q; source_replay.qvel[:] = start_v
    source_replay.ctrl[:] = ctrl_trace[0]
    mujoco.mj_forward(source, source_replay)

    def contact_rows(model, data):
        rows = []
        for c in range(data.ncon):
            con = data.contact[c]
            a = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(con.geom1)) or str(con.geom1)
            b = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(con.geom2)) or str(con.geom2)
            rows.append({"geom1": a, "geom2": b, "dist": float(con.dist)})
        return rows

    body_thorax = mujoco.mj_name2id(target, mujoco.mjtObj.mjOBJ_BODY, "fly/c_thorax")
    body_start_x = float(target_data.xpos[body_thorax, 0])
    body_q_trace = np.empty((nstep + 1, target.nq), np.float64)
    body_q_trace[0] = target_data.qpos
    contact_start = {"flygym": contact_rows(source, source_replay),
                     "body_v2": contact_rows(target, target_data)}
    target_ctrl_base = target_data.ctrl.copy()
    for name, ai in target_act_by_name.items():
        if name not in source_act_by_name:
            j = int(target.actuator_trnid[ai, 0])
            if j < target.njnt and int(target.actuator_trntype[ai]) == int(mujoco.mjtTrn.mjTRN_JOINT):
                target_ctrl_base[ai] = target_data.qpos[int(target.jnt_qposadr[j])]
    source_to_target = [(source_act_by_name[n], target_act_by_name[n]) for n in common_acts]
    first = None
    max_d = 0.0
    contact_after_first = None
    for t in range(nstep):
        source_replay.ctrl[:] = ctrl_trace[t]
        mujoco.mj_step(source, source_replay)
        target_data.ctrl[:] = target_ctrl_base
        for si, ti in source_to_target:
            target_data.ctrl[ti] = ctrl_trace[t, si]
        mujoco.mj_step(target, target_data)
        body_q_trace[t + 1] = target_data.qpos
        if t == 0:
            contact_after_first = {"flygym": contact_rows(source, source_replay),
                                   "body_v2": contact_rows(target, target_data)}
        for jn in common_joints:
            si, ti = source_joint_by_name[jn], target_joint_by_name[jn]
            qs, qt = int(source.jnt_qposadr[si]), int(target.jnt_qposadr[ti])
            nq = qdim[int(source.jnt_type[si])]
            src = q_trace[t + 1, qs:qs + nq]
            dst = target_data.qpos[qt:qt + nq]
            error = float(np.max(np.abs(src - dst)))
            max_d = max(max_d, error)
            if error > 1e-6 and first is None:
                k = int(np.argmax(np.abs(src - dst)))
                first = {"step": t + 1, "joint": jn, "component": k,
                         "flygym_qpos": float(src[k]), "body_v2_qpos": float(dst[k]), "abs_error": error}
    from flygym_demo.complex_terrain import HybridTurningController
    from flygym_demo.complex_terrain.hybrid_controller import HybridControllerObservation

    def turn_trial(signal):
        turn_fly = make_locomotion_fly(name="smoke", add_adhesion=True, colorize=False)
        turn_world = FlatGroundWorld()
        turn_world.add_fly(turn_fly, [0, 0, 0.5], Rotation3D("quat", [1, 0, 0, 0]))
        turn_sim = Simulation(turn_world)
        turn_model, turn_data = turn_sim.mj_model, turn_sim.mj_data
        turn_steps = PreprogrammedSteps()
        turn_dofs = turn_fly.get_actuated_jointdofs_order("position")
        turn_ctl = HybridTurningController(timestep=turn_sim.timestep,
                                           preprogrammed_steps=turn_steps,
                                           output_dof_order=turn_dofs)
        turn_sim.reset()
        apply_locomotion_action(turn_sim, turn_fly.name, LocomotionAction(
            joint_angles=turn_steps.default_pose_by_dof_order(turn_dofs),
            adhesion_onoff=np.ones(6, dtype=bool)))
        turn_sim.warmup()
        obs = HybridControllerObservation(thorax_z=0.0, tarsus5_z=np.zeros(6),
                                          stumbling_contact_forces=np.zeros((6, 3, 3)),
                                          fly_heading=np.array([1.0, 0.0, 0.0]))
        tq, tv = turn_data.qpos.copy(), turn_data.qvel.copy()
        replay = mujoco.MjData(target)
        replay.qpos[:] = target.qpos0
        replay.qvel[:] = 0
        for jn, si in source_joint_by_name.items():
            if jn not in target_joint_by_name:
                continue
            ti = target_joint_by_name[jn]
            qs, qt = int(turn_model.jnt_qposadr[si]), int(target.jnt_qposadr[ti])
            vs, vt = int(turn_model.jnt_dofadr[si]), int(target.jnt_dofadr[ti])
            nq = qdim[int(turn_model.jnt_type[si])]
            nv = 6 if int(turn_model.jnt_type[si]) == int(mujoco.mjtJoint.mjJNT_FREE) else 3 if int(turn_model.jnt_type[si]) == int(mujoco.mjtJoint.mjJNT_BALL) else 1
            replay.qpos[qt:qt + nq] = tq[qs:qs + nq]
            replay.qvel[vt:vt + nv] = tv[vs:vs + nv]
        mujoco.mj_forward(target, replay)
        base = replay.ctrl.copy()
        for name, ai in target_act_by_name.items():
            if name not in source_act_by_name:
                jid = int(target.actuator_trnid[ai, 0])
                if int(target.actuator_trntype[ai]) == int(mujoco.mjtTrn.mjTRN_JOINT):
                    base[ai] = replay.qpos[int(target.jnt_qposadr[jid])]
        src_root = mujoco.mj_name2id(turn_model, mujoco.mjtObj.mjOBJ_BODY, "smoke")
        dst_root = mujoco.mj_name2id(target, mujoco.mjtObj.mjOBJ_BODY, "fly")
        src_thorax = mujoco.mj_name2id(turn_model, mujoco.mjtObj.mjOBJ_BODY, "smoke/c_thorax")
        dst_thorax = mujoco.mj_name2id(target, mujoco.mjtObj.mjOBJ_BODY, "fly/c_thorax")
        def yaw_of(data, body):
            mat = data.xmat[body].reshape(3, 3)
            return float(np.arctan2(mat[1, 0], mat[0, 0]))
        sy, ty = yaw_of(turn_data, src_root), yaw_of(replay, dst_root)
        syaw = tyaw = 0.0
        sx, tx = float(turn_data.xpos[src_thorax, 0]), float(replay.xpos[dst_thorax, 0])
        for _ in range(nstep):
            action = turn_ctl.step(np.asarray(signal, np.float64), obs)
            apply_locomotion_action(turn_sim, turn_fly.name, action)
            # apply_locomotion_action has set the exact ctrl vector for this tick.
            for name, ti in target_act_by_name.items():
                if name in source_act_by_name:
                    replay.ctrl[ti] = turn_data.ctrl[source_act_by_name[name]]
            mujoco.mj_step(target, replay)
            turn_sim.step()
            ns, nt = yaw_of(turn_data, src_root), yaw_of(replay, dst_root)
            syaw += float(np.arctan2(np.sin(ns - sy), np.cos(ns - sy)))
            tyaw += float(np.arctan2(np.sin(nt - ty), np.cos(nt - ty)))
            sy, ty = ns, nt
        return {"signal": list(signal), "flygym_yaw_rad": syaw, "body_v2_yaw_rad": tyaw,
                "flygym_dx": float(turn_data.xpos[src_thorax, 0] - sx),
                "body_v2_dx": float(replay.xpos[dst_thorax, 0] - tx)}

    turns = [turn_trial((1.0, 1.0)), turn_trial((0.0, 0.0)),
             turn_trial((1.0, -1.0)), turn_trial((-1.0, 1.0))]
    body_thorax = mujoco.mj_name2id(target, mujoco.mjtObj.mjOBJ_BODY, "fly/c_thorax")
    body_end_x = float(target_data.xpos[body_thorax, 0])
    body_forward = body_end_x - body_start_x
    body_root_joint = int(target.body_jntadr[int(target.body_rootid[body_thorax])])
    body_root_qadr = int(target.jnt_qposadr[body_root_joint])
    from scipy.spatial.transform import Rotation
    root_quat = body_q_trace[:, body_root_qadr + 3:body_root_qadr + 7]
    body_rot = Rotation.from_quat(np.column_stack((root_quat[:, 1:], root_quat[:, 0])))
    body_tilt_trace = np.degrees(np.arccos(np.clip(body_rot.as_matrix()[:, 2, 2], -1, 1)))
    body_tilt_max = float(np.max(body_tilt_trace))
    np.savez_compressed(out / "flygym_rollout.npz", ctrl=ctrl_trace, qpos=q_trace,
                        body_v2_qpos=body_q_trace,
                        start_qpos=start_q, start_qvel=start_v,
                        actuator_names=np.asarray([n or "" for n in source_act]),
                        joint_names=np.asarray([n or "" for n in source_joints]))
    receipt = {"flygym_forward_model_units": forward, "steps": nstep, "source_model": {
        "nu": source.nu, "nq": source.nq, "njnt": source.njnt, "ngeom": source.ngeom, "npair": source.npair},
        "target_model": {"nu": target.nu, "nq": target.nq, "njnt": target.njnt, "ngeom": target.ngeom, "npair": target.npair},
        "common_actuators": len(common_acts), "common_joints": len(common_joints),
        "first_divergence": first, "max_abs_qpos_error_over_rollout": max_d,
        "body_v2_forward_displacement": body_forward, "body_v2_max_root_tilt_deg": body_tilt_max,
        "opposite_turns": turns,
        "active_contacts_start": contact_start, "active_contacts_after_first_step": contact_after_first}
    (out / "replay.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if first:
        print(f"FIRST DIVERGENCE step={first['step']} joint={first['joint']} component={first['component']} "
              f"FlyGym={first['flygym_qpos']:.9g} body-v2={first['body_v2_qpos']:.9g} abs={first['abs_error']:.9g}")
        print(f"BODY-V2 replay 1s forward displacement={body_forward:.6g} model units; "
              f"max root tilt={body_tilt_max:.4g} deg; max abs qpos error={max_d:.6g}")
    else:
        print(f"No mapped qpos divergence > 1e-6 over {nstep} steps; max_abs={max_d:.9g}")
    print(f"opposite-turn replay: {turns}")
    print(f"model table: {out / 'private note model_diff'}; ctrl replay and receipt: {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mjcf", type=Path, nargs="?", help="body-v2/body.xml")
    parser.add_argument("out", type=Path, nargs="?", help="output .npz")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--check-export", type=Path, help="verify a prior export against the installed FlyGym oracle")
    parser.add_argument("--check-controller", action="store_true", help="compare mapped targets against HybridTurningController")
    parser.add_argument("--rollout", type=Path, help="one-second body-v2 MuJoCo morphology preflight")
    parser.add_argument("--diagnose", type=Path, help="compare compiled FlyGym/body-v2 models and replay FlyGym CPG controls")
    parser.add_argument("--out-dir", type=Path, default=Path(".work/team/status/flyloop-hooks/model-diff"))
    args = parser.parse_args()
    if args.diagnose:
        diagnose_models(args.diagnose, args.out_dir)
        return
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
