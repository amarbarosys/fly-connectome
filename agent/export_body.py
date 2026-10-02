"""Export the complete fly and world, checking compiled mechanical equivalence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from agent.neural_single import EthologyArena


FIELDS = (
    'body_parentid', 'body_pos', 'body_quat', 'body_mass', 'body_inertia',
    'body_ipos', 'body_iquat', 'jnt_type', 'jnt_bodyid', 'jnt_pos', 'jnt_axis',
    'jnt_range', 'jnt_limited', 'dof_armature', 'dof_damping', 'dof_frictionloss',
    'geom_type', 'geom_bodyid', 'geom_pos', 'geom_quat', 'geom_size',
    'geom_contype', 'geom_conaffinity', 'geom_friction', 'geom_solref', 'geom_solimp',
    'mesh_vert', 'mesh_face', 'pair_geom1', 'pair_geom2', 'pair_friction',
    'pair_solref', 'pair_solimp', 'pair_margin', 'pair_gap',
    'actuator_trntype', 'actuator_trnid', 'actuator_gainprm', 'actuator_biasprm',
    'actuator_gear', 'actuator_ctrlrange', 'actuator_forcerange', 'qpos0',
)


def export(output: Path):
    output.mkdir(parents=True, exist_ok=False)
    arena = EthologyArena(seed=0)
    try:
        source = arena.sim.mj_model
        arena.world.save_xml_with_assets(output, 'body.xml')
        xml_path = output / 'body.xml'
        tree = ET.parse(xml_path)
        option = tree.getroot().find('option')
        if option is None:
            option = ET.SubElement(tree.getroot(), 'option')
        option.set('timestep', repr(float(source.opt.timestep)))
        # XML serialization rounds inferred inertias. Preserve the actual compiled
        # body frames and inertials rather than recomputing them from rounded XML.
        def values(array):
            return ' '.join(format(float(v), '.17g') for v in array)
        for body in tree.getroot().iter('body'):
            body_id = mujoco.mj_name2id(source, mujoco.mjtObj.mjOBJ_BODY, body.get('name', ''))
            if body_id < 0:
                continue
            body.set('pos', values(source.body_pos[body_id]))
            for rotation in ('euler', 'axisangle', 'xyaxes', 'zaxis'):
                body.attrib.pop(rotation, None)
            body.set('quat', values(source.body_quat[body_id]))
            if source.body_mass[body_id] > 0:
                inertial = body.find('inertial')
                if inertial is None:
                    inertial = ET.SubElement(body, 'inertial')
                inertial.attrib.clear()
                inertial.attrib.update(pos=values(source.body_ipos[body_id]),
                    quat=values(source.body_iquat[body_id]),
                    mass=format(float(source.body_mass[body_id]), '.17g'),
                    diaginertia=values(source.body_inertia[body_id]))
        tree.write(xml_path, encoding='unicode')
        restored = mujoco.MjModel.from_xml_path(str(xml_path.resolve()))
        checks = {}
        for name in FIELDS:
            before, after = getattr(source, name), getattr(restored, name)
            checks[name] = bool(before.shape == after.shape and np.allclose(
                before, after, rtol=1e-10, atol=1e-12))
        for name in ('timestep', 'gravity', 'integrator', 'solver', 'iterations'):
            checks['option.' + name] = bool(np.array_equal(
                getattr(source.opt, name), getattr(restored.opt, name)))
        names = {}
        for kind, count in (('BODY', source.nbody), ('JOINT', source.njnt),
                            ('ACTUATOR', source.nu), ('GEOM', source.ngeom)):
            obj = getattr(mujoco.mjtObj, 'mjOBJ_' + kind)
            names[kind.lower()] = [mujoco.mj_id2name(source, obj, i) for i in range(count)]
            checks[kind.lower() + '_names'] = names[kind.lower()] == [
                mujoco.mj_id2name(restored, obj, i) for i in range(count)]
        actuators = []
        for i in range(source.nu):
            actuators.append({'name': names['actuator'][i],
                              'transmission': mujoco.mjtTrn(int(source.actuator_trntype[i])).name,
                              'target_ids': source.actuator_trnid[i].tolist(),
                              'gain': source.actuator_gainprm[i].tolist()})
        np.savez(output / 'initial-state.npz', qpos=arena.sim.mj_data.qpos,
                 qvel=arena.sim.mj_data.qvel, ctrl=arena.sim.mj_data.ctrl)
        receipt = {
            'status': 'PASS' if all(checks.values()) else 'FAIL',
            'scope': 'self-contained MJCF export and MuJoCo mechanical round trip; GPU port unverified',
            'mujoco_version': mujoco.__version__, 'units': 'existing FlyGym mm-based model; no rescaling',
            'counts': {name: int(getattr(source, name)) for name in
                       ('nbody', 'njnt', 'nq', 'nv', 'nu', 'ngeom', 'nmesh', 'npair', 'nsensor')},
            'checks': checks, 'actuators': actuators, 'names': names,
            'files_sha256': {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(output.iterdir()) if p.is_file()},
            'required_backend_features': ['BODY adhesion transmission', 'explicit contact pairs',
                'per-pair friction and solver parameters', 'unmodified body inertia and joint limits',
                'joint position actuators', 'existing CPG and reflex motor controller'],
        }
        (output / 'manifest.json').write_text(json.dumps(receipt, indent=2) + '\n')
        if receipt['status'] != 'PASS':
            raise RuntimeError('Mechanical round trip failed: ' + repr(
                [name for name, passed in checks.items() if not passed]))
        return receipt
    finally:
        arena.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    receipt = export(args.output)
    print(json.dumps({'status': receipt['status'], 'counts': receipt['counts']}))
