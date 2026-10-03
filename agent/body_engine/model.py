"""Any MuJoCo model (fly, spider, any MJCF body) to the engine's two flat arrays, plus the oracle that
checks each ported stage against MuJoCo itself.

The engine reads no MJCF. `export` writes mi.bin (int32) and mf.bin (float32): the mjModel fields a
ported stage needs, by their MuJoCo names, concatenated in the order of INTS / FLOATS. Sizes go to
sizes.txt as -D flags, so each body type gets its own compiled engine with its loop bounds and array
offsets as constants (the shiu_worker pattern). A new species is a new MJCF file, not new code.

`check STAGE`: random states from ~/iTools/science/mj-oracle (free joint anywhere, any orientation;
hinges and slides inside their ranges), MuJoCo runs the stage in float64, the engine runs it in
float32 on the GPU, and every output field must agree to REL of its own scale (mj-oracle compare).

usage: python -m agent.body_engine.model export MJCF OUT_DIR
       python -m agent.body_engine.model check  MJCF OUT_DIR STAGE [STATES]
"""
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path.home() / 'iTools/science/mj-oracle'))
from mj_oracle import compare, reference, states, velocities  # noqa: E402  (random states, MuJoCo reference, per-field compare)

INTS = ('body_parentid', 'body_jntadr', 'body_jntnum', 'jnt_type', 'jnt_qposadr', 'geom_bodyid',
        'body_rootid', 'jnt_dofadr', 'dof_parentid', 'dof_bodyid', 'M_rowadr', 'M_rownnz', 'M_colind',
        'body_dofnum', 'body_dofadr')
FLOATS = ('body_pos', 'body_quat', 'jnt_pos', 'jnt_axis', 'qpos0', 'body_ipos', 'body_iquat', 'geom_pos', 'geom_quat',
          'body_mass', 'body_subtreemass', 'body_inertia', 'dof_armature',
          'dof_damping', 'jnt_stiffness', 'qpos_spring', 'opt.gravity', 'opt.timestep')
KIN = ('xpos', 'xquat', 'xipos', 'geom_xpos', 'xanchor', 'xaxis')
STAGES = {  # name: (STAGE number in smooth.mojo, MuJoCo functions run in order, output fields)
    'kinematics': (1, 'mj_kinematics', KIN),
    'crb': (2, 'mj_kinematics,mj_comPos,mj_crb', KIN + ('subtree_com', 'cinert', 'cdof', 'crb', 'M')),
    'factorM': (3, 'mj_kinematics,mj_comPos,mj_crb,mj_factorM', KIN + ('subtree_com', 'cinert', 'cdof', 'crb', 'M', 'qLD', 'qLDiagInv')),
    'rne': (4, 'mj_kinematics,mj_comPos,mj_crb,mj_factorM,mj_fwdVelocity',  # mj_comVel, mj_passive, mj_rne(flg_acc 0)
            KIN + ('subtree_com', 'cinert', 'cdof', 'crb', 'M', 'qLD', 'qLDiagInv', 'cvel', 'cdof_dot', 'qfrc_passive', 'qfrc_bias')),
}
# 'step': one mj_step with contacts, constraints and actuators off (S1 is contact-free): mj_Euler with implicit
# dof damping. qacc_euler = (qvel after - qvel) / dt and qpos_next are derived from MuJoCo's own step.
STAGES['step'] = (5, 'mj_step', STAGES['rne'][2] + ('qH', 'qHDiagInv', 'qacc_euler', 'qpos_next'))
VSCALE = 10  # random qvel in +-10 per dof: velocity terms well above float32 noise; qpos moves ~1e-3 per fly step
REL = 1e-5
WORLD_FIRST = ('xpos', 'xquat', 'xipos', 'cinert', 'crb', 'cvel')  # body-indexed fields whose row 0 is the world
ENGINE = Path(__file__).parent


def load(mjcf):
    m = mujoco.MjModel.from_xml_path(str(mjcf))
    if m.nmocap or m.ntendon or m.dof_simplenum.any():
        raise SystemExit('FAIL export: mocap bodies, tendons and simple dofs are not ported')
    if (m.nflex or m.opt.integrator or m.opt.density or m.opt.viscosity or m.body_gravcomp.any() or m.actuator_damping.any()
            or m.dof_dampingpoly.any() or m.jnt_stiffnesspoly.any() or m.opt.disableflags):
        raise SystemExit('FAIL export: only Euler, linear joint springs and dof dampers are ported (no flex, fluid, gravcomp, '
                         'actuator damping, polynomial springs or dampers, disable flags)')
    return m


def export(mjcf, out):
    m, out = load(mjcf), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.concatenate([getattr(m, f).ravel() for f in INTS]).astype(np.int32).tofile(out / 'mi.bin')
    np.concatenate([np.ravel(getattr(m.opt, f[4:]) if f.startswith('opt.') else getattr(m, f)) for f in FLOATS]).astype(np.float32).tofile(out / 'mf.bin')
    sizes = {'NBODY': m.nbody, 'NJNT': m.njnt, 'NQ': m.nq, 'NGEOM': m.ngeom, 'NV': m.nv, 'NC': m.nC}
    (out / 'sizes.txt').write_text(' '.join(f'-D{k}={v}' for k, v in sizes.items()) + '\n')
    print(f'{mjcf}: {sizes} -> {out}')
    return m


def check(mjcf, out, stage, count=64):
    m, out = export(mjcf, out), Path(out)
    number, fn, fields = STAGES[stage]
    q, v = states(m, count), velocities(m, count, scale=VSCALE)
    if stage == 'step':
        s = load(mjcf)
        s.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT | mujoco.mjtDisableBit.mjDSBL_CONSTRAINT | mujoco.mjtDisableBit.mjDSBL_ACTUATION
        want = reference(s, fn, fields[:-2] + ('qpos', 'qvel'), q, v)
        want['qacc_euler'] = (want.pop('qvel') - v) / m.opt.timestep
        want['qpos_next'] = want.pop('qpos')
    else:
        want = reference(m, fn, fields, q, v)
    want = {f: v[:, 1:] if f in WORLD_FIRST else v for f, v in want.items()}  # the engine skips body 0, the world
    q.astype(np.float32).T.copy().tofile(out / 'qpos.bin')  # (nq, flies): flies fastest, as the engine reads it
    v.astype(np.float32).T.copy().tofile(out / 'qvel.bin')  # (nv, flies)
    binary = out / f'{stage}_{count}'
    flags = (out / 'sizes.txt').read_text().split()
    subprocess.run([str(Path.home() / '.local/bin/mojo'), 'build', *flags, f'-DFLIES={count}', f'-DSTAGE={number}', str(ENGINE / 'smooth.mojo'), '-o', str(binary)], check=True)
    subprocess.run([str(binary), str(out) + '/'], check=True)
    rows = np.fromfile(out / 'smooth.bin', np.float32).reshape(-1, count).T
    sizes = [want[f][0].size for f in fields]
    if rows.shape[1] != sum(sizes):
        raise SystemExit(f'FAIL {stage}: engine wrote {rows.shape[1]} values per fly, MuJoCo {sum(sizes)}')
    got = np.split(rows, np.cumsum(sizes)[:-1], axis=1)
    bad = compare({f: (want[f].reshape(count, -1), g) for f, g in zip(fields, got)}, REL)
    if bad:
        print(f'FAIL {len(bad)}/{len(fields)} {stage} fields: {" ".join(bad)}')
        sys.exit(1)
    print(f'PASS {stage}: {len(fields)} fields x {count} random states within {REL:g}')


if __name__ == '__main__':
    cmd, args = sys.argv[1], sys.argv[2:]
    export(*args) if cmd == 'export' else check(args[0], args[1], args[2], *map(int, args[3:]))
