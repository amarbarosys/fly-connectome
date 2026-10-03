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
       python -m agent.body_engine.model meshes MJCF OUT_DIR   (renderer tables; fails above 1e-4 decode error)
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
# 'contact' (S2 probe): plane-mesh pairs whose mesh geom matches FEET, pyramidal cone, Newton solver, then the
# implicit Euler step. States come from an actuated MuJoCo rollout (feet on the floor), not random poses.
STAGES['contact'] = (6, 'mj_step', STAGES['step'][2] + ('ncon', 'qacc', 'qfrc_constraint'))
FEET = 'tarsus5'
VSCALE = 10  # random qvel in +-10 per dof: velocity terms well above float32 noise; qpos moves ~1e-3 per fly step
REL = 1e-5
FLOOR_X = 3  # contact stage: bar = max(REL, 3 x the field's float32 input floor), see check()
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


def feet(m, regex):
    """Contact pairs (plane geom1, mesh geom2 named by regex): ints, floats and -D sizes for smooth.mojo STAGE 6.

    ints   per pair: plane geom, mesh geom, mesh body, hull vertex count, hull vertex offset, edge-address offset,
           chain length, chain offset (8); then every hull vertex's first edge (absolute, into the edge list), the
           edge lists (hull-local vertex ids, -1 ends a vertex), the dof chains (ancestor dofs of the body, ascending)
           and opt.iterations, opt.ls_iterations
    floats per pair: margin, gap, friction 0-1, solref 2, solimp 5, invweight0 translation (both bodies), rbound (13);
           then hull vertices in the geom frame; then opt.impratio, opt.tolerance, opt.ls_tolerance
    """
    import re
    if m.opt.cone != mujoco.mjtCone.mjCONE_PYRAMIDAL or (m.geom_contype & m.geom_conaffinity).any():
        raise SystemExit('FAIL export: contacts need the pyramidal cone and explicit pairs only')
    pairs = [i for i in range(m.npair) if m.geom_type[m.pair_geom1[i]] == mujoco.mjtGeom.mjGEOM_PLANE
             and m.geom_type[m.pair_geom2[i]] == mujoco.mjtGeom.mjGEOM_MESH and re.search(regex, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, m.pair_geom2[i]))]
    if any(m.pair_dim[i] != 3 for i in pairs):
        raise SystemExit('FAIL export: only condim 3 contacts are ported')
    head, first, edges, chains, fl, verts = [], [], [], [], [], []
    for i in pairs:
        g1, g2 = m.pair_geom1[i], m.pair_geom2[i]
        mesh, b = m.geom_dataid[g2], m.geom_bodyid[g2]
        ga = m.mesh_graphadr[mesh]
        nv = m.mesh_graph[ga]
        edgeadr = m.mesh_graph[ga + 2:ga + 2 + nv]
        gid = m.mesh_graph[ga + 2 + nv:ga + 2 + 2 * nv]
        local = m.mesh_graph[ga + 2 + 2 * nv:]
        local = local[:edgeadr.max() + list(local[edgeadr.max():]).index(-1) + 1]  # through the last vertex's -1
        chain, bb = [], b
        while bb > 0:
            chain = list(range(m.body_dofadr[bb], m.body_dofadr[bb] + m.body_dofnum[bb])) + chain
            bb = m.body_parentid[bb]
        if any(m.M_rownnz[c] != k + 1 for k, c in enumerate(chain)) or m.pair_solimp[i][4] != int(m.pair_solimp[i][4]):
            raise SystemExit('FAIL export: a foot dof chain is not its own M row pattern, or solimp power is not whole')
        head += [g1, g2, b, nv, len(verts) // 3, len(first), len(chain), len(chains)]
        first += list(len(edges) + edgeadr)
        edges += list(local)
        chains += chain
        verts += list(m.mesh_vert[m.mesh_vertadr[mesh] + gid].ravel())
        fl += [m.pair_margin[i], m.pair_gap[i], *m.pair_friction[i][:2], *m.pair_solref[i], *m.pair_solimp[i],
               m.body_invweight0[m.geom_bodyid[g1], 0] + m.body_invweight0[b, 0], m.geom_rbound[g2]]
    ints = np.array(head + first + edges + chains + [m.opt.iterations, m.opt.ls_iterations])
    floats = np.array(fl + verts + [m.opt.impratio, m.opt.tolerance, m.opt.ls_tolerance])
    sizes = {'NCP': len(pairs), 'NHV': len(verts) // 3, 'NHE': len(edges), 'NCH': len(chains)}
    return ints, floats, sizes


def export(mjcf, out, regex=None):
    m, out = load(mjcf), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ints = [getattr(m, f).ravel() for f in INTS]
    floats = [np.ravel(getattr(m.opt, f[4:]) if f.startswith('opt.') else getattr(m, f)) for f in FLOATS]
    sizes = {'NBODY': m.nbody, 'NJNT': m.njnt, 'NQ': m.nq, 'NGEOM': m.ngeom, 'NV': m.nv, 'NC': m.nC}
    if regex:
        ci, cf, cs = feet(m, regex)
        ints, floats, sizes = ints + [ci], floats + [cf], sizes | cs
    np.concatenate(ints).astype(np.int32).tofile(out / 'mi.bin')
    np.concatenate(floats).astype(np.float32).tofile(out / 'mf.bin')
    (out / 'sizes.txt').write_text(' '.join(f'-D{k}={v}' for k, v in sizes.items()) + '\n')
    print(f'{mjcf}: {sizes} -> {out}')
    return m


def meshes(mjcf, out):
    """Renderer tables (docs/plans/RENDERER.md) for every mesh geom: rmi.bin int32 per part (body, vertex offset,
    vertex count, triangle offset, triangle count), rmf.bin float32 per part (local pos 3, local quat 4, box
    offset 3, box scale 3, bounding sphere centre 3 and radius 1, rgba 4), rv.bin int16 vertices quantised to the
    part's box, rt.bin uint16 triangles local to the part, rp.bin uint8 part per triangle. Returns the decode error relative to part size."""
    m, out = load(mjcf), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ints, floats, verts, tris, part, worst = [], [], [], [], [], 0.0
    for g in np.flatnonzero(m.geom_type == mujoco.mjtGeom.mjGEOM_MESH):
        k = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k] + m.mesh_vertnum[k]].astype(np.float64)
        f = m.mesh_face[m.mesh_faceadr[k]:m.mesh_faceadr[k] + m.mesh_facenum[k]]
        if len(v) > 65535 or len(ints) // 5 >= 255:
            raise SystemExit(f'FAIL meshes: geom {g}: over 65,535 vertices (uint16 triangles) or 255 parts (uint8 part ids)')
        lo, hi = v.min(0), v.max(0)
        scale = np.maximum(hi - lo, 1e-12) / 65535
        qv = np.round((v - lo) / scale - 32768).astype(np.int16)
        worst = max(worst, float(np.abs((qv.astype(np.float64) + 32768) * scale + lo - v).max() / (hi - lo).max()))
        rgba = m.mat_rgba[m.geom_matid[g]] if m.geom_matid[g] >= 0 else m.geom_rgba[g]
        centre = (lo + hi) / 2
        ints += [m.geom_bodyid[g], len(verts), len(v), len(tris), len(f)]
        floats += [*m.geom_pos[g], *m.geom_quat[g], *(lo + 32768 * scale), *scale, *centre, np.linalg.norm(v - centre, axis=1).max(), *rgba]
        verts += list(qv)
        tris += list(f.astype(np.uint16))
        part += [len(ints) // 5 - 1] * len(f)
    np.array(ints, np.int32).tofile(out / 'rmi.bin')
    np.array(floats, np.float32).tofile(out / 'rmf.bin')
    np.array(verts, np.int16).tofile(out / 'rv.bin')
    np.array(tris, np.uint16).tofile(out / 'rt.bin')
    np.array(part, np.uint8).tofile(out / 'rp.bin')  # part of each triangle (69 parts here; export refuses 256+)
    (out / 'rsizes.txt').write_text(f'-DNPART={len(ints) // 5} -DNRV={len(verts)} -DNRT={3 * len(tris)}\n')  # NRT: indices
    print(f'{mjcf}: {len(ints) // 5} parts, {len(verts)} vertices, {len(tris)} triangles, '
          f'{(len(verts) * 6 + len(tris) * 6 + len(part) + len(floats) * 4 + len(ints) * 4) / 1024:.0f} KB; decode error {worst:.1e} of part size')
    return worst


def standing(m, count, seed=0):
    """(qpos, qvel) of count states from actuated rollouts of the full model, adhesion on, one state every 2 ms after
    0.2 s of settling: half standing still (all 6 feet down), half with servos jittered +-0.2 rad every 50 ms
    (1 to 5 feet down)."""
    rng, q, v = np.random.default_rng(seed), [], []
    for jitter, n in ((0.0, count // 2), (0.2, count - count // 2)):
        d = mujoco.MjData(m)
        for k in range(2000 + 20 * n):
            if k % 500 == 0:
                d.ctrl[:] = np.where(m.actuator_trntype == mujoco.mjtTrn.mjTRN_BODY, 1.0, rng.uniform(-jitter, jitter, m.nu))
            mujoco.mj_step(m, d)
            if k >= 2000 and k % 20 == 0:
                q.append(d.qpos.copy()); v.append(d.qvel.copy())
    return np.array(q), np.array(v)


def check(mjcf, out, stage, count=64):
    m, out = export(mjcf, out, FEET if stage == 'contact' else None), Path(out)
    number, fn, fields = STAGES[stage]
    q, v = standing(m, count) if stage == 'contact' else (states(m, count), velocities(m, count, scale=VSCALE))
    if stage == 'contact':  # only the exported foot pairs collide; actuators and noslip (not ported) off
        s = load(mjcf)
        s.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ACTUATION
        s.opt.noslip_iterations = 0
        import re
        for i in range(s.npair):
            if not re.search(FEET, mujoco.mj_id2name(s, mujoco.mjtObj.mjOBJ_GEOM, s.pair_geom2[i])):
                s.pair_margin[i] = -1e10
        want = reference(s, fn, tuple(f for f in fields if f not in ('qacc_euler', 'qpos_next')) + ('qpos', 'qvel'), q, v)
        want['qacc_euler'] = (want.pop('qvel') - v) / m.opt.timestep
        want['qpos_next'] = want.pop('qpos')
        d = mujoco.MjData(s)  # ncon: contacts that carry constraint rows (MuJoCo also lists excluded ones)
        for n in range(count):
            d.qpos[:], d.qvel[:] = q[n], v[n]
            mujoco.mj_forward(s, d)
            want['ncon'][n] = sum(c.efc_address >= 0 for c in d.contact[:d.ncon])
        # float32 floor: MuJoCo on the float32-rounded state (the engine's actual input) vs MuJoCo on the float64 state.
        # A contact force at solref 2e-4 moves 2.4e-5 of its scale from input rounding alone, so a field's bar is
        # max(REL, FLOOR_X x its floor); every field still has to beat 1e-5 unless float32 inputs cannot.
        r32 = lambda x: x.astype(np.float32).astype(np.float64)
        alt = reference(s, fn, ('qacc', 'qfrc_constraint'), r32(q), r32(v))
        floor = {f: float(np.abs(alt[f] - want[f]).max() / np.abs(want[f]).max()) for f in alt}
        print('float32 input floor: ' + ', '.join(f'{f} {e:.2e}' for f, e in floor.items()))
    elif stage == 'step':
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
    floor = locals().get('floor', {})
    bad = [b for f, g in zip(fields, got) for b in compare({f: (want[f].reshape(count, -1), g)}, max(REL, FLOOR_X * floor.get(f, 0)))]
    if bad:
        print(f'FAIL {len(bad)}/{len(fields)} {stage} fields: {" ".join(bad)}')
        sys.exit(1)
    lifted = ', '.join(f'{f} {FLOOR_X * e:.1e}' for f, e in floor.items() if FLOOR_X * e > REL)
    print(f'PASS {stage}: {len(fields)} fields x {count} states within {REL:g}' + (f' (float32-floor bars: {lifted})' if lifted else ''))


if __name__ == '__main__':
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == 'meshes':
        sys.exit(0 if meshes(*args) <= 1e-4 else 'FAIL meshes: decode error above 1e-4 of part size')
    export(*args) if cmd == 'export' else check(args[0], args[1], args[2], *map(int, args[3:]))
