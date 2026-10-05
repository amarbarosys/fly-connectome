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
import os
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
# 'actuated' (S3): + mj_transmission (hinge/slide joints, adhesion bodies) and mj_fwdActuation (fixed gain, affine
# or no bias, no activation dynamics), random controls, actuators on.
STAGES['actuated'] = (7, 'mj_step', STAGES['contact'][2] + ('actuator_force', 'qfrc_actuator'))
FEET = '.'  # every plane-mesh pair (55 on the fly: each body part on the floor); 'tarsus5' (6 feet) sank through the floor in MuJoCo too
VSCALE = float(os.environ.get('VSCALE', 10))  # random qvel in +-VSCALE per dof (10: velocity terms well above float32 noise; qpos moves ~1e-3 per fly step); the flight gate draws up to flapping rate
REL = 1e-5
FLOOR_X = 3  # contact stage: bar = max(REL, 3 x the field's float32 input floor), see check()
WORLD_FIRST = ('xpos', 'xquat', 'xipos', 'cinert', 'crb', 'cvel')  # body-indexed fields whose row 0 is the world
ENGINE = Path(__file__).parent


def cube(mjcf, side=40.0):
    """A closed cube of side `side` (mm) around the arena: the ground plane plus 4 wall planes and a ceiling plane,
    normals inward, each colliding with every fly part the ground does (the ground pair's parameters copied). The
    old box walls, lid and stones and their pairs are removed, so every contact is plane-mesh (ported). Written next
    to mjcf (its meshes are relative paths); returns the new path."""
    import xml.etree.ElementTree as ET
    tree = ET.parse(mjcf)
    root = tree.getroot()
    world, contact = root.find('worldbody'), root.find('contact')
    ground = world.find("geom[@name='ground_plane']")
    for g in list(world.findall('geom')):
        if g is not ground and g.get('type') in ('box', 'ellipsoid') and not g.get('name', '').startswith('stripe'):
            world.remove(g)
    keep = [q for q in contact.findall('pair') if q.get('geom1') == 'ground_plane']
    for q in list(contact):
        contact.remove(q)
    h = side / 2
    faces = {'wall_xn': (f'{-h} 0 {h}', '1 0 0'), 'wall_xp': (f'{h} 0 {h}', '-1 0 0'), 'wall_yn': (f'0 {-h} {h}', '0 1 0'),
             'wall_yp': (f'0 {h} {h}', '0 -1 0'), 'ceiling': (f'0 0 {side}', '0 0 -1')}
    for name, (pos, z) in faces.items():
        g = ET.SubElement(world, 'geom', dict(ground.attrib))
        g.set('name', name); g.set('pos', pos); g.set('zaxis', z); g.set('size', f'{h} {h} 1')
        g.attrib.pop('quat', None)
    for name in ('ground_plane', *faces):
        for q in keep:
            r = ET.SubElement(contact, 'pair', dict(q.attrib))
            r.set('geom1', name); r.set('name', f"{q.get('geom2')}-{name}")
    out = Path(mjcf).with_name(f'cube{int(side)}.xml')
    tree.write(out)
    return str(out)


def terrain(mjcf, params=(0.1, 0.05, 0, 8.0, 0, 0, 2.0, 0.5), n=257, half=40.0):
    """Ground plane replaced by the world.py terrain: its 8 params ride in <custom><numeric name="terrain"> (what the
    engine evaluates) and an n x n heightfield of +-half mm samples them (what MuJoCo collides with). Writes
    terrain.xml next to mjcf and, with octaves 0 (a pure ramp), its exact twin terrain_plane.xml: the same tilted
    plane as a plane geom, the oracle of the tilt gate. Returns the paths."""
    import xml.etree.ElementTree as ET
    from agent.body_engine.world import full, height_grad
    xs = np.linspace(-half, half, n)
    X, Y = np.meshgrid(xs, xs)  # row r along +y, column c along +x (mjModel.hfield_data order)
    H = height_grad(X, Y, params)[0].astype(np.float64)
    lo, span = H.min(), max(np.ptp(H), 1e-6)
    out = []
    for kind in ('hfield', 'plane') if not params[4] else ('hfield',):
        tree = ET.parse(mjcf)
        root = tree.getroot()
        ground = root.find("worldbody/geom[@name='ground_plane']")
        ground.attrib.pop('quat', None)
        if kind == 'hfield':
            asset = root.find('asset') if root.find('asset') is not None else ET.SubElement(root, 'asset')
            ET.SubElement(asset, 'hfield', name='terrain', nrow=str(n), ncol=str(n), size=f'{half} {half} {span} 1',
                          elevation=' '.join(f'{e:.9g}' for e in ((H - lo) / span)[::-1].ravel()))  # XML rows run +y first
            ground.set('type', 'hfield'); ground.set('hfield', 'terrain'); ground.set('pos', f'0 0 {lo}')
            ground.attrib.pop('size', None)
            custom = root.find('custom') if root.find('custom') is not None else ET.SubElement(root, 'custom')
            ET.SubElement(custom, 'numeric', name='terrain', data=' '.join(map(str, full(params))))
        else:
            ground.set('pos', '0 0 0'); ground.set('zaxis', f'{-params[0]} {-params[1]} 1')
        path = Path(mjcf).with_name('terrain.xml' if kind == 'hfield' else 'terrain_plane.xml')
        tree.write(path)
        out.append(str(path))
    return out


def combine(fly, spider, offset=(8.0, 0.0, 0.0), name='world'):
    """Fly and spider in one model (one export, one engine binary): the spider's spec attached at `offset` with prefix
    sp_ (MjSpec.attach copies its assets, actuators and pairs), its own ground removed and its pairs retargeted to the
    shared ground_plane, mesh paths made absolute. MuJoCo 3.9 has no compiler/conflict (3.10), so every global option
    must already agree or the merge fails. Written next to the fly MJCF as NAME.xml; returns the path."""
    a, b = mujoco.MjSpec.from_file(str(fly)), mujoco.MjSpec.from_file(str(spider))
    oa, ob = a.compile().opt, b.compile().opt
    diff = [f for f in dir(oa) if not f.startswith('_') and not callable(getattr(oa, f)) and np.any(np.asarray(getattr(oa, f)) != np.asarray(getattr(ob, f)))
            and f not in ('enableflags', 'noslip_iterations')]  # the fly's win: energy reporting only, noslip 5
    if diff:
        raise SystemExit(f'FAIL combine: global options differ: {diff}')
    for spec, path in ((a, fly), (b, spider)):  # absolute mesh files: the merged MJCF loads from anywhere
        base = Path(path).parent / (spec.meshdir or '')
        for mesh in spec.meshes:
            mesh.file = str((base / mesh.file).resolve()) if mesh.file else mesh.file
        spec.meshdir = ''
    a.attach(b, frame=a.worldbody.add_frame(pos=list(offset)), prefix='sp_')
    a.delete(a.geom('sp_ground_plane'))
    for pair in a.pairs:
        if pair.geomname1 == 'sp_ground_plane':
            pair.geomname1 = 'ground_plane'
    out = Path(fly).with_name(f'{name}.xml')
    out.write_text(a.to_xml())
    return str(out)


def airborne(mjcf, spring=(2.0, 0.3, -0.04, 0.04)):
    """The flight MJCF as the tendon gate meets it (stage `actuated` needs the pyramidal cone and explicit pairs only): pyramidal
    cone, every geom contype/conaffinity 0 (no contacts, the model flies), and every fixed tendon given a spring, damper and
    spring length range (spring, damper, lower, upper; the shipped tendons have none, so their passive port would go
    untested). Written next to mjcf as airborne.xml; returns the path."""
    import xml.etree.ElementTree as ET
    tree = ET.parse(mjcf)
    root = tree.getroot()
    opt = root.find('option') if root.find('option') is not None else ET.SubElement(root, 'option')
    opt.set('cone', 'pyramidal')
    for g in root.iter('geom'):
        g.set('contype', '0'); g.set('conaffinity', '0')
    for t in root.iter('fixed'):
        t.set('stiffness', str(spring[0])); t.set('damping', str(spring[1])); t.set('springlength', f'{spring[2]} {spring[3]}')
    out = Path(mjcf).with_name('airborne.xml')
    tree.write(out)
    return str(out)


def onfaces(q, v, side):
    """Standing states moved onto the cube's 6 faces in turn (state n on face n % 6): the whole fly rotated so the
    floor's up becomes the face's inward normal and placed at the face; free-joint linear velocity rotated with it,
    angular velocity (body frame) unchanged. Gravity stays -z, so on walls and ceiling the fly hangs by contacts."""
    h = side / 2
    ups = [(0, 0, 1), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, -1)]
    centres = [(0, 0, 0), (-h, 0, h), (h, 0, h), (0, -h, h), (0, h, h), (0, 0, side)]
    q, v = q.copy(), v.copy()
    for n in range(len(q)):
        up, c = np.array(ups[n % 6], float), np.array(centres[n % 6], float)
        axis, angle = np.cross([0, 0, 1], up), np.arccos(np.clip(up[2], -1, 1))
        axis = axis / np.linalg.norm(axis) if np.linalg.norm(axis) > 1e-9 else np.array([1., 0, 0])
        r = np.r_[np.cos(angle / 2), axis * np.sin(angle / 2)]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, r)
        R = R.reshape(3, 3)
        q[n, :3] = c + R @ q[n, :3]
        res = np.zeros(4)
        mujoco.mju_mulQuat(res, r, q[n, 3:7])
        q[n, 3:7] = res
        v[n, :3] = R @ v[n, :3]
    return q, v


def load(mjcf):
    m = mjcf if isinstance(mjcf, mujoco.MjModel) else mujoco.MjModel.from_xml_path(str(mjcf))
    if m.nmocap or m.dof_simplenum.any():
        raise SystemExit('FAIL export: mocap bodies and simple dofs are not ported')
    if (m.nflex or m.opt.integrator or m.body_gravcomp.any() or m.actuator_damping.any()
            or m.dof_dampingpoly.any() or m.jnt_stiffnesspoly.any() or m.opt.disableflags):
        raise SystemExit('FAIL export: only Euler, linear joint springs and dof dampers are ported (no flex, gravcomp, '
                         'actuator damping, polynomial springs or dampers, disable flags)')
    if (m.wrap_type != mujoco.mjtWrap.mjWRAP_JOINT).any() or m.tendon_limited.any() or m.tendon_frictionloss.any() \
            or m.tendon_stiffnesspoly.any() or m.tendon_dampingpoly.any() or m.tendon_actfrclimited.any():
        raise SystemExit('FAIL export: only fixed tendons (joint wraps), unlimited, no friction loss, linear spring and damper are ported')
    return m


def fluid(m):
    """MuJoCo mj_fluid block (engine_passive.c:842), appended last: ints per ellipsoid row (body, geom) then the inertia-box
    bodies (every body with mass that has no ellipsoid geom); floats opt.density, opt.viscosity, opt.wind, then per ellipsoid
    row the geom's semi-axes (mju_geomSemiAxes) and its 12 geom_fluid coefficients. Rows are geoms with geom_fluid[0] > 0;
    a body with any such geom takes the ellipsoid model only. Empty for a model without density and viscosity."""
    if not (m.opt.density or m.opt.viscosity):
        return None
    G = mujoco.mjtGeom
    rows = np.nonzero(m.geom_fluid[:, 0] > 0)[0]
    axes = []
    for g in rows:
        s, t = m.geom_size[g], m.geom_type[g]
        axes.append([s[0]] * 3 if t == G.mjGEOM_SPHERE else [s[0], s[0], s[1] + s[0]] if t == G.mjGEOM_CAPSULE
                    else [s[0], s[0], s[1]] if t == G.mjGEOM_CYLINDER else list(s))
    ell = {m.geom_bodyid[g] for g in rows}
    box = [b for b in range(m.nbody) if m.body_mass[b] >= 1e-15 and b not in ell]
    ints = np.array([v for g in rows for v in (m.geom_bodyid[g], g)] + box)
    floats = np.concatenate([[m.opt.density, m.opt.viscosity], m.opt.wind] + [np.r_[a, m.geom_fluid[g]] for a, g in zip(axes, rows)])
    return ints, floats, {'NFLROW': len(rows), 'NFLBOX': len(box)}


def tendons(m):
    """Fixed tendons (mj_tendon: length = sum coef q, J = coef, constant) plus mj_passive's tendon spring and damper
    (engine_passive.c:775): ints per tendon wrap start and count, then each wrap's dof address, then each wrap's qpos
    address; floats each wrap's coef, then per tendon stiffness, damping, lengthspring 2. Joints are hinge or slide only.
    New wrap kinds (spatial tendons) append their own arrays after these."""
    if not m.ntendon:
        return None
    if (m.jnt_type[m.wrap_objid] > 3).any() or (m.jnt_type[m.wrap_objid] < 2).any():
        raise SystemExit('FAIL export: tendon wraps only on hinge and slide joints')
    ints = np.concatenate([m.tendon_adr, m.tendon_num, m.jnt_dofadr[m.wrap_objid], m.jnt_qposadr[m.wrap_objid]])
    floats = np.concatenate([m.wrap_prm, m.tendon_stiffness, m.tendon_damping, m.tendon_lengthspring.ravel()])
    return ints, floats, {'NTEN': m.ntendon, 'NWRAP': m.nwrap}


def pairs(m, regex):
    """Exported contact pairs: geom2 named by regex, of a ported kind (plane or heightfield vs mesh, sphere vs sphere).
    Returns (pair ids, {unported kind: count}); the caller must say what it drops."""
    import re
    from collections import Counter
    G = mujoco.mjtGeom
    ported = {(G.mjGEOM_PLANE, G.mjGEOM_MESH), (G.mjGEOM_HFIELD, G.mjGEOM_MESH), (G.mjGEOM_SPHERE, G.mjGEOM_SPHERE)}
    kind = lambda i: (G(m.geom_type[m.pair_geom1[i]]), G(m.geom_type[m.pair_geom2[i]]))
    sel = [i for i in range(m.npair) if re.search(regex, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, m.pair_geom2[i]))]
    return [i for i in sel if kind(i) in ported], Counter('-'.join(g.name[7:].lower() for g in kind(i)) for i in sel if kind(i) not in ported)


def hpattern(m, couple):
    """Sparsity of H = M + J'DJ for mj_factorI: rows of M plus every pair's coupled dofs, filled by backward
    elimination. Row k = k's chain in the elimination tree (parent = largest column left after elimination), a
    superset of the fill, so each row's pattern is a prefix of the rows below it, as factorI requires."""
    rows = [set() for _ in range(m.nv)]
    for v in range(m.nv):
        p = m.dof_parentid[v]
        while p >= 0:
            rows[v].add(p)
            p = m.dof_parentid[p]
    for u in couple:
        for a in u:
            rows[a] |= {c for c in u if c < a}
    par = [-1] * m.nv
    for k in range(m.nv - 1, -1, -1):
        if rows[k]:
            par[k] = max(rows[k])
            rows[par[k]] |= rows[k] - {par[k]}
    hrow = []
    for k in range(m.nv):
        c, p = [k], par[k]
        while p >= 0:
            c, p = [p] + c, par[p]
        hrow.append(c)
    return hrow


def feet(m, regex):
    """Contact pairs (pairs()): ints, floats and -D sizes for smooth.mojo STAGE 6.

    ints   per pair: geom1, geom2, geom2's body, hull vertex count (0: sphere-sphere), hull vertex offset,
           edge-address offset, chain length, chain offset, contact slot offset, slot count (3 mesh, 1 sphere),
           Jacobian offset (4 rows x slots x chain) (11); then every hull vertex's first edge (absolute, into the
           edge list), the edge lists (hull-local vertex ids, -1 ends a vertex), the dof chains (H's row pattern of the
           pair's deepest dof, ascending), their Jacobian signs (+1 geom2's body only, -1 geom1's only, 0 both or
           neither), H's pattern (rowadr, rownnz, colind as hpattern()), M's entry -> H's entry, opt.iterations,
           opt.ls_iterations, opt.noslip_iterations
    floats per pair: margin, gap, friction 0-1, solref 2, solimp 5, invweight0 translation (both bodies), rbound of
           geom2 and geom1 (14); then hull vertices in the geom frame; then opt.impratio, opt.tolerance, opt.ls_tolerance, opt.noslip_tolerance, stat.meaninertia
    """
    if m.opt.cone != mujoco.mjtCone.mjCONE_PYRAMIDAL or (m.geom_contype & m.geom_conaffinity).any():
        raise SystemExit('FAIL export: contacts need the pyramidal cone and explicit pairs only')
    sel, drop = pairs(m, regex)
    if drop:
        print(f'export: {sum(drop.values())} pairs of unported kinds NOT exported: {dict(drop)}')
    if any(m.pair_dim[i] != 3 or m.pair_solimp[i][4] != int(m.pair_solimp[i][4]) for i in sel):
        raise SystemExit('FAIL export: only condim 3 contacts with a whole solimp power are ported')

    def anc(b):
        c = set()
        while b > 0:
            c |= set(range(m.body_dofadr[b], m.body_dofadr[b] + m.body_dofnum[b]))
            b = m.body_parentid[b]
        return c
    two = [(anc(m.geom_bodyid[m.pair_geom1[i]]), anc(m.geom_bodyid[m.pair_geom2[i]])) for i in sel]
    if any(m.body_rootid[m.geom_bodyid[m.pair_geom1[i]]] not in (0, m.body_rootid[m.geom_bodyid[m.pair_geom2[i]]]) for i in sel):
        raise SystemExit('FAIL export: a pair across two animals (Jacobian about one root only)')
    hrow = hpattern(m, [a ^ b for a, b in two])
    head, first, edges, chains, signs, fl, verts, hulls = [], [], [], [], [], [], [], {}
    nslot = nje = 0
    for i, (a1, a2) in zip(sel, two):
        g1, g2 = m.pair_geom1[i], m.pair_geom2[i]
        b = m.geom_bodyid[g2]
        chain = hrow[max(a1 ^ a2)]
        nv = 0
        if m.geom_type[g2] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = m.geom_dataid[g2]
            ga = m.mesh_graphadr[mesh]
            nv = m.mesh_graph[ga]
            edgeadr = m.mesh_graph[ga + 2:ga + 2 + nv]
            gid = m.mesh_graph[ga + 2 + nv:ga + 2 + 2 * nv]
            local = m.mesh_graph[ga + 2 + 2 * nv:]
            local = local[:edgeadr.max() + list(local[edgeadr.max():]).index(-1) + 1]  # through the last vertex's -1
            if mesh not in hulls:  # one copy of each hull however many planes it meets (the cube: 6 pairs per part)
                hulls[mesh] = (len(verts) // 3, len(first))
                first += list(len(edges) + edgeadr)
                edges += list(local)
                verts += list(m.mesh_vert[m.mesh_vertadr[mesh] + gid].ravel())
        ns = 3 if nv else 1
        head += [g1, g2, b, nv, *(hulls[m.geom_dataid[g2]] if nv else (0, 0)), len(chain), len(chains), nslot, ns, nje]
        nslot, nje = nslot + ns, nje + 4 * ns * len(chain)
        chains += chain
        signs += [(c in a2) - (c in a1) for c in chain]
        fl += [m.pair_margin[i], m.pair_gap[i], *m.pair_friction[i][:2], *m.pair_solref[i], *m.pair_solimp[i],
               m.body_invweight0[m.geom_bodyid[g1], 0] + m.body_invweight0[b, 0], m.geom_rbound[g2], m.geom_rbound[g1]]
    hnnz = [len(r) for r in hrow]
    hadr = list(np.cumsum([0] + hnnz[:-1]))
    hcol = [c for r in hrow for c in r]
    mtoh = [hadr[v] + hrow[v].index(m.M_colind[a]) for v in range(m.nv) for a in range(m.M_rowadr[v], m.M_rowadr[v] + m.M_rownnz[v])]
    ints = np.array(head + first + edges + chains + signs + hadr + hnnz + hcol + mtoh + [m.opt.iterations, m.opt.ls_iterations, m.opt.noslip_iterations])
    floats = fl + verts + [m.opt.impratio, m.opt.tolerance, m.opt.ls_tolerance, m.opt.noslip_tolerance, m.stat.meaninertia]
    sizes = {'NCP': len(sel), 'NHV': len(verts) // 3, 'NHE': len(edges), 'NCH': len(chains), 'NH': len(hcol), 'NSLOT': nslot, 'NJE': nje}
    hf = sorted({m.pair_geom1[i] for i in sel if m.geom_type[m.pair_geom1[i]] == mujoco.mjtGeom.mjGEOM_HFIELD})
    if hf:  # terrain: the heightfield is MuJoCo's sample; the engine evaluates world.py's function from its 8 params
        t = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_NUMERIC, 'terrain')
        if len(hf) > 1 or t < 0 or m.numeric_size[t] != 16:
            raise SystemExit('FAIL export: one heightfield ground, with its 16 world params in <numeric name="terrain">')
        floats += list(m.numeric_data[m.numeric_adr[t]:m.numeric_adr[t] + 16])
        sizes |= {'HFGEOM': hf[0]}
    floats = np.array(floats)
    return ints, floats, sizes


def actuators(m):
    """S3 tables: ints per actuator (transmission type, dof or body, ctrllimited, forcelimited, qpos address or -1),
    floats per actuator
    (gain, bias 0-2, ctrlrange 2, forcerange 2, gear)."""
    T = mujoco.mjtTrn
    for i in range(m.nu):
        joint = m.actuator_trntype[i] == T.mjTRN_JOINT and m.jnt_type[m.actuator_trnid[i, 0]] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
        if (not (joint or m.actuator_trntype[i] in (T.mjTRN_BODY, T.mjTRN_TENDON)) or m.actuator_gaintype[i] != mujoco.mjtGain.mjGAIN_FIXED
                or m.actuator_biastype[i] not in (mujoco.mjtBias.mjBIAS_NONE, mujoco.mjtBias.mjBIAS_AFFINE)
                or m.actuator_dyntype[i] != mujoco.mjtDyn.mjDYN_NONE or m.actuator_actearly[i] or m.actuator_delay[i]):
            raise SystemExit(f'FAIL export: actuator {i} is not a ported kind (hinge/slide joint, adhesion body or fixed tendon, fixed gain, no dynamics)')
    ints = np.array([[m.actuator_trntype[i], m.jnt_dofadr[m.actuator_trnid[i, 0]] if m.actuator_trntype[i] == T.mjTRN_JOINT else m.actuator_trnid[i, 0],  # tendon: its id
                      m.actuator_ctrllimited[i], m.actuator_forcelimited[i],
                      m.jnt_qposadr[m.actuator_trnid[i, 0]] if m.actuator_trntype[i] == T.mjTRN_JOINT else -1] for i in range(m.nu)])
    bias = np.where((m.actuator_biastype == mujoco.mjtBias.mjBIAS_AFFINE)[:, None], m.actuator_biasprm[:, :3], 0)
    floats = np.concatenate([m.actuator_gainprm[:, :1], bias, m.actuator_ctrlrange, m.actuator_forcerange, m.actuator_gear[:, :1]], axis=1)
    return ints.ravel(), floats.ravel()


def export(mjcf, out, regex=None, act=False):
    m, out = load(mjcf), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ints = [getattr(m, f).ravel() for f in INTS]
    floats = [np.ravel(getattr(m.opt, f[4:]) if f.startswith('opt.') else getattr(m, f)) for f in FLOATS]
    sizes = {'NBODY': m.nbody, 'NJNT': m.njnt, 'NQ': m.nq, 'NGEOM': m.ngeom, 'NV': m.nv, 'NC': m.nC, 'NU': m.nu}
    if regex:
        ci, cf, cs = feet(m, regex)
        ints, floats, sizes = ints + [ci], floats + [cf], sizes | cs
    if act:
        ai, af = actuators(m)
        ints, floats = ints + [ai], floats + [af]
    for block in (fluid(m), tendons(m)):  # last, so every earlier block keeps its offsets and a model without them its bytes
        if block:
            ints, floats, sizes = ints + [block[0]], floats + [block[1]], sizes | block[2]
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


def check(mjcf, out, stage, count=64, k=0, oracle=None):
    """oracle: MuJoCo runs this MJCF instead of mjcf (same bodies and pair order), e.g. terrain_plane.xml for the
    engine's heightfield terrain.xml; standing states come from it too."""
    feet = stage in ('contact', 'actuated')
    m, out = export(mjcf, out, FEET if feet else None, stage == 'actuated'), Path(out)
    number, fn, fields = STAGES[stage]
    q, v = standing(load(oracle) if oracle else m, count) if feet else (states(m, count), velocities(m, count, scale=VSCALE))
    if os.environ.get('STATES'):  # saved states instead (e.g. grooming poses: legs on the head, .work/.../groom_states.npz)
        z = np.load(os.environ['STATES'])
        q, v = z['qpos'][:count], z['qvel'][:count]
    if feet and Path(mjcf).name.startswith('cube'):  # every face of the cube carries states
        q, v = onfaces(q, v, float(Path(mjcf).stem[4:]))
    u = np.zeros((count, m.nu))
    if stage == 'actuated':  # servos within 0.3 rad of the current joint angle, adhesion anywhere in [0, 1]
        rng = np.random.default_rng(2)
        for i in range(m.nu):
            if m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT:
                u[:, i] = q[:, m.jnt_qposadr[m.actuator_trnid[i, 0]]] + rng.uniform(-0.3, 0.3, count)
            else:
                u[:, i] = rng.uniform(0, 1, count)
    u.astype(np.float32).T.copy().tofile(out / 'ctrl.bin')  # (nu, flies); zeros below the actuated stage
    if feet:  # only the exported pairs collide; actuators off below S3
        s = load(oracle or mjcf)
        if stage == 'contact':
            s.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ACTUATION
        on = set(pairs(s, FEET)[0])
        for i in range(s.npair):
            if i not in on:
                s.pair_margin[i] = -1e10
        want = reference(s, fn, tuple(f for f in fields if f not in ('qacc_euler', 'qpos_next')) + ('qpos', 'qvel'), q, v, u)
        want['qacc_euler'] = (want.pop('qvel') - v) / m.opt.timestep
        want['qpos_next'] = want.pop('qpos')
        d = mujoco.MjData(s)  # ncon: contacts that carry constraint rows (MuJoCo also lists excluded ones)
        ambiguous = np.zeros(count, int)  # pairs whose deepest vertex sits within float32 resolution of the margin
        switch = np.zeros(count)  # per state: the constraint row nearest its active-set switch, |J qacc - aref| / max |aref|
        live = [i for i in range(s.npair) if s.pair_margin[i] > -1 and s.geom_type[s.pair_geom1[i]] == mujoco.mjtGeom.mjGEOM_PLANE]
        noslip, s.opt.noslip_iterations = s.opt.noslip_iterations, 0  # switch: Newton's qacc (noslip moves it off the switch)
        for n in range(count):
            d.qpos[:], d.qvel[:], d.ctrl[:] = q[n], v[n], u[n]
            mujoco.mj_forward(s, d)
            want['ncon'][n] = sum(c.efc_address >= 0 for c in d.contact[:d.ncon])
            jar = np.zeros(d.nefc)
            mujoco.mj_mulJacVec(s, d, jar, d.qacc)
            switch[n] = (np.abs(jar - d.efc_aref).min() / max(np.abs(d.efc_aref).max(), 1e-30)) if d.nefc else 1.0
            for i in live:
                g1, g2 = s.pair_geom1[i], s.pair_geom2[i]
                a, c = s.mesh_vertadr[s.geom_dataid[g2]], s.mesh_vertnum[s.geom_dataid[g2]]
                depth = ((s.mesh_vert[a:a + c] @ d.geom_xmat[g2].reshape(3, 3).T + d.geom_xpos[g2] - d.geom_xpos[g1]) @ d.geom_xmat[g1].reshape(3, 3)[:, 2]).min()
                ambiguous[n] += abs(depth - s.pair_margin[i]) < 4 * np.spacing(np.float32(np.abs(d.geom_xpos[g2]).max()))
        s.opt.noslip_iterations = noslip
        # float32 floor: MuJoCo on the float32-rounded state (the engine's actual input) vs MuJoCo on the float64 state.
        # A contact force at solref 2e-4 moves 2.4e-5 of its scale from input rounding alone, so a field's bar is
        # max(REL, FLOOR_X x its floor); every field still has to beat 1e-5 unless float32 inputs cannot.
        r32 = lambda x: x.astype(np.float32).astype(np.float64)
        alt = reference(s, fn, tuple(f for f in fields if f not in ('qacc_euler', 'qpos_next', 'ncon')) + ('qpos', 'qvel'), r32(q), r32(v), r32(u))
        alt['qacc_euler'] = (alt.pop('qvel') - r32(v)) / m.opt.timestep  # same derivation as the reference's
        alt['qpos_next'] = alt.pop('qpos')
        floor = {f: float(np.abs(alt[f] - want[f]).max() / max(np.abs(want[f]).max(), 1e-30)) for f in alt}
        if k:  # K-step loop: the state after k chained mj_step calls, ctrl held; its own float32 floor
            chain = ','.join(['mj_step'] * k)
            want = reference(s, chain, ('qpos', 'qvel'), q, v, u)
            alt = reference(s, chain, ('qpos', 'qvel'), r32(q), r32(v), r32(u))
            floor = {f: float(np.abs(alt[f] - want[f]).max() / np.abs(want[f]).max()) for f in alt}
            fields = ('qpos', 'qvel')
        print('float32 input floor: ' + ', '.join(f'{f} {e:.2e}' for f, e in floor.items()))
    elif stage == 'step':
        s = load(mjcf)
        s.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT | mujoco.mjtDisableBit.mjDSBL_CONSTRAINT | mujoco.mjtDisableBit.mjDSBL_ACTUATION
        want = reference(s, fn, fields[:-2] + ('qpos', 'qvel'), q, v)
        want['qacc_euler'] = (want.pop('qvel') - v) / m.opt.timestep
        want['qpos_next'] = want.pop('qpos')
    else:
        want = reference(m, fn, fields, q, v)
    if oracle and 'geom_xpos' in want:  # world geoms are constants: take the engine model's (the heightfield sits elsewhere than its twin plane)
        w = np.flatnonzero(m.geom_bodyid == 0)
        want['geom_xpos'].reshape(count, -1, 3)[:, w] = m.geom_pos[w]
    want = {f: v[:, 1:] if f in WORLD_FIRST else v for f, v in want.items()}  # the engine skips body 0, the world
    q.astype(np.float32).T.copy().tofile(out / 'qpos.bin')  # (nq, flies): flies fastest, as the engine reads it
    v.astype(np.float32).T.copy().tofile(out / 'qvel.bin')  # (nv, flies)
    binary = out / f'{stage}_{count}'
    flags = (out / 'sizes.txt').read_text().split()
    subprocess.run([str(Path.home() / '.local/bin/mojo'), 'build', *flags, f'-DFLIES={count}', f'-DSTAGE={number}', f'-DK={k}', f"-DWAVE={os.environ.get('WAVE', 0)}", str(ENGINE / 'smooth.mojo'), '-o', str(binary)], check=True)
    subprocess.run([str(binary), str(out) + '/'], check=True)
    rows = np.fromfile(out / ('state.bin' if k else 'smooth.bin'), np.float32).reshape(-1, count).T
    sizes = [want[f][0].size for f in fields]
    if rows.shape[1] != sum(sizes):
        raise SystemExit(f'FAIL {stage}: engine wrote {rows.shape[1]} values per fly, MuJoCo {sum(sizes)}')
    got = np.split(rows, np.cumsum(sizes)[:-1], axis=1)
    floor = locals().get('floor', {})
    bad = []
    if 'ambiguous' in locals() and 'ncon' in fields:  # a contact at the margin to within float32 resolution may go either way, up to 3 rows
        i = fields.index('ncon')
        off = np.abs(got[i][:, 0] - want['ncon'].reshape(count)) > 3 * ambiguous
        print(f'{"PASS" if not off.any() else "FAIL"} ncon         {int(off.sum())} states off beyond float32-ambiguous pairs '
              f'({int(ambiguous.sum())} ambiguous pairs in {int((ambiguous > 0).sum())} states)')
        bad += ['ncon'] if off.any() else []
        got, fields = got[:i] + got[i + 1:], fields[:i] + fields[i + 1:]
    if 'switch' in locals() and 'qfrc_constraint' in fields:  # a row within the qacc bar of its active-set switch
        # may land on either side in float32: one row's force then redistributes over the stance (fly ramp state 37:
        # |jar| 1.8e-5 of scale, force error 2.4e-4 along that row's direction). Those states skip qfrc_constraint only.
        i = fields.index('qfrc_constraint')
        w = want['qfrc_constraint'].reshape(count, -1)
        over = np.abs(got[i] - w).max(1) / max(np.abs(w).max(), 1e-30) > max(REL, FLOOR_X * floor.get('qfrc_constraint', 0))
        flip = (switch < max(REL, FLOOR_X * floor.get('qacc', 0))) & over  # only states the bar would fail count
        print(f'{"PASS" if flip.mean() <= 0.1 else "FAIL"} active-set ambiguous states over the bar {int(flip.sum())}/{count} '
              f'(qfrc_constraint not compared there; at most 10 %)')
        bad += ['active-set'] if flip.mean() > 0.1 else []
        compare({'qfrc_constraint (ambiguous states)': (want['qfrc_constraint'].reshape(count, -1)[flip], got[i][flip])}, 1.0) if flip.any() else None
        want['qfrc_constraint'] = want['qfrc_constraint'].reshape(count, -1)[~flip]
        got[i] = got[i][~flip]
    bad += [b for f, g in zip(fields, got) for b in compare({f: (want[f].reshape(len(g), -1), g)}, max(REL, FLOOR_X * floor.get(f, 0)))]
    if bad:
        print(f'FAIL {len(bad)}/{len(fields)} {stage} fields: {" ".join(bad)}')
        sys.exit(1)
    lifted = ', '.join(f'{f} {FLOOR_X * e:.1e}' for f, e in floor.items() if FLOOR_X * e > REL)
    print(f'PASS {stage}' + (f' K={k}' if k else '') + f': {len(fields)} fields x {count} states within {REL:g}' + (f' (float32-floor bars: {lifted})' if lifted else ''))


if __name__ == '__main__':
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == 'meshes':
        sys.exit(0 if meshes(*args) <= 1e-4 else 'FAIL meshes: decode error above 1e-4 of part size')
    import os
    export(*args) if cmd == 'export' else check(args[0], args[1], args[2], *map(int, args[3:]), oracle=os.environ.get('ORACLE'))
