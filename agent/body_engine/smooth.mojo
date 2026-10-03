"""Smooth dynamics, ported from MuJoCo 3.9.0 src/engine/engine_core_smooth.c (Apache-2.0), batched:
one GPU thread per animal. STAGE 1: mj_kinematics1 + the inertia and geom part of mj_kinematics2.
STAGE 2: + mj_comPos (subtree_com, cinert, cdof) + mj_crb (sparse M in MuJoCo's own M_rowadr layout).
STAGE 3: + mj_factorM (mj_factorI: sparse L'DL of M in place, qLD, qLDiagInv).
STAGE 4: + mj_comVel (cvel, cdof_dot), mj_passive (joint springs, dof dampers), mj_rne without qacc (qfrc_bias).
STAGE 5: + mj_Euler, contact-free: implicit dof damping (qH = M + h diag(damping)), qacc, mj_integratePos.
STAGE 6: + plane-mesh foot contacts, pyramidal cone, mj_solPrimal (Newton); STAGE 7: + actuators (joint servos,
adhesion bodies) from ctrl.bin (NU x FLIES).

Model arrays come from agent/body_engine/model.py export (mi.bin int32, mf.bin float32, field order
INTS / FLOATS there); sizes are -D constants, so any MJCF body (fly, spider) builds its own binary.
Orientation is kept as quaternions only (MuJoCo also builds 3x3 xmat; rotating by the quaternion is
the same map and stores 4 floats, not 9). Mocap bodies and the sleep filter are not ported.

Layout: every per-fly array is [row][fly], flies fastest, so a warp reads 32 neighbouring flies
in one transaction. Model arrays are shared by all flies and stay in cache.

usage: smooth DIR/      reads DIR/mi.bin, mf.bin, qpos.bin (NQ x FLIES), qvel.bin (NV x FLIES); writes DIR/smooth.bin, rows:
       xpos, xquat, xipos (bodies 1..), geom_xpos, xanchor, xaxis; STAGE 2 adds subtree_com,
       cinert (bodies 1..), cdof, crb (bodies 1..), M; STAGE 3 adds qLD, qLDiagInv; STAGE 4 adds cvel (bodies
       1..), cdof_dot, qfrc_passive, qfrc_bias; STAGE 5 adds qH, qHDiagInv, qacc_euler, qpos_next; prints us per launch.
       -DK=k: k steps per launch, state kept on the card; the state after the first launch goes to DIR/state.bin
       (qpos rows, then qvel rows), the state after 10 steps to DIR/state10.bin when K divides 10
"""
from std.ffi import external_call
from std.math import cos, sin, sqrt
from std.sys import argv, get_defined_int
from std.time import perf_counter_ns
from std.gpu import block_idx, block_dim, thread_idx
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from world import NP, PW, height_grad

comptime NB = get_defined_int["NBODY"]()
comptime NJ = get_defined_int["NJNT"]()
comptime NQ = get_defined_int["NQ"]()
comptime NG = get_defined_int["NGEOM"]()
comptime NV = get_defined_int["NV"]()
comptime NC = get_defined_int["NC"]()
comptime F = get_defined_int["FLIES"]()
comptime STAGE = get_defined_int["STAGE", 1]()
comptime STOP = get_defined_int["STOP", 99]()
comptime K = get_defined_int["K", 0]()  # K > 0 (STAGE >= 5): K steps per launch, qpos/qvel written back in place
comptime STEPTOL = Float32(get_defined_int["STEPTOL_E7", 10]()) * 1e-7  # Newton stops when a step moves qacc less than this
comptime LSFLOOR = Float32(get_defined_int["LSFLOOR_E7", 10]()) * 1e-7  # line-search slope floor, relative to alpha = 0  # phase timing (phases.sh): return after phase STOP
# mi.bin: body_parentid, body_jntadr, body_jntnum, jnt_type, jnt_qposadr, geom_bodyid
comptime PARENT = 0
comptime JNTADR = NB
comptime JNTNUM = 2 * NB
comptime JTYPE = 3 * NB
comptime QADR = 3 * NB + NJ
comptime GBODY = 3 * NB + 2 * NJ
# STAGE 2: body_rootid, jnt_dofadr, dof_parentid, dof_bodyid, M_rowadr, M_rownnz
comptime ROOT = 3 * NB + 2 * NJ + NG
comptime DOFADR = ROOT + NB
comptime DPARENT = DOFADR + NJ
comptime DBODY = DPARENT + NV
comptime MADR = DBODY + NV
comptime MNNZ = MADR + NV
comptime MCOL = MNNZ + NV  # STAGE 3: M_colind
comptime BDOFNUM = MCOL + NC  # STAGE 4: body_dofnum, body_dofadr
comptime BDOFADR = BDOFNUM + NB
comptime NI0 = BDOFADR + NB
# STAGE 6 contact block (model.py feet): per pair 8 ints, hull first edges, edge lists, dof chains, 2 solver ints
comptime NCP = get_defined_int["NCP", 0]()
comptime NHV = get_defined_int["NHV", 0]()
comptime NHE = get_defined_int["NHE", 0]()
comptime NCH = get_defined_int["NCH", 0]()
comptime CPH = NI0
comptime CFIRST = CPH + 8 * NCP
comptime CEDGE = CFIRST + NHV
comptime CCHAIN = CEDGE + NHE
comptime CITER = CCHAIN + NCH
comptime NU = get_defined_int["NU", 0]()
comptime CACT = CITER + 2  # STAGE 7 (model.py actuators): per actuator trntype, dof or body, ctrllimited, forcelimited, qpos address
comptime NI = CACT + 5 * NU if STAGE >= 7 else (CITER + 2 if STAGE >= 6 else NI0)
# mf.bin: body_pos, body_quat, jnt_pos, jnt_axis, qpos0, body_ipos, body_iquat, geom_pos, geom_quat
comptime BPOS = 0
comptime BQUAT = 3 * NB
comptime JPOS = 7 * NB
comptime JAXIS = 7 * NB + 3 * NJ
comptime QPOS0 = 7 * NB + 6 * NJ
comptime IPOS = 7 * NB + 6 * NJ + NQ
comptime GPOS = 14 * NB + 6 * NJ + NQ
comptime IQUAT = IPOS + 3 * NB
comptime GQUAT = GPOS + 3 * NG
# STAGE 2: body_mass, body_subtreemass, body_inertia, dof_armature
comptime MASS = 14 * NB + 6 * NJ + NQ + 7 * NG
comptime SUBMASS = MASS + NB
comptime INERTIA = SUBMASS + NB
comptime ARMATURE = INERTIA + 3 * NB
comptime DAMP = ARMATURE + NV  # STAGE 4: dof_damping, jnt_stiffness, qpos_spring, opt.gravity, opt.timestep
comptime STIFF = DAMP + NV
comptime QSPRING = STIFF + NJ
comptime GRAV = QSPRING + NQ
comptime TSTEP = GRAV + 3
comptime NFL0 = TSTEP + 1
comptime CPF = NFL0  # STAGE 6: per pair margin, gap, friction 2, solref 2, solimp 5, invweight, rbound; hull verts; 3 opts
comptime CVERT = CPF + 13 * NCP
comptime COPT = CVERT + 3 * NHV
comptime HFG = get_defined_int["HFGEOM", -1]()  # heightfield ground geom (model.py feet), else none
comptime HFO = COPT + 3  # world.mojo params (NP)
comptime CACF = HFO + (NP if HFG >= 0 else 0)  # STAGE 7: per actuator gain, bias 0-2, ctrlrange 2, forcerange 2, gear
comptime NFL = CACF + 9 * NU if STAGE >= 7 else (CACF if STAGE >= 6 else NFL0)
# smooth.bin rows (per animal): xpos, xquat, xipos of bodies 1.., geom_xpos, xanchor, xaxis
comptime OXQ = 3 * (NB - 1)
comptime OXI = 7 * (NB - 1)
comptime OG = 10 * (NB - 1)
comptime OA = 10 * (NB - 1) + 3 * NG
comptime OX = 10 * (NB - 1) + 3 * NG + 3 * NJ
comptime OSC = 10 * (NB - 1) + 3 * NG + 6 * NJ  # STAGE 2: subtree_com, cinert, cdof, crb, M
comptime OCI = OSC + 3 * NB
comptime OCD = OCI + 10 * (NB - 1)
comptime OCRB = OCD + 6 * NV
comptime OM = OCRB + 10 * (NB - 1)
comptime OLD = OM + NC  # STAGE 3: qLD, qLDiagInv
comptime OLDI = OLD + NC
comptime OCV = OLDI + NV  # STAGE 4: cvel (bodies 1..), cdof_dot, qfrc_passive, qfrc_bias
comptime OCDD = OCV + 6 * (NB - 1)
comptime OPAS = OCDD + 6 * NV
comptime OBIAS = OPAS + NV
comptime OQH = OBIAS + NV  # STAGE 5: qH, qHDiagInv, qacc_euler, qpos_next
comptime OQHI = OQH + NC
comptime OQACC = OQHI + NV
comptime OQPOS = OQACC + NV
comptime ONCON = OQPOS + NQ  # STAGE 6: ncon, qacc (constrained), qfrc_constraint
comptime OCQACC = ONCON + 1
comptime OQFC = OCQACC + NV
comptime OAF = OQFC + NV  # STAGE 7: actuator_force, qfrc_actuator
comptime OQACT = OAF + NU
comptime NOUT = OSC if STAGE == 1 else (OM + NC if STAGE == 2 else (OLDI + NV if STAGE == 3 else (OQH if STAGE == 4 else (OQPOS + NQ if STAGE == 5 else (OQFC + NV if STAGE == 6 else OQACT + NV)))))
comptime OCACC = NOUT  # scratch rows after the output: cacc, cfrc_body per body (world included)
comptime OCFRC = OCACC + 6 * NB
comptime NSLOT = 3 * NCP  # up to 3 contacts per plane-mesh pair (MuJoCo maxplanemesh)
comptime NEFC = 4 * NSLOT  # pyramidal condim 3: 4 rows per contact
comptime OVALID = OCFRC + 6 * NB  # STAGE 6 scratch: slot valid, per row D aref jar jv state, J rows, solver vectors, H
comptime OD = OVALID + NSLOT
comptime OAREF = OD + NEFC
comptime OJAR = OAREF + NEFC
comptime OJV = OJAR + NEFC
comptime OST = OJV + NEFC
comptime OJ = OST + NEFC
comptime OQSM = OJ + 12 * NCH
comptime OQACCS = OQSM + NV
comptime OMA = OQACCS + NV
comptime OGRAD = OMA + NV
comptime OSRCH = OGRAD + NV
comptime OMV = OSRCH + NV
comptime OHN = OMV + NV
comptime OHNI = OHN + NC
comptime ODIAG = OHNI + NV  # Newton iterations, line-search evaluations (printed by main)
comptime NROW = ODIAG + 2 if STAGE >= 6 else (OCFRC + 6 * NB if STAGE >= 4 else NOUT)
comptime LI = row_major[NI]()
comptime LF = row_major[NFL]()
comptime LQ = row_major[NQ * F]()
comptime LO = row_major[NROW * F]()
comptime LV = row_major[NV * F]()
comptime V3 = SIMD[DType.float32, 4]  # x, y, z, unused
comptime Q4 = SIMD[DType.float32, 4]  # w, x, y, z


@always_inline
def mul(a: Q4, b: Q4) -> Q4:
    return Q4(a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3], a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
              a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1], a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0])


@always_inline
def rot(q: Q4, v: V3) -> V3:
    """MuJoCo mju_rotVecQuat: t = w v + q x v, v + 2 q x t."""
    var t = V3(q[0] * v[0] + q[2] * v[2] - q[3] * v[1], q[0] * v[1] + q[3] * v[0] - q[1] * v[2], q[0] * v[2] + q[1] * v[1] - q[2] * v[0], 0)
    return V3(v[0] + 2 * (q[2] * t[2] - q[3] * t[1]), v[1] + 2 * (q[3] * t[0] - q[1] * t[2]), v[2] + 2 * (q[1] * t[1] - q[2] * t[0]), 0)


@always_inline
def cross(a: V3, b: V3) -> V3:
    return V3(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0], 0)


@always_inline
def unit(q: Q4) -> Q4:
    var n = sqrt((q * q).reduce_add())
    return q / n if n > 1e-15 else Q4(1, 0, 0, 0)


comptime TI = TileTensor[DType.int32, type_of(LI), MutAnyOrigin]
comptime TF = TileTensor[DType.float32, type_of(LF), MutAnyOrigin]
comptime TQ = TileTensor[DType.float32, type_of(LQ), MutAnyOrigin]
comptime TV = TileTensor[DType.float32, type_of(LV), MutAnyOrigin]
comptime LU = row_major[max(NU, 1) * F]()
comptime TU = TileTensor[DType.float32, type_of(LU), MutAnyOrigin]
comptime TO = TileTensor[DType.float32, type_of(LO), MutAnyOrigin]


@always_inline
def m3(mf: TF, k: Int) -> V3:
    return V3(mf[k], mf[k + 1], mf[k + 2], 0)


@always_inline
def m4(mf: TF, k: Int) -> Q4:
    return Q4(mf[k], mf[k + 1], mf[k + 2], mf[k + 3])


@always_inline
def put(res: TO, f: Int, row: Int, v: SIMD[DType.float32, 4], n: Int):
    for c in range(n):
        res[(row + c) * F + f] = v[c]


@always_inline
def get(res: TO, f: Int, row: Int) -> Float32:
    return res[row * F + f]


@always_inline
def col(q: Q4, k: Int) -> V3:
    """Column k of the rotation matrix of q (MuJoCo xmat[k], xmat[k+3], xmat[k+6])."""
    return rot(q, basis(k))


@always_inline
def basis(k: Int) -> V3:
    var e = V3(0)
    e[k] = 1
    return e


@always_inline
def body(res: TO, f: Int, i: Int) -> Tuple[V3, Q4]:
    """World frame of body i (0 = world), already written by this thread."""
    if i == 0:
        return (V3(0), Q4(1, 0, 0, 0))
    var r = 3 * (i - 1)
    var s = OXQ + 4 * (i - 1)
    return (V3(res[r * F + f], res[(r + 1) * F + f], res[(r + 2) * F + f], 0),
            Q4(res[s * F + f], res[(s + 1) * F + f], res[(s + 2) * F + f], res[(s + 3) * F + f]))


comptime S6 = SIMD[DType.float32, 8]  # spatial vector: rotation 0..2, translation 3..5


@always_inline
def get6(res: TO, f: Int, row: Int) -> S6:
    return S6(get(res, f, row), get(res, f, row + 1), get(res, f, row + 2), get(res, f, row + 3), get(res, f, row + 4), get(res, f, row + 5), 0, 0)


@always_inline
def put6(res: TO, f: Int, row: Int, v: S6):
    for c in range(6):
        res[(row + c) * F + f] = v[c]


@always_inline
def cross_motion(a: S6, v: S6) -> S6:
    """MuJoCo mju_crossMotion: a x v for motion vectors."""
    return S6(-a[2] * v[1] + a[1] * v[2], a[2] * v[0] - a[0] * v[2], -a[1] * v[0] + a[0] * v[1],
              -a[2] * v[4] + a[1] * v[5] - a[5] * v[1] + a[4] * v[2], a[2] * v[3] - a[0] * v[5] + a[5] * v[0] - a[3] * v[2],
              -a[1] * v[3] + a[0] * v[4] - a[4] * v[0] + a[3] * v[1], 0, 0)


@always_inline
def cross_force(a: S6, v: S6) -> S6:
    """MuJoCo mju_crossForce: a x* v for force vectors."""
    return S6(-a[2] * v[1] + a[1] * v[2] - a[5] * v[4] + a[4] * v[5], a[2] * v[0] - a[0] * v[2] + a[5] * v[3] - a[3] * v[5],
              -a[1] * v[0] + a[0] * v[1] - a[4] * v[3] + a[3] * v[4], -a[2] * v[4] + a[1] * v[5], a[2] * v[3] - a[0] * v[5],
              -a[1] * v[3] + a[0] * v[4], 0, 0)


@always_inline
def mul_inert(res: TO, f: Int, row: Int, v: S6) -> S6:
    """MuJoCo mju_mulInertVec with the 10-float com inertia stored at rows row..row+9."""
    var i = SIMD[DType.float32, 16](0)
    for k in range(10):
        i[k] = get(res, f, row + k)
    return S6(i[0] * v[0] + i[3] * v[1] + i[4] * v[2] - i[8] * v[4] + i[7] * v[5],
              i[3] * v[0] + i[1] * v[1] + i[5] * v[2] + i[8] * v[3] - i[6] * v[5],
              i[4] * v[0] + i[5] * v[1] + i[2] * v[2] - i[7] * v[3] + i[6] * v[4],
              i[8] * v[1] - i[7] * v[2] + i[9] * v[3], i[6] * v[2] - i[8] * v[0] + i[9] * v[4],
              i[7] * v[0] - i[6] * v[1] + i[9] * v[5], 0, 0)


@always_inline
def atan2p(y: Float32, x: Float32) -> Float32:
    """Atan2 for y >= 0 (amdgpu has no atan2f): 2 atan(y / (r + x)), atan reduced to |z| <= tan(pi/16), Taylor to z^9."""
    var r = sqrt(x * x + y * y)
    if r + x <= 0:
        return 3.14159265358979 if r > 0 else 0.0
    var z = y / (r + x)
    var flip = z > 1
    if flip:
        z = 1 / z
    z = z / (1 + sqrt(1 + z * z))
    z = z / (1 + sqrt(1 + z * z))
    var z2 = z * z
    var a = 4 * z * (1 + z2 * (-1.0 / 3 + z2 * (1.0 / 5 + z2 * (-1.0 / 7 + z2 / 9))))
    return 2 * (1.5707963267949 - a if flip else a)


def factorI(mi: TI, res: TO, f: Int, L: Int, LI_: Int):
    """MuJoCo mj_factorI: sparse L'DL of rows L.. in place, backward over rows k: L(i, 0..i) -= L(i, 0..i) L(k, i) / L(k, k),
    then L(k, :) /= L(k, k); 1 / L(k, k) to rows LI_.."""
    for k in range(NV - 1, -1, -1):
        var start = Int(mi[MADR + k])
        var diag = Int(mi[MNNZ + k]) - 1
        var end = start + diag
        var invD = 1 / get(res, f, L + end)
        res[(LI_ + k) * F + f] = invD
        for adr in range(end - 1, start - 1, -1):
            var i = Int(mi[MCOL + adr])
            var scl = -get(res, f, L + adr) * invD
            var ri = Int(mi[MADR + i])
            for c in range(Int(mi[MNNZ + i])):
                res[(L + ri + c) * F + f] += get(res, f, L + start + c) * scl
        for c in range(diag):
            res[(L + start + c) * F + f] = get(res, f, L + start + c) * invD


def solveLD(mi: TI, res: TO, f: Int, X: Int, L: Int, LI_: Int):
    """MuJoCo mj_solveLD, one vector: x <- (L'DL)^-1 x for the factor in rows L.., 1/D in rows LI_.."""
    for i in range(NV - 1, -1, -1):  # x <- L^-T x
        var start = Int(mi[MADR + i])
        var x_i = get(res, f, X + i)
        for adr in range(start, start + Int(mi[MNNZ + i]) - 1):
            res[(X + Int(mi[MCOL + adr])) * F + f] -= get(res, f, L + adr) * x_i
    for i in range(NV):  # x <- D^-1 x
        res[(X + i) * F + f] *= get(res, f, LI_ + i)
    for i in range(NV):  # x <- L^-1 x
        var start = Int(mi[MADR + i])
        var dot: Float32 = 0
        for adr in range(start, start + Int(mi[MNNZ + i]) - 1):
            dot += get(res, f, L + adr) * get(res, f, X + Int(mi[MCOL + adr]))
        res[(X + i) * F + f] -= dot


def mulM(mi: TI, res: TO, f: Int, Y: Int, X: Int):
    """MuJoCo mju_mulSymVecSparse with M (rows OM..): y = M x, lower triangle stored."""
    for i in range(NV):
        res[(Y + i) * F + f] = 0
    for i in range(NV):
        var start = Int(mi[MADR + i])
        var end = start + Int(mi[MNNZ + i]) - 1
        var x_i = get(res, f, X + i)
        var acc = get(res, f, OM + end) * x_i
        for adr in range(start, end):
            var j = Int(mi[MCOL + adr])
            acc += get(res, f, OM + adr) * get(res, f, X + j)
            res[(Y + j) * F + f] += get(res, f, OM + adr) * x_i
        res[(Y + i) * F + f] += acc


def velocity(mi: TI, mf: TF, qpos: TQ, qvel: TV, res: TO, f: Int):
    """MuJoCo mj_comVel, mj_passive (joint springs, dof dampers), mj_rne(flg_acc = 0) into qfrc_bias."""
    # mj_comVel: cvel = cvel_parent + cdof qvel, cdof_dot = cvel x cdof with the velocity before the joint
    for i in range(1, NB):
        var p = Int(mi[PARENT + i])
        var cvel = get6(res, f, OCV + 6 * (p - 1)) if p > 0 else S6(0)
        for j in range(Int(mi[JNTADR + i]), Int(mi[JNTADR + i]) + Int(mi[JNTNUM + i])):
            var t = Int(mi[JTYPE + j])
            var da = Int(mi[DOFADR + j])
            if t == 0:  # free: translations have cdof_dot 0, then rotations as a ball
                for k in range(3):
                    put6(res, f, OCDD + 6 * (da + k), S6(0))
                    cvel += get6(res, f, OCD + 6 * (da + k)) * qvel[(da + k) * F + f]
                da += 3
            var n = 3 if t <= 1 else 1
            for k in range(n):
                put6(res, f, OCDD + 6 * (da + k), cross_motion(cvel, get6(res, f, OCD + 6 * (da + k))))
            for k in range(n):
                cvel += get6(res, f, OCD + 6 * (da + k)) * qvel[(da + k) * F + f]
        put6(res, f, OCV + 6 * (i - 1), cvel)
    # mj_passive: qfrc_passive = qfrc_spring + qfrc_damper
    for v in range(NV):
        res[(OPAS + v) * F + f] = -mf[DAMP + v] * qvel[v * F + f]
    for j in range(NJ):
        var k = mf[STIFF + j]
        if k == 0:
            continue
        var a = Int(mi[QADR + j])
        var da = Int(mi[DOFADR + j])
        var t = Int(mi[JTYPE + j])
        if t == 0:
            for c in range(3):
                res[(OPAS + da + c) * F + f] += -k * (qpos[(a + c) * F + f] - mf[QSPRING + a + c])
            a += 3
            da += 3
        if t <= 1:  # mju_subQuat(dif, unit(q), q_spring): neg(q_spring) q to axis-angle
            var q = unit(Q4(qpos[a * F + f], qpos[(a + 1) * F + f], qpos[(a + 2) * F + f], qpos[(a + 3) * F + f]))
            var qs = m4(mf, QSPRING + a)
            var d = mul(Q4(qs[0], -qs[1], -qs[2], -qs[3]), q)
            var axis = V3(d[1], d[2], d[3], 0)
            var s = sqrt((axis * axis).reduce_add())
            axis = axis / s if s >= 1e-15 else basis(0)
            var speed = 2 * atan2p(s, d[0])
            if speed > 3.14159265358979:
                speed -= 2 * 3.14159265358979
            for c in range(3):
                res[(OPAS + da + c) * F + f] += -k * axis[c] * speed
        else:  # slide, hinge
            res[(OPAS + da) * F + f] += -k * (qpos[a * F + f] - mf[QSPRING + a])
    # mj_rne: cacc = cacc_parent + cdof_dot qvel (world: -gravity), cfrc = cinert cacc + cvel x* (cinert cvel)
    put6(res, f, OCACC, S6(0, 0, 0, -mf[GRAV], -mf[GRAV + 1], -mf[GRAV + 2], 0, 0))
    for i in range(1, NB):
        var acc = get6(res, f, OCACC + 6 * Int(mi[PARENT + i]))
        var bda = Int(mi[BDOFADR + i])
        for k in range(Int(mi[BDOFNUM + i])):
            acc += get6(res, f, OCDD + 6 * (bda + k)) * qvel[(bda + k) * F + f]
        put6(res, f, OCACC + 6 * i, acc)
        var cvel = get6(res, f, OCV + 6 * (i - 1))
        var ci = OCI + 10 * (i - 1)
        put6(res, f, OCFRC + 6 * i, mul_inert(res, f, ci, acc) + cross_force(cvel, mul_inert(res, f, ci, cvel)))
    for i in range(NB - 1, 0, -1):
        var p = Int(mi[PARENT + i])
        if p > 0:
            put6(res, f, OCFRC + 6 * p, get6(res, f, OCFRC + 6 * p) + get6(res, f, OCFRC + 6 * i))
    for v in range(NV):
        var prod = get6(res, f, OCD + 6 * v) * get6(res, f, OCFRC + 6 * Int(mi[DBODY + v]))
        res[(OBIAS + v) * F + f] = prod.reduce_add()


@always_inline
def jdot(mi: TI, res: TO, f: Int, jadr: Int, nch: Int, choff: Int, X: Int) -> Float32:
    """Sparse Jacobian row (chain order) dotted with the dof vector in rows X.."""
    var acc: Float32 = 0
    for c in range(nch):
        acc += get(res, f, jadr + c) * get(res, f, X + Int(mi[CCHAIN + choff + c]))
    return acc


@always_inline
def powi(a: Float32, b: Float32) -> Float32:
    """MuJoCo power() for a whole exponent (export refuses others): amdgpu links no powf here."""
    var r: Float32 = 1
    for _ in range(Int(b)):
        r *= a
    return r


@always_inline
def terrain(mf: TF) -> PW:
    var p = PW()
    for k in range(NP):
        p[k] = mf[HFO + k]
    return p


def contacts(mi: TI, mf: TF, qpos: TQ, qvel: TV, res: TO, f: Int):
    """MuJoCo mjc_PlaneConvex (support vertex + hull neighbours below the margin, at most 3, none within 0.3 rbound of
    the first), mj_instantiateContact (pyramidal condim 3, Jacobian over the body's dof chain), mj_diagApprox,
    mj_makeImpedance and mj_referenceConstraint, per exported plane-mesh pair. A heightfield ground (HFGEOM) is not
    mjc_ConvexHField (GJK+EPA on cell prisms): the plane is the tangent plane of world.mojo's terrain function under
    the mesh centre, exact on linear terrain, wrong where the terrain bends inside one mesh."""
    var h = mf[TSTEP]
    var impratio = mf[COPT]
    for p in range(NCP):
        var g1 = Int(mi[CPH + 8 * p]); var g2 = Int(mi[CPH + 8 * p + 1]); var b = Int(mi[CPH + 8 * p + 2])
        var nvert = Int(mi[CPH + 8 * p + 3]); var voff = Int(mi[CPH + 8 * p + 4]); var eoff = Int(mi[CPH + 8 * p + 5])
        var nch = Int(mi[CPH + 8 * p + 6]); var choff = Int(mi[CPH + 8 * p + 7])
        var pf = CPF + 13 * p
        var margin = mf[pf]; var gap = mf[pf + 1]; var f0 = mf[pf + 2]; var f1 = mf[pf + 3]
        var b1 = body(res, f, Int(mi[GBODY + g1]))
        var q1 = mul(b1[1], m4(mf, GQUAT + 4 * g1))
        var o = origin(mi, qpos, f, b)  # the plane (a world geom, world coordinates) in the animal's frame
        var pos1 = V3(get(res, f, OG + 3 * g1), get(res, f, OG + 3 * g1 + 1), get(res, f, OG + 3 * g1 + 2), 0) - o
        var b2 = body(res, f, b)
        var q2 = mul(b2[1], m4(mf, GQUAT + 4 * g2))
        var pos2 = V3(get(res, f, OG + 3 * g2), get(res, f, OG + 3 * g2 + 1), get(res, f, OG + 3 * g2 + 2), 0)
        var normal = rot(q1, basis(2))
        comptime if HFG >= 0:
            if g1 == HFG:
                var w = pos2 + o
                var hg = height_grad(w[0], w[1], terrain(mf))
                normal = V3(-hg[1], -hg[2], 1, 0) / sqrt(hg[1] * hg[1] + hg[2] * hg[2] + 1)
                pos1 = V3(pos2[0], pos2[1], hg[0] - o[2], 0)  # pos2's own x, y: no round trip through world coordinates
        if (normal * (pos2 - pos1)).reduce_add() - mf[pf + 12] > margin:  # bounding sphere clear of the plane: the
            for c in range(3):                                          # support vertex would be too (no contact)
                res[(OVALID + 3 * p + c) * F + f] = 0
            continue
        var locdir = rot(Q4(q2[0], -q2[1], -q2[2], -q2[3]), -normal)
        var best = 0
        var bdot: Float32 = -3.4e38
        for v in range(nvert):
            var vd = (locdir * m3(mf, CVERT + 3 * (voff + v))).reduce_add()
            if vd > bdot:
                bdot = vd
                best = v
        # contact frame (mju_makeFrame from the normal), shared by the pair's contacts
        var y = basis(1) if normal[1] < 0.5 and normal[1] > -0.5 else basis(2)
        y = y - normal * (normal * y).reduce_add()
        y = y / sqrt((y * y).reduce_add())
        var z = cross(normal, y)
        var com = V3(get(res, f, OSC + 3 * Int(mi[ROOT + b])), get(res, f, OSC + 3 * Int(mi[ROOT + b]) + 1), get(res, f, OSC + 3 * Int(mi[ROOT + b]) + 2), 0)
        var pnt = rot(q2, m3(mf, CVERT + 3 * (voff + best))) + pos2
        var dist = (normal * (pnt - pos1)).reduce_add()
        var first = pnt - normal * (0.5 * dist)
        var threshold = (normal * (pos2 - pos1)).reduce_add() - margin
        var count = 0
        var e = Int(mi[CFIRST + eoff + best])
        while count < 3:
            if count > 0:  # hull neighbours of the support vertex
                var loc = Int(mi[CEDGE + e])
                if loc < 0:
                    break
                e += 1
                if (locdir * m3(mf, CVERT + 3 * (voff + loc))).reduce_add() <= threshold:
                    continue
                pnt = rot(q2, m3(mf, CVERT + 3 * (voff + loc))) + pos2
                var dd = pnt - first
                if sqrt((dd * dd).reduce_add()) < 0.3 * mf[pf + 12]:
                    continue
                dist = (normal * (pnt - pos1)).reduce_add()
            elif dist > margin:
                break
            var cpos = pnt - normal * (0.5 * dist)
            var slot = 3 * p + count
            res[(OVALID + slot) * F + f] = 1
            # mj_jac over the chain: jacp = cdof_lin + cdof_ang x (pos - subtree_com[root]); rows to the contact frame
            var off = cpos - com
            for c in range(nch):
                var cd = OCD + 6 * Int(mi[CCHAIN + choff + c])
                var ang = V3(get(res, f, cd), get(res, f, cd + 1), get(res, f, cd + 2), 0)
                var jp = V3(get(res, f, cd + 3), get(res, f, cd + 4), get(res, f, cd + 5), 0) + cross(ang, off)
                var jn = (normal * jp).reduce_add(); var j1 = (y * jp).reduce_add(); var j2 = (z * jp).reduce_add()
                var jadr = OJ + 12 * choff + 4 * count * nch + c
                res[jadr * F + f] = jn + f0 * j1
                res[(jadr + nch) * F + f] = jn - f0 * j1
                res[(jadr + 2 * nch) * F + f] = jn + f1 * j2
                res[(jadr + 3 * nch) * F + f] = jn - f1 * j2
            # getsolparam, getimpedance, makeImpedance (pyramidal), referenceConstraint
            var ref0 = max(mf[pf + 4], 2 * h); var ref1 = mf[pf + 5]
            var s0 = min(Float32(0.9999), max(Float32(1e-4), mf[pf + 6])); var s1 = min(Float32(0.9999), max(Float32(1e-4), mf[pf + 7]))
            var s2 = max(Float32(0), mf[pf + 8]); var s3 = min(Float32(0.9999), max(Float32(1e-4), mf[pf + 9])); var s4 = max(Float32(1), mf[pf + 10])
            var imargin = margin - gap
            var imp: Float32
            if s0 == s1 or s2 <= 1e-15:
                imp = 0.5 * (s0 + s1)
            else:
                var x = abs((dist - imargin) / s2)
                if x >= 1 or x <= 0:
                    imp = s1 if x >= 1 else s0
                else:
                    var yy: Float32
                    if s4 == 1:
                        yy = x
                    elif x <= s3:
                        yy = powi(x, s4) / powi(s3, s4 - 1)
                    else:
                        yy = 1 - powi(1 - x, s4) / powi(1 - s3, s4 - 1)
                    imp = s0 + yy * (s1 - s0)
            var tran = mf[pf + 11]
            var R0 = max(Float32(1e-15), (1 - imp) * (tran + f0 * f0 * tran) / imp)
            var R1 = R0 / max(Float32(1e-15), impratio)
            var mu = f0 * sqrt(R1 / R0)
            var D = 1 / (2 * mu * mu * R0)
            var K = 1 / max(Float32(1e-15), s1 * s1 * ref0 * ref0 * ref1 * ref1)
            var B = 2 / max(Float32(1e-15), s1 * ref0)
            for k in range(4):
                var row = 4 * slot + k
                res[(OD + row) * F + f] = D
                var vel: Float32 = 0
                for c in range(nch):
                    vel += get(res, f, OJ + 12 * choff + (4 * count + k) * nch + c) * qvel[Int(mi[CCHAIN + choff + c]) * F + f]
                res[(OAREF + row) * F + f] = -B * vel - K * imp * (dist - imargin)
            count += 1
        for c in range(count, 3):
            res[(OVALID + 3 * p + c) * F + f] = 0


def actuation(mi: TI, mf: TF, qpos: TQ, qvel: TV, ctrl: TU, res: TO, f: Int):
    """MuJoCo mj_transmission + mj_fwdActuation for the ported kinds. Joint (hinge, slide): moment = gear on the
    joint's dof, length = qpos gear. Body (adhesion): moment = -(mean contact normal Jacobian) over the body's
    contacts (pyramid rows weighted 0.5 / (dim - 1), which sums to the normal row), length 0. Force = gain ctrl
    (ctrl clamped to ctrlrange) + bias0 + bias1 length + bias2 velocity, clamped to forcerange; qfrc = moment' force."""
    for v in range(NV):
        res[(OQACT + v) * F + f] = 0
    for u in range(NU):
        var a = CACT + 5 * u
        var af = CACF + 9 * u
        var c = ctrl[u * F + f]
        if mi[a + 2] != 0:
            c = min(max(c, mf[af + 4]), mf[af + 5])
        var gear = mf[af + 8]
        if mi[a] == 0:  # joint
            var dof = Int(mi[a + 1])
            var length = qpos[Int(mi[a + 4]) * F + f] * gear
            var vel = gear * qvel[dof * F + f]
            var force = mf[af] * c + mf[af + 1] + mf[af + 2] * length + mf[af + 3] * vel
            if mi[a + 3] != 0:
                force = min(max(force, mf[af + 6]), mf[af + 7])
            res[(OAF + u) * F + f] = force
            res[(OQACT + dof) * F + f] += gear * force
        else:  # body: adhesion over the body's contacts
            var body = Int(mi[a + 1])
            var counter = 0
            for p in range(NCP):
                if Int(mi[CPH + 8 * p + 2]) != body:
                    continue
                for sl in range(3):
                    counter += 1 if get(res, f, OVALID + 3 * p + sl) != 0 else 0
            var force = mf[af] * c + mf[af + 1]  # length and velocity of a body transmission are 0
            if mi[a + 3] != 0:
                force = min(max(force, mf[af + 6]), mf[af + 7])
            res[(OAF + u) * F + f] = force
            if counter == 0:
                continue
            var scale = -force / Float32(counter)
            for p in range(NCP):
                if Int(mi[CPH + 8 * p + 2]) != body:
                    continue
                var nch = Int(mi[CPH + 8 * p + 6]); var choff = Int(mi[CPH + 8 * p + 7])
                for sl in range(3):
                    if get(res, f, OVALID + 3 * p + sl) == 0:
                        continue
                    var jadr = OJ + 12 * choff + 4 * sl * nch
                    for k in range(nch):  # normal row = mean of the pyramid's first pair
                        var jn = 0.5 * (get(res, f, jadr + k) + get(res, f, jadr + nch + k))
                        res[(OQACT + Int(mi[CCHAIN + choff + k])) * F + f] += scale * jn


def update(mi: TI, res: TO, f: Int) -> Float32:
    """MuJoCo PrimalUpdateConstraint, pyramidal rows: state = jar < 0, force = -D jar, qfrc_constraint = J' force,
    returns cost = sum 0.5 D jar^2 over active rows + Gauss."""
    var cost: Float32 = 0
    for v in range(NV):
        res[(OQFC + v) * F + f] = 0
    for p in range(NCP):
        var nch = Int(mi[CPH + 8 * p + 6]); var choff = Int(mi[CPH + 8 * p + 7])
        for sl in range(3):
            if get(res, f, OVALID + 3 * p + sl) == 0:
                continue
            for k in range(4):
                var row = 4 * (3 * p + sl) + k
                var jar = get(res, f, OJAR + row)
                var act = jar < 0
                res[(OST + row) * F + f] = 1 if act else 0
                if act:
                    var D = get(res, f, OD + row)
                    cost += 0.5 * D * jar * jar
                    var force = -D * jar
                    var jadr = OJ + 12 * choff + (4 * sl + k) * nch
                    for c in range(nch):
                        var v = Int(mi[CCHAIN + choff + c])
                        res[(OQFC + v) * F + f] += get(res, f, jadr + c) * force
    for v in range(NV):
        cost += 0.5 * (get(res, f, OMA + v) - get(res, f, OQSM + v)) * (get(res, f, OCQACC + v) - get(res, f, OQACCS + v))
    return cost


def hessian(mi: TI, res: TO, f: Int):
    """MuJoCo MakeHessian + FactorizeHessian: H = M + J' D J over active rows, in M's own sparse pattern (a foot's
    dof chain is exactly the pattern of its deepest row), factored L'DL with mj_factorI."""
    for c in range(NC):
        res[(OHN + c) * F + f] = get(res, f, OM + c)
    for p in range(NCP):
        var nch = Int(mi[CPH + 8 * p + 6]); var choff = Int(mi[CPH + 8 * p + 7])
        for sl in range(3):
            if get(res, f, OVALID + 3 * p + sl) == 0:
                continue
            for k in range(4):
                var row = 4 * (3 * p + sl) + k
                if get(res, f, OST + row) == 0:
                    continue
                var D = get(res, f, OD + row)
                var jadr = OJ + 12 * choff + (4 * sl + k) * nch
                for a in range(nch):
                    var ja = get(res, f, jadr + a) * D
                    var ra = Int(mi[MADR + Int(mi[CCHAIN + choff + a])])
                    for bb in range(a + 1):
                        res[(OHN + ra + bb) * F + f] += ja * get(res, f, jadr + bb)
    factorI(mi, res, f, OHN, OHNI)


def gradient(mi: TI, res: TO, f: Int):
    """MuJoCo PrimalUpdateGradient (Newton): grad = M qacc - qfrc_smooth - qfrc_constraint, search = -H^-1 grad."""
    for v in range(NV):
        var g = get(res, f, OMA + v) - get(res, f, OQSM + v) - get(res, f, OQFC + v)
        res[(OGRAD + v) * F + f] = g
        res[(OSRCH + v) * F + f] = g
    solveLD(mi, res, f, OSRCH, OHN, OHNI)
    for v in range(NV):
        res[(OSRCH + v) * F + f] = -get(res, f, OSRCH + v)


comptime PNT = SIMD[DType.float32, 4]  # line-search point: alpha, cost, deriv0, deriv1


def peval(mi: TI, res: TO, f: Int, gauss: PNT, alpha: Float32) -> PNT:
    """MuJoCo PrimalEval for pyramidal rows: Gauss quadratic plus every row active at jar + alpha jv."""
    var q0 = gauss[0]; var q1 = gauss[1]; var q2 = gauss[2]
    for r in range(NEFC):
        if get(res, f, OVALID + r // 4) == 0:
            continue
        var jar = get(res, f, OJAR + r); var jv = get(res, f, OJV + r)
        if jar + alpha * jv < 0:
            var D = get(res, f, OD + r)
            q0 += 0.5 * D * jar * jar
            q1 += jv * D * jar
            q2 += 0.5 * jv * D * jv
    res[(ODIAG + 1) * F + f] += 1
    var d1 = 2 * q2
    return PNT(alpha, alpha * alpha * q2 + alpha * q1 + q0, 2 * alpha * q2 + q1, d1 if d1 > 0 else Float32(1e-15))


@always_inline
def bracket(mut p: PNT, c0: PNT, c1: PNT, c2: PNT) -> Bool:
    """MuJoCo updateBracket, the point update; the caller evaluates the next Newton point."""
    var flag = False
    for i in range(3):
        var c = c0 if i == 0 else (c1 if i == 1 else c2)
        if p[2] < 0 and c[2] < 0 and p[2] < c[2]:
            p = c
            flag = True
        elif p[2] > 0 and c[2] > 0 and p[2] > c[2]:
            p = c
            flag = True
    return flag


def search(mi: TI, res: TO, f: Int, scale: Float32, tolerance: Float32, lsiter: Int) -> Float32:
    """MuJoCo PrimalSearch: Newton steps along the search direction, then a bracketed search; returns alpha."""
    var snorm: Float32 = 0
    for v in range(NV):
        snorm += get(res, f, OSRCH + v) * get(res, f, OSRCH + v)
    snorm = sqrt(snorm)
    if snorm < 1e-15:
        return 0
    var gtol = tolerance * snorm / scale
    mulM(mi, res, f, OMV, OSRCH)
    for p in range(NCP):
        var nch = Int(mi[CPH + 8 * p + 6]); var choff = Int(mi[CPH + 8 * p + 7])
        for sl in range(3):
            if get(res, f, OVALID + 3 * p + sl) == 0:
                continue
            for k in range(4):
                res[(OJV + 4 * (3 * p + sl) + k) * F + f] = jdot(mi, res, f, OJ + 12 * choff + (4 * sl + k) * nch, nch, choff, OSRCH)
    var g1: Float32 = 0; var g2: Float32 = 0; var g0: Float32 = 0
    for v in range(NV):
        var sv = get(res, f, OSRCH + v)
        g1 += sv * (get(res, f, OMA + v) - get(res, f, OQSM + v))
        g2 += 0.5 * sv * get(res, f, OMV + v)
        g0 += 0.5 * (get(res, f, OMA + v) - get(res, f, OQSM + v)) * (get(res, f, OCQACC + v) - get(res, f, OQACCS + v))
    var gauss = PNT(g0, g1, g2, 0)
    var it = 0
    var p0 = peval(mi, res, f, gauss, 0); it += 1
    gtol = max(gtol, LSFLOOR * abs(p0[2]))  # float32: the slope cannot fall below ~1e-7 of its start
    var p1 = peval(mi, res, f, gauss, p0[0] - p0[2] / p0[3]); it += 1
    if abs(p1[2]) < gtol:
        return p1[0]
    var dir: Float32 = 1 if p1[2] < 0 else -1
    var p2 = p0
    while p1[2] * dir <= -gtol and it < lsiter:
        p2 = p1
        p1 = peval(mi, res, f, gauss, p1[0] - p1[2] / p1[3]); it += 1
        if abs(p1[2]) < gtol:
            return p1[0]
    if it >= lsiter:
        return p1[0]
    var p2next = p1
    var p1next = peval(mi, res, f, gauss, p1[0] - p1[2] / p1[3]); it += 1
    var pmid = p0
    while it < lsiter:
        pmid = peval(mi, res, f, gauss, 0.5 * (p1[0] + p2[0])); it += 1
        var bestcost: Float32 = 0
        var bestind = -1
        for i in range(3):
            var c = p1next if i == 0 else (p2next if i == 1 else pmid)
            if abs(c[2]) < gtol and (bestind == -1 or c[1] < bestcost):
                bestcost = c[1]
                bestind = i
        if bestind >= 0:
            return p1next[0] if bestind == 0 else (p2next[0] if bestind == 1 else pmid[0])
        var c0 = p1next; var c1 = p2next; var c2 = pmid
        var b1 = bracket(p1, c0, c1, c2)
        if b1:
            p1next = peval(mi, res, f, gauss, p1[0] - p1[2] / p1[3]); it += 1
        var b2 = bracket(p2, c0, c1, c2)
        if b2:
            p2next = peval(mi, res, f, gauss, p2[0] - p2[2] / p2[3]); it += 1
        if not b1 and not b2:
            return pmid[0]
    if p1[1] <= p2[1] and p1[1] < p0[1]:
        return p1[0]
    if p2[1] <= p1[1] and p2[1] < p0[1]:
        return p2[0]
    return 0


def newton(mi: TI, mf: TF, res: TO, f: Int):
    """MuJoCo mj_solPrimal (Newton), one island: start at qacc_smooth, line search, Hessian rebuilt when the
    active set changes (MuJoCo updates it incrementally; same matrix)."""
    for v in range(NV):
        res[(OQSM + v) * F + f] = get(res, f, OPAS + v) - get(res, f, OBIAS + v)
        comptime if STAGE >= 7:
            res[(OQSM + v) * F + f] += get(res, f, OQACT + v)
        res[(OQACCS + v) * F + f] = get(res, f, OQSM + v)
    solveLD(mi, res, f, OQACCS, OLD, OLDI)
    var scale: Float32 = 0
    for v in range(NV):
        res[(OCQACC + v) * F + f] = get(res, f, OQACCS + v)
        scale += get(res, f, OM + Int(mi[MADR + v]) + Int(mi[MNNZ + v]) - 1)
    scale = 1 / scale
    mulM(mi, res, f, OMA, OCQACC)
    var ncon: Float32 = 0
    for p in range(NCP):
        var nch = Int(mi[CPH + 8 * p + 6]); var choff = Int(mi[CPH + 8 * p + 7])
        for sl in range(3):
            if get(res, f, OVALID + 3 * p + sl) == 0:
                continue
            ncon += 1
            for k in range(4):
                var row = 4 * (3 * p + sl) + k
                res[(OJAR + row) * F + f] = jdot(mi, res, f, OJ + 12 * choff + (4 * sl + k) * nch, nch, choff, OCQACC) - get(res, f, OAREF + row)
    res[ONCON * F + f] = ncon
    var tol = mf[COPT + 1]
    var cost = update(mi, res, f)
    hessian(mi, res, f)
    gradient(mi, res, f)
    res[ODIAG * F + f] = 0
    res[(ODIAG + 1) * F + f] = 0
    for _ in range(Int(mi[CITER])):
        res[ODIAG * F + f] += 1
        var alpha = search(mi, res, f, scale, tol * mf[COPT + 2], Int(mi[CITER + 1]))
        if alpha == 0:
            break
        for v in range(NV):
            res[(OCQACC + v) * F + f] += alpha * get(res, f, OSRCH + v)
            res[(OMA + v) * F + f] += alpha * get(res, f, OMV + v)
        var changed = False
        for r in range(NEFC):
            res[(OJAR + r) * F + f] += alpha * get(res, f, OJV + r)
            if get(res, f, OVALID + r // 4) != 0 and (get(res, f, OJAR + r) < 0) != (get(res, f, OST + r) != 0):
                changed = True
        var oldcost = cost
        cost = update(mi, res, f)
        if changed:
            hessian(mi, res, f)
        gradient(mi, res, f)
        var gn: Float32 = 0
        for v in range(NV):
            gn += get(res, f, OGRAD + v) * get(res, f, OGRAD + v)
        if scale * (oldcost - cost) < tol or scale * sqrt(gn) < tol:
            break
        # float32 stop: MuJoCo's 1e-8 tests sit below float32 noise, so a few animals ran all 100 iterations and
        # held their wave. Stop once a step moves qacc by under 1e-6 of its size (max norms): further steps only
        # move it within float32 rounding. (Stopping when a full step kept the active set was wrong: with stiff
        # contacts H is ill-conditioned, so one float32 Newton step is not exact.)
        var dq: Float32 = 0
        var qa: Float32 = 0
        for v in range(NV):
            dq = max(dq, abs(alpha * get(res, f, OSRCH + v)))
            qa = max(qa, abs(get(res, f, OCQACC + v)))
        if dq <= STEPTOL * qa:
            break


def euler(mi: TI, mf: TF, qpos: TQ, qvel: TV, res: TO, f: Int):
    """MuJoCo mj_Euler with implicit dof damping (contact-free: qfrc = qfrc_passive - qfrc_bias): qH = M + h diag(damping),
    factor, solve qH qacc = qfrc (mj_solveLD), qvel += h qacc, mj_integratePos. qvel is read, not written back."""
    var h = mf[TSTEP]
    for c in range(NC):
        res[(OQH + c) * F + f] = get(res, f, OM + c)
    for v in range(NV):
        res[(OQH + Int(mi[MADR + v]) + Int(mi[MNNZ + v]) - 1) * F + f] += h * mf[DAMP + v]
    factorI(mi, res, f, OQH, OQHI)
    for v in range(NV):
        res[(OQACC + v) * F + f] = get(res, f, OPAS + v) - get(res, f, OBIAS + v)
        comptime if STAGE >= 6:
            res[(OQACC + v) * F + f] += get(res, f, OQFC + v)
        comptime if STAGE >= 7:
            res[(OQACC + v) * F + f] += get(res, f, OQACT + v)
    solveLD(mi, res, f, OQACC, OQH, OQHI)
    for a in range(NQ):
        res[(OQPOS + a) * F + f] = qpos[a * F + f]
    for j in range(NJ):  # mj_integratePos with the new qvel
        var a = Int(mi[QADR + j])
        var da = Int(mi[DOFADR + j])
        var t = Int(mi[JTYPE + j])
        if t == 0:
            for c in range(3):
                res[(OQPOS + a + c) * F + f] += h * (qvel[(da + c) * F + f] + h * get(res, f, OQACC + da + c))
            a += 3
            da += 3
        if t <= 1:  # mju_quatIntegrate: q = unit(q) * quat(axis, h |w|), w in the body frame
            var w = V3(0)
            for c in range(3):
                w[c] = qvel[(da + c) * F + f] + h * get(res, f, OQACC + da + c)
            var n = sqrt((w * w).reduce_add())
            var axis = w / n if n >= 1e-15 else basis(0)
            var half = 0.5 * h * n
            var s = sin(half)
            var q = unit(Q4(get(res, f, OQPOS + a), get(res, f, OQPOS + a + 1), get(res, f, OQPOS + a + 2), get(res, f, OQPOS + a + 3)))
            put(res, f, OQPOS + a, mul(q, Q4(cos(half), axis[0] * s, axis[1] * s, axis[2] * s)), 4)
        else:
            res[(OQPOS + a) * F + f] += h * (qvel[da * F + f] + h * get(res, f, OQACC + da))


def origin(mi: TI, qpos: TQ, f: Int, b: Int) -> V3:
    """Floating origin of body b's tree: the position of its root's free joint (each animal its own), else 0 (the
    world and trees fixed to it). The step computes every position relative to it, so differences of nearby points
    stay exact in float32 however far each animal is from the world origin or from the other animals (one shared
    origin put a spider 8 mm from the fly at a ~100x float32 input floor)."""
    var r = Int(mi[ROOT + b])
    var ja = Int(mi[JNTADR + r])
    if r > 0 and Int(mi[JNTNUM + r]) >= 1 and mi[JTYPE + ja] == 0:
        var a = Int(mi[QADR + ja])
        return V3(qpos[a * F + f], qpos[(a + 1) * F + f], qpos[(a + 2) * F + f], 0)
    return V3(0)


def absolute(mi: TI, qpos: TQ, res: TO, f: Int):
    """Back to world positions, once per step, in the rows a later reader takes as world: xpos, xipos, geom_xpos,
    xanchor, subtree_com (STAGE >= 2), each by its own tree's origin (world geoms are already world)."""
    for i in range(NB - 1):
        var o = origin(mi, qpos, f, i + 1)
        put(res, f, 3 * i, V3(get(res, f, 3 * i), get(res, f, 3 * i + 1), get(res, f, 3 * i + 2), 0) + o, 3)
        put(res, f, OXI + 3 * i, V3(get(res, f, OXI + 3 * i), get(res, f, OXI + 3 * i + 1), get(res, f, OXI + 3 * i + 2), 0) + o, 3)
    for g in range(NG):
        var o = origin(mi, qpos, f, Int(mi[GBODY + g]))
        put(res, f, OG + 3 * g, V3(get(res, f, OG + 3 * g), get(res, f, OG + 3 * g + 1), get(res, f, OG + 3 * g + 2), 0) + o, 3)
    for i in range(1, NB):
        var o = origin(mi, qpos, f, i)
        for j in range(Int(mi[JNTADR + i]), Int(mi[JNTADR + i]) + Int(mi[JNTNUM + i])):
            put(res, f, OA + 3 * j, V3(get(res, f, OA + 3 * j), get(res, f, OA + 3 * j + 1), get(res, f, OA + 3 * j + 2), 0) + o, 3)
    comptime if STAGE >= 2:
        for i in range(NB):
            var o = origin(mi, qpos, f, i)
            put(res, f, OSC + 3 * i, V3(get(res, f, OSC + 3 * i), get(res, f, OSC + 3 * i + 1), get(res, f, OSC + 3 * i + 2), 0) + o, 3)


def smooth(mi: TI, mf: TF, qpos: TQ, qvel: TV, ctrl: TU, res: TO):
    var f = block_idx.x * block_dim.x + thread_idx.x
    if f >= F:
        return
    for _ in range(max(K, 1)):
        step(mi, mf, qpos, qvel, ctrl, res, f)
        absolute(mi, qpos, res, f)
        comptime if K > 0:  # mj_Euler's write-back: qvel += h qacc, qpos = integrated qpos
            for v in range(NV):
                qvel[v * F + f] += mf[TSTEP] * get(res, f, OQACC + v)
            for a in range(NQ):
                qpos[a * F + f] = get(res, f, OQPOS + a)


def step(mi: TI, mf: TF, qpos: TQ, qvel: TV, ctrl: TU, res: TO, f: Int):

    for i in range(1, NB):
        var ja = Int(mi[JNTADR + i])
        var jn = Int(mi[JNTNUM + i])
        var xpos: V3
        var xquat: Q4
        if jn == 1 and mi[JTYPE + ja] == 0:  # free joint: pose straight from qpos
            var a = Int(mi[QADR + ja])
            xpos = V3(0)  # its own tree's origin
            xquat = unit(Q4(qpos[(a + 3) * F + f], qpos[(a + 4) * F + f], qpos[(a + 5) * F + f], qpos[(a + 6) * F + f]))
            put(res, f, OA + 3 * ja, xpos, 3)
            put(res, f, OX + 3 * ja, m3(mf, JAXIS + 3 * ja), 3)
        else:
            var pid = Int(mi[PARENT + i])
            var p = body(res, f, pid)
            xpos = rot(p[1], m3(mf, BPOS + 3 * i)) + p[0] if pid else m3(mf, BPOS + 3 * i)  # a fixed tree's origin is 0
            xquat = mul(p[1], m4(mf, BQUAT + 4 * i)) if pid else m4(mf, BQUAT + 4 * i)
            for j in range(ja, ja + jn):
                var a = Int(mi[QADR + j])
                var t = mi[JTYPE + j]
                var xaxis = rot(xquat, m3(mf, JAXIS + 3 * j))
                var xanchor = rot(xquat, m3(mf, JPOS + 3 * j)) + xpos
                if t == 2:  # slide
                    xpos += xaxis * (qpos[a * F + f] - mf[QPOS0 + a])
                else:  # ball (1) or hinge (3)
                    var loc: Q4
                    if t == 1:
                        loc = unit(Q4(qpos[a * F + f], qpos[(a + 1) * F + f], qpos[(a + 2) * F + f], qpos[(a + 3) * F + f]))
                    else:
                        var h = 0.5 * (qpos[a * F + f] - mf[QPOS0 + a])
                        var s = sin(h)
                        loc = Q4(cos(h), mf[JAXIS + 3 * j] * s, mf[JAXIS + 3 * j + 1] * s, mf[JAXIS + 3 * j + 2] * s)
                    xquat = mul(xquat, loc)
                    xpos = xanchor - rot(xquat, m3(mf, JPOS + 3 * j))
                put(res, f, OA + 3 * j, xanchor, 3)
                put(res, f, OX + 3 * j, xaxis, 3)
        xquat = unit(xquat)
        put(res, f, 3 * (i - 1), xpos, 3)
        put(res, f, OXQ + 4 * (i - 1), xquat, 4)
        put(res, f, OXI + 3 * (i - 1), rot(xquat, m3(mf, IPOS + 3 * i)) + xpos, 3)
    for g in range(NG):
        var b = body(res, f, Int(mi[GBODY + g]))
        put(res, f, OG + 3 * g, rot(b[1], m3(mf, GPOS + 3 * g)) + b[0], 3)
    comptime if STOP == 1:
        return
    comptime if STAGE >= 2:
        # mj_comPos: subtree_com = sum(mass x xipos) over the subtree / subtree mass
        for i in range(NB):
            var xi = V3(0) if i == 0 else V3(get(res, f, OXI + 3 * (i - 1)), get(res, f, OXI + 3 * (i - 1) + 1), get(res, f, OXI + 3 * (i - 1) + 2), 0)
            put(res, f, OSC + 3 * i, xi * mf[MASS + i], 3)
        for i in range(NB - 1, 0, -1):
            var p = Int(mi[PARENT + i])
            var o = origin(mi, qpos, f, i) * mf[SUBMASS + i] if p == 0 else V3(0)  # a tree's sum into the world: world frame
            for c in range(3):
                res[(OSC + 3 * p + c) * F + f] += get(res, f, OSC + 3 * i + c) + o[c]
        for i in range(NB):
            var sm = mf[SUBMASS + i]
            for c in range(3):
                if sm < 1e-15:
                    res[(OSC + 3 * i + c) * F + f] = 0 if i == 0 else get(res, f, OXI + 3 * (i - 1) + c)
                else:
                    res[(OSC + 3 * i + c) * F + f] = get(res, f, OSC + 3 * i + c) / sm
        # cinert: body inertia about subtree_com of its root, in the world frame (mju_inertCom)
        for i in range(1, NB):
            var r = 3 * Int(mi[ROOT + i])
            var dif = V3(get(res, f, OXI + 3 * (i - 1)) - get(res, f, OSC + r), get(res, f, OXI + 3 * (i - 1) + 1) - get(res, f, OSC + r + 1),
                         get(res, f, OXI + 3 * (i - 1) + 2) - get(res, f, OSC + r + 2), 0)
            var b = body(res, f, i)
            var iq = mul(b[1], m4(mf, IQUAT + 4 * i))
            var c0 = col(iq, 0); var c1 = col(iq, 1); var c2 = col(iq, 2)
            var I = m3(mf, INERTIA + 3 * i)
            var mass = mf[MASS + i]
            # mat * diag(I) * mat': entry (a, b) = sum_k mat[a][k] I_k mat[b][k], column k of mat = c_k
            var o = OCI + 10 * (i - 1)
            for a in range(3):
                for bb in range(a, 3):
                    var v = c0[a] * I[0] * c0[bb] + c1[a] * I[1] * c1[bb] + c2[a] * I[2] * c2[bb]
                    var slot = a if a == bb else (3 if a == 0 and bb == 1 else (4 if a == 0 else 5))
                    res[(o + slot) * F + f] = v
            res[(o + 0) * F + f] += mass * (dif[1] * dif[1] + dif[2] * dif[2])
            res[(o + 1) * F + f] += mass * (dif[0] * dif[0] + dif[2] * dif[2])
            res[(o + 2) * F + f] += mass * (dif[0] * dif[0] + dif[1] * dif[1])
            res[(o + 3) * F + f] -= mass * dif[0] * dif[1]
            res[(o + 4) * F + f] -= mass * dif[0] * dif[2]
            res[(o + 5) * F + f] -= mass * dif[1] * dif[2]
            put(res, f, o + 6, dif * mass, 3)
            res[(o + 9) * F + f] = mass
        # cdof: motion dofs in the frame centred at subtree_com (mju_dofCom)
        for i in range(1, NB):
            var r = 3 * Int(mi[ROOT + i])
            var com = V3(get(res, f, OSC + r), get(res, f, OSC + r + 1), get(res, f, OSC + r + 2), 0)
            var bq = body(res, f, i)[1]
            for j in range(Int(mi[JNTADR + i]), Int(mi[JNTADR + i]) + Int(mi[JNTNUM + i])):
                var da = OCD + 6 * Int(mi[DOFADR + j])
                var t = mi[JTYPE + j]
                var anchor = V3(get(res, f, OA + 3 * j), get(res, f, OA + 3 * j + 1), get(res, f, OA + 3 * j + 2), 0)
                var axis = V3(get(res, f, OX + 3 * j), get(res, f, OX + 3 * j + 1), get(res, f, OX + 3 * j + 2), 0)
                var off = com - anchor
                if t == 0:  # free: 3 translations along world x, y, z, then 3 rotations as a ball
                    for k in range(3):
                        put(res, f, da + 6 * k, V3(0), 3)
                        put(res, f, da + 6 * k + 3, basis(k), 3)
                    da += 18
                if t <= 1:  # free or ball: rotation about each body axis
                    for k in range(3):
                        var ax = col(bq, k)
                        put(res, f, da + 6 * k, ax, 3)
                        put(res, f, da + 6 * k + 3, cross(ax, off), 3)
                elif t == 2:  # slide
                    put(res, f, da, V3(0), 3)
                    put(res, f, da + 3, axis, 3)
                else:  # hinge
                    put(res, f, da, axis, 3)
                    put(res, f, da + 3, cross(axis, off), 3)
        # mj_crb: crb = cinert, then accumulate each body into its parent, backward
        for c in range(10 * (NB - 1)):
            res[(OCRB + c) * F + f] = get(res, f, OCI + c)
        for i in range(NB - 1, 0, -1):
            var p = Int(mi[PARENT + i])
            if p > 0:
                for c in range(10):
                    res[(OCRB + 10 * (p - 1) + c) * F + f] += get(res, f, OCRB + 10 * (i - 1) + c)
        # M(i, j) for every dof i and its ancestors j, MuJoCo's sparse row layout
        for i in range(NV):
            var adr = OM + Int(mi[MADR + i]) + Int(mi[MNNZ + i]) - 1
            var c = OCRB + 10 * (Int(mi[DBODY + i]) - 1)
            var v = OCD + 6 * i
            var I = SIMD[DType.float32, 16](0)
            for k in range(10):
                I[k] = get(res, f, c + k)
            var d0 = get(res, f, v); var d1 = get(res, f, v + 1); var d2 = get(res, f, v + 2)
            var d3 = get(res, f, v + 3); var d4 = get(res, f, v + 4); var d5 = get(res, f, v + 5)
            var buf = SIMD[DType.float32, 8](  # mju_mulInertVec
                I[0] * d0 + I[3] * d1 + I[4] * d2 - I[8] * d4 + I[7] * d5,
                I[3] * d0 + I[1] * d1 + I[5] * d2 + I[8] * d3 - I[6] * d5,
                I[4] * d0 + I[5] * d1 + I[2] * d2 - I[7] * d3 + I[6] * d4,
                I[8] * d1 - I[7] * d2 + I[9] * d3,
                I[6] * d2 - I[8] * d0 + I[9] * d4,
                I[7] * d0 - I[6] * d1 + I[9] * d5, 0, 0)
            var j = i
            var first = True
            while j >= 0:
                var w = OCD + 6 * j
                var dot: Float32 = 0
                for k in range(6):
                    dot += get(res, f, w + k) * buf[k]
                res[adr * F + f] = (mf[ARMATURE + i] if first else Float32(0)) + dot
                first = False
                adr -= 1
                j = Int(mi[DPARENT + j])
    comptime if STOP == 2:
        return
    comptime if STAGE >= 3:
        for c in range(NC):
            res[(OLD + c) * F + f] = get(res, f, OM + c)
        factorI(mi, res, f, OLD, OLDI)
    comptime if STOP == 3:
        return
    comptime if STAGE >= 4:
        velocity(mi, mf, qpos, qvel, res, f)
    comptime if STOP == 4:
        return
    comptime if STAGE >= 6:
        contacts(mi, mf, qpos, qvel, res, f)
        comptime if STAGE >= 7:
            actuation(mi, mf, qpos, qvel, ctrl, res, f)
        comptime if STOP == 5:
            return
        newton(mi, mf, res, f)
        comptime if STOP == 6:
            return
    comptime if STAGE >= 5:
        euler(mi, mf, qpos, qvel, res, f)

def load(path: String, size: Int) raises -> List[UInt8]:
    var raw = open(path, "r").read_bytes()
    if len(raw) != size * 4:
        raise Error(path + ": expected " + String(size) + " words, got " + String(len(raw) // 4))
    return raw^


def main() raises:
    var dir = String(argv()[1])
    # no MAX pool: every buffer is made once, so VRAM is exactly the buffers plus the HIP runtime; a caller wins
    _ = external_call["setenv", Int32]("MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT".unsafe_ptr(), "0".unsafe_ptr(), Int32(0))
    var ctx = DeviceContext(0)
    var mi_b = ctx.enqueue_create_buffer[DType.int32](NI)
    var mf_b = ctx.enqueue_create_buffer[DType.float32](NFL)
    var q_b = ctx.enqueue_create_buffer[DType.float32](NQ * F)
    var v_b = ctx.enqueue_create_buffer[DType.float32](NV * F)
    var u_b = ctx.enqueue_create_buffer[DType.float32](max(NU, 1) * F)
    var o_b = ctx.enqueue_create_buffer[DType.float32](NROW * F)
    var mi = load(dir + "mi.bin", NI)
    var mf = load(dir + "mf.bin", NFL)
    var qp = load(dir + "qpos.bin", NQ * F)
    var qv = load(dir + "qvel.bin", NV * F)
    var uv = load(dir + "ctrl.bin", NU * F) if NU > 0 else List[UInt8](length=4 * F, fill=0)
    ctx.enqueue_copy(mi_b, mi.unsafe_ptr().unsafe_bitcast[Int32]())
    ctx.enqueue_copy(mf_b, mf.unsafe_ptr().unsafe_bitcast[Float32]())
    ctx.enqueue_copy(q_b, qp.unsafe_ptr().unsafe_bitcast[Float32]())
    ctx.enqueue_copy(v_b, qv.unsafe_ptr().unsafe_bitcast[Float32]())
    ctx.enqueue_copy(u_b, uv.unsafe_ptr().unsafe_bitcast[Float32]())
    var kernel = ctx.compile_function[smooth]()
    var grid = (F + 63) // 64
    ctx.enqueue_function(kernel, TileTensor(mi_b, LI), TileTensor(mf_b, LF), TileTensor(q_b, LQ), TileTensor(v_b, LV), TileTensor(u_b, LU), TileTensor(o_b, LO), grid_dim=grid, block_dim=64)
    ctx.synchronize()
    comptime if K > 0:
        var state = List[Float32](length=(NQ + NV) * F, fill=0)
        ctx.enqueue_copy(state.unsafe_ptr(), q_b)
        ctx.enqueue_copy(state.unsafe_ptr().unsafe_offset(NQ * F), v_b)
        ctx.synchronize()
        with open(dir + "state.bin", "w") as file:
            file.write_bytes(Span(unsafe_ptr=state.unsafe_ptr().unsafe_bitcast[UInt8](), length=(NQ + NV) * F * 4))
        comptime if 10 % K == 0:  # state after 10 steps (10 / K launches): K=1 and K=10 must give the same bytes
            for _ in range(10 // K - 1):
                ctx.enqueue_function(kernel, TileTensor(mi_b, LI), TileTensor(mf_b, LF), TileTensor(q_b, LQ), TileTensor(v_b, LV), TileTensor(u_b, LU), TileTensor(o_b, LO), grid_dim=grid, block_dim=64)
            ctx.enqueue_copy(state.unsafe_ptr(), q_b)
            ctx.enqueue_copy(state.unsafe_ptr().unsafe_offset(NQ * F), v_b)
            ctx.synchronize()
            with open(dir + "state10.bin", "w") as file:
                file.write_bytes(Span(unsafe_ptr=state.unsafe_ptr().unsafe_bitcast[UInt8](), length=(NQ + NV) * F * 4))
    var t0 = perf_counter_ns()
    for _ in range(100):
        ctx.enqueue_function(kernel, TileTensor(mi_b, LI), TileTensor(mf_b, LF), TileTensor(q_b, LQ), TileTensor(v_b, LV), TileTensor(u_b, LU), TileTensor(o_b, LO), grid_dim=grid, block_dim=64)
    ctx.synchronize()
    var us = Float64(perf_counter_ns() - t0) / 100 / 1000
    var host = List[Float32](length=NROW * F, fill=0)
    ctx.enqueue_copy(host.unsafe_ptr(), o_b)
    ctx.synchronize()
    with open(dir + "smooth.bin", "w") as file:
        file.write_bytes(Span(unsafe_ptr=host.unsafe_ptr().unsafe_bitcast[UInt8](), length=NOUT * F * 4))
    print(F, "animals, stage", STAGE, ", K", max(K, 1), ":", us, "us per launch,", Float64(F * max(K, 1)) / us * 1e6, "animal-steps/s")
    comptime if STAGE >= 6:
        var it: Float64 = 0; var ev: Float64 = 0; var itmax: Float32 = 0
        for a in range(F):
            it += Float64(host[ODIAG * F + a]); ev += Float64(host[(ODIAG + 1) * F + a]); itmax = max(itmax, host[ODIAG * F + a])
        print("Newton iterations per animal", it / Float64(F), "max", itmax, "; line-search evaluations", ev / Float64(F))
