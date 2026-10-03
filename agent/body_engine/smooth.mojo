"""Smooth dynamics, ported from MuJoCo 3.9.0 src/engine/engine_core_smooth.c (Apache-2.0), batched:
one GPU thread per animal. STAGE 1: mj_kinematics1 + the inertia and geom part of mj_kinematics2.
STAGE 2: + mj_comPos (subtree_com, cinert, cdof) + mj_crb (sparse M in MuJoCo's own M_rowadr layout).
STAGE 3: + mj_factorM (mj_factorI: sparse L'DL of M in place, qLD, qLDiagInv).
STAGE 4: + mj_comVel (cvel, cdof_dot), mj_passive (joint springs, dof dampers), mj_rne without qacc (qfrc_bias).
STAGE 5: + mj_Euler, contact-free: implicit dof damping (qH = M + h diag(damping)), qacc, mj_integratePos.

Model arrays come from agent/body_engine/model.py export (mi.bin int32, mf.bin float32, field order
INTS / FLOATS there); sizes are -D constants, so any MJCF body (fly, spider) builds its own binary.
Orientation is kept as quaternions only (MuJoCo also builds 3x3 xmat; rotating by the quaternion is
the same map and stores 4 floats, not 9). Mocap bodies and the sleep filter are not ported.

Layout: every per-fly array is [row][fly], flies fastest, so a warp reads 32 neighbouring flies
in one transaction. Model arrays are shared by all flies and stay in cache.

usage: smooth DIR/      reads DIR/mi.bin, mf.bin, qpos.bin (NQ x FLIES), qvel.bin (NV x FLIES); writes DIR/smooth.bin, rows:
       xpos, xquat, xipos (bodies 1..), geom_xpos, xanchor, xaxis; STAGE 2 adds subtree_com,
       cinert (bodies 1..), cdof, crb (bodies 1..), M; STAGE 3 adds qLD, qLDiagInv; STAGE 4 adds cvel (bodies
       1..), cdof_dot, qfrc_passive, qfrc_bias; STAGE 5 adds qH, qHDiagInv, qacc_euler, qpos_next; prints us per launch
"""
from std.math import cos, sin, sqrt
from std.sys import argv, get_defined_int
from std.time import perf_counter_ns
from std.gpu import block_idx, block_dim, thread_idx
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major

comptime NB = get_defined_int["NBODY"]()
comptime NJ = get_defined_int["NJNT"]()
comptime NQ = get_defined_int["NQ"]()
comptime NG = get_defined_int["NGEOM"]()
comptime NV = get_defined_int["NV"]()
comptime NC = get_defined_int["NC"]()
comptime F = get_defined_int["FLIES"]()
comptime STAGE = get_defined_int["STAGE", 1]()
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
comptime NI = BDOFADR + NB
# mf.bin: body_pos, body_quat, jnt_pos, jnt_axis, qpos0, body_ipos, body_iquat, geom_pos, geom_quat
comptime BPOS = 0
comptime BQUAT = 3 * NB
comptime JPOS = 7 * NB
comptime JAXIS = 7 * NB + 3 * NJ
comptime QPOS0 = 7 * NB + 6 * NJ
comptime IPOS = 7 * NB + 6 * NJ + NQ
comptime GPOS = 14 * NB + 6 * NJ + NQ
comptime IQUAT = IPOS + 3 * NB
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
comptime NFL = TSTEP + 1
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
comptime NOUT = OSC if STAGE == 1 else (OM + NC if STAGE == 2 else (OLDI + NV if STAGE == 3 else (OQH if STAGE == 4 else OQPOS + NQ)))
comptime OCACC = NOUT  # scratch rows after the output: cacc, cfrc_body per body (world included)
comptime OCFRC = OCACC + 6 * NB
comptime NROW = OCFRC + 6 * NB if STAGE >= 4 else NOUT
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
    for i in range(NV - 1, -1, -1):  # x <- L^-T x
        var start = Int(mi[MADR + i])
        var x_i = get(res, f, OQACC + i)
        for adr in range(start, start + Int(mi[MNNZ + i]) - 1):
            res[(OQACC + Int(mi[MCOL + adr])) * F + f] -= get(res, f, OQH + adr) * x_i
    for i in range(NV):  # x <- D^-1 x
        res[(OQACC + i) * F + f] *= get(res, f, OQHI + i)
    for i in range(NV):  # x <- L^-1 x
        var start = Int(mi[MADR + i])
        var dot: Float32 = 0
        for adr in range(start, start + Int(mi[MNNZ + i]) - 1):
            dot += get(res, f, OQH + adr) * get(res, f, OQACC + Int(mi[MCOL + adr]))
        res[(OQACC + i) * F + f] -= dot
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


def smooth(mi: TI, mf: TF, qpos: TQ, qvel: TV, res: TO):
    var f = block_idx.x * block_dim.x + thread_idx.x
    if f >= F:
        return

    for i in range(1, NB):
        var ja = Int(mi[JNTADR + i])
        var jn = Int(mi[JNTNUM + i])
        var xpos: V3
        var xquat: Q4
        if jn == 1 and mi[JTYPE + ja] == 0:  # free joint: pose straight from qpos
            var a = Int(mi[QADR + ja])
            xpos = V3(qpos[a * F + f], qpos[(a + 1) * F + f], qpos[(a + 2) * F + f], 0)
            xquat = unit(Q4(qpos[(a + 3) * F + f], qpos[(a + 4) * F + f], qpos[(a + 5) * F + f], qpos[(a + 6) * F + f]))
            put(res, f, OA + 3 * ja, xpos, 3)
            put(res, f, OX + 3 * ja, m3(mf, JAXIS + 3 * ja), 3)
        else:
            var pid = Int(mi[PARENT + i])
            var p = body(res, f, pid)
            xpos = rot(p[1], m3(mf, BPOS + 3 * i)) + p[0] if pid else m3(mf, BPOS + 3 * i)
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
    comptime if STAGE >= 2:
        # mj_comPos: subtree_com = sum(mass x xipos) over the subtree / subtree mass
        for i in range(NB):
            var xi = V3(0) if i == 0 else V3(get(res, f, OXI + 3 * (i - 1)), get(res, f, OXI + 3 * (i - 1) + 1), get(res, f, OXI + 3 * (i - 1) + 2), 0)
            put(res, f, OSC + 3 * i, xi * mf[MASS + i], 3)
        for i in range(NB - 1, 0, -1):
            var p = Int(mi[PARENT + i])
            for c in range(3):
                res[(OSC + 3 * p + c) * F + f] += get(res, f, OSC + 3 * i + c)
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
    comptime if STAGE >= 3:
        for c in range(NC):
            res[(OLD + c) * F + f] = get(res, f, OM + c)
        factorI(mi, res, f, OLD, OLDI)
    comptime if STAGE >= 4:
        velocity(mi, mf, qpos, qvel, res, f)
    comptime if STAGE >= 5:
        euler(mi, mf, qpos, qvel, res, f)

def load(path: String, size: Int) raises -> List[UInt8]:
    var raw = open(path, "r").read_bytes()
    if len(raw) != size * 4:
        raise Error(path + ": expected " + String(size) + " words, got " + String(len(raw) // 4))
    return raw^


def main() raises:
    var dir = String(argv()[1])
    var ctx = DeviceContext(0)
    var mi_b = ctx.enqueue_create_buffer[DType.int32](NI)
    var mf_b = ctx.enqueue_create_buffer[DType.float32](NFL)
    var q_b = ctx.enqueue_create_buffer[DType.float32](NQ * F)
    var v_b = ctx.enqueue_create_buffer[DType.float32](NV * F)
    var o_b = ctx.enqueue_create_buffer[DType.float32](NROW * F)
    var mi = load(dir + "mi.bin", NI)
    var mf = load(dir + "mf.bin", NFL)
    var qp = load(dir + "qpos.bin", NQ * F)
    var qv = load(dir + "qvel.bin", NV * F)
    ctx.enqueue_copy(mi_b, mi.unsafe_ptr().unsafe_bitcast[Int32]())
    ctx.enqueue_copy(mf_b, mf.unsafe_ptr().unsafe_bitcast[Float32]())
    ctx.enqueue_copy(q_b, qp.unsafe_ptr().unsafe_bitcast[Float32]())
    ctx.enqueue_copy(v_b, qv.unsafe_ptr().unsafe_bitcast[Float32]())
    var kernel = ctx.compile_function[smooth]()
    var grid = (F + 63) // 64
    ctx.enqueue_function(kernel, TileTensor(mi_b, LI), TileTensor(mf_b, LF), TileTensor(q_b, LQ), TileTensor(v_b, LV), TileTensor(o_b, LO), grid_dim=grid, block_dim=64)
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(100):
        ctx.enqueue_function(kernel, TileTensor(mi_b, LI), TileTensor(mf_b, LF), TileTensor(q_b, LQ), TileTensor(v_b, LV), TileTensor(o_b, LO), grid_dim=grid, block_dim=64)
    ctx.synchronize()
    var us = Float64(perf_counter_ns() - t0) / 100 / 1000
    var host = List[Float32](length=NROW * F, fill=0)
    ctx.enqueue_copy(host.unsafe_ptr(), o_b)
    ctx.synchronize()
    with open(dir + "smooth.bin", "w") as file:
        file.write_bytes(Span(unsafe_ptr=host.unsafe_ptr().unsafe_bitcast[UInt8](), length=NOUT * F * 4))
    print(F, "animals, stage", STAGE, ":", us, "us per launch,", Float64(F) / us * 1e6, "animals/s")
