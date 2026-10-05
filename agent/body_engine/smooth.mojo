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
from max.gpu.sync import barrier
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
comptime WAVE = get_defined_int["WAVE", 0]() == 1  # one animal per 32-lane wave (live): serial code runs on every lane
comptime LN = 32 if WAVE else 1  # alike; loops written range(lane(), n, LN) split over lanes, sync() before others read
comptime BLOCK = 32 if WAVE else 64
comptime GRID = F if WAVE else (F + 63) // 64
comptime STOP = get_defined_int["STOP", 99]()
comptime INTEGRATOR = get_defined_int["INTEGRATOR", 0]()  # opt.integrator: 0 Euler, 3 implicitfast (model.py export; absent = Euler)
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
# STAGE 6 contact block (model.py feet): per pair 11 ints, hull first edges, edge lists, dof chains, 2 solver ints
comptime NCP = get_defined_int["NCP", 0]()
comptime NHV = get_defined_int["NHV", 0]()
comptime NHE = get_defined_int["NHE", 0]()
comptime NCH = get_defined_int["NCH", 0]()
comptime NH = get_defined_int["NH", 0]()  # H's nonzeros (M's plus the fill of pairs that couple two dof chains)
comptime CPH = NI0
comptime CFIRST = CPH + 11 * NCP
comptime CEDGE = CFIRST + NHV
comptime CCHAIN = CEDGE + NHE
comptime CSIGN = CCHAIN + NCH
comptime HADR = CSIGN + NCH  # H's pattern: rowadr, rownnz, colind (same layout as M's at MADR), then M entry -> H entry
comptime MTOH = HADR + 2 * NV + NH
comptime CITER = MTOH + NC
comptime NU = get_defined_int["NU", 0]()
comptime CACT = CITER + 3  # STAGE 7 (model.py actuators): per actuator trntype, dof or body, ctrllimited, forcelimited, qpos address
comptime NI1 = CACT + 5 * NU if STAGE >= 7 else (CITER + 3 if STAGE >= 6 else NI0)
# model.py fluid() then tendons(), appended last: fluid ints per ellipsoid row (body, geom) then the inertia-box bodies
comptime NFLROW = get_defined_int["NFLROW", -1]()  # -1: no fluid block (walking models: the code below folds out)
comptime NFLBOX = get_defined_int["NFLBOX", 0]()
comptime NFLUID = NFLROW >= 0
comptime FLI = NI1
comptime NTEN = get_defined_int["NTEN", 0]()  # fixed tendons; tendon ints: wrap start and count per tendon, then each wrap's dof, then its qpos address
comptime NWRAP = get_defined_int["NWRAP", 0]()
comptime TNI = FLI + (2 * NFLROW + NFLBOX if NFLUID else 0)
comptime NI = TNI + 2 * NTEN + 2 * NWRAP
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
comptime CPF = NFL0  # STAGE 6: per pair margin, gap, friction 2, solref 2, solimp 5, invweight, rbound 2 and 1; hull verts; 3 opts
comptime CVERT = CPF + 14 * NCP
comptime COPT = CVERT + 3 * NHV
comptime HFG = get_defined_int["HFGEOM", -1]()  # heightfield ground geom (model.py feet), else none
comptime HFO = COPT + 5  # world.mojo params (NP)
comptime CACF = HFO + (NP if HFG >= 0 else 0)  # STAGE 7: per actuator gain, bias 0-2, ctrlrange 2, forcerange 2, gear
comptime NFL1 = CACF + 9 * NU if STAGE >= 7 else (CACF if STAGE >= 6 else NFL0)
comptime FLF = NFL1  # fluid floats: density, viscosity, wind 3, then per ellipsoid row semi-axes 3 and the 12 geom_fluid coefs
comptime TNF = FLF + (5 + 15 * NFLROW if NFLUID else 0)  # tendon floats: wrap coefs, then stiffness, damping, lengthspring 2 per tendon
comptime NFL = TNF + NWRAP + 4 * NTEN
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
comptime NSLOT = get_defined_int["NSLOT", 0]()  # contact slots: 3 per plane-mesh pair (MuJoCo maxplanemesh), 1 per sphere pair
comptime NJE = get_defined_int["NJE", 0]()  # Jacobian entries: 4 rows x slots x chain length, summed over pairs
comptime NEFC = 4 * NSLOT  # pyramidal condim 3: 4 rows per contact
comptime OFLW = OCFRC + 6 * NB  # NFLUID scratch: fluid wrench per body about its root's subtree_com
comptime OVALID = OFLW + (6 * NB if NFLUID else 0)  # STAGE 6 scratch: slot valid, per row D aref jar jv state, J rows, solver vectors, H
comptime OD = OVALID + NSLOT
comptime OAREF = OD + NEFC
comptime OJAR = OAREF + NEFC
comptime OJV = OJAR + NEFC
comptime OST = OJV + NEFC
comptime OJ = OST + NEFC
comptime OQSM = OJ + NJE
comptime OQACCS = OQSM + NV
comptime OMA = OQACCS + NV
comptime OGRAD = OMA + NV
comptime OSRCH = OGRAD + NV
comptime OMV = OSRCH + NV
comptime OHN = OMV + NV
comptime OHNI = OHN + NH
comptime ODIAG = OHNI + NV  # Newton iterations, line-search evaluations (printed by main)
comptime OFRC = ODIAG + 3  # Newton diagnostics, then noslip scratch
comptime OZ = OFRC + NEFC
comptime OW = OZ + NV
comptime OWC = OW + 2 * NV  # noslip per step: w of every valid row (laid out as J), then per row pair A 00 01 11, base 0 1
comptime OAC = OWC + NJE
comptime OACT = OAC + 10 * NSLOT  # valid slots: count, then their ids ascending (row loops skip the ~95 % empty)
comptime OPART = OACT + 1 + NSLOT  # WAVE: per-lane partial sums, 3 x LN
comptime OWARM = OPART + 3 * LN  # K > 0: last step's qacc (MuJoCo qacc_warmstart), persists across launches
comptime NROW = OWARM + NV if STAGE >= 6 else (OVALID if STAGE >= 4 else NOUT)
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


@always_inline
def lane() -> Int:
    comptime if WAVE:
        return Int(thread_idx.x)
    return 0


@always_inline
def sync():
    comptime if WAVE:
        barrier()


def factorI(mi: TI, res: TO, f: Int, L: Int, LI_: Int, A: Int = MADR):
    """MuJoCo mj_factorI: sparse L'DL of rows L.. in place, backward over rows k: L(i, 0..i) -= L(i, 0..i) L(k, i) / L(k, k),
    then L(k, :) /= L(k, k); 1 / L(k, k) to rows LI_.. Pattern at A: rowadr, rownnz, colind (M's, or H's at HADR).
    WAVE: row k's updates (j, c), c <= j, are independent (a row's pattern is its chain, so row j's is a prefix of k's)
    and spread over lanes (proto_factor.mojo, 10x at 1,024 animals)."""
    comptime if WAVE:
        sync()
        for k in range(NV - 1, -1, -1):
            var start = Int(mi[A + k])
            var diag = Int(mi[A + NV + k]) - 1
            var invD = 1 / get(res, f, L + start + diag)
            for t in range(lane(), diag * (diag + 1) // 2, LN):
                var j = Int((sqrt(Float32(8 * t + 1)) - 1) / 2)
                if (j + 1) * (j + 2) // 2 <= t:  # float rounding of the triangular root
                    j += 1
                var c = t - j * (j + 1) // 2
                var i = Int(mi[A + 2 * NV + start + j])
                res[(L + Int(mi[A + i]) + c) * F + f] += get(res, f, L + start + c) * (-get(res, f, L + start + j) * invD)
            sync()
            for c in range(lane(), diag, LN):
                res[(L + start + c) * F + f] = get(res, f, L + start + c) * invD
            res[(LI_ + k) * F + f] = invD
            sync()
        return
    for k in range(NV - 1, -1, -1):
        var start = Int(mi[A + k])
        var diag = Int(mi[A + NV + k]) - 1
        var end = start + diag
        var invD = 1 / get(res, f, L + end)
        res[(LI_ + k) * F + f] = invD
        for adr in range(end - 1, start - 1, -1):
            var i = Int(mi[A + 2 * NV + adr])
            var scl = -get(res, f, L + adr) * invD
            var ri = Int(mi[A + i])
            for c in range(Int(mi[A + NV + i])):
                res[(L + ri + c) * F + f] += get(res, f, L + start + c) * scl
        for c in range(diag):
            res[(L + start + c) * F + f] = get(res, f, L + start + c) * invD


def solveLD(mi: TI, res: TO, f: Int, X: Int, L: Int, LI_: Int, A: Int = MADR):
    """MuJoCo mj_solveLD, one vector: x <- (L'DL)^-1 x for the factor in rows L.., 1/D in rows LI_.."""
    for i in range(NV - 1, -1, -1):  # x <- L^-T x
        var start = Int(mi[A + i])
        var x_i = get(res, f, X + i)
        for adr in range(start, start + Int(mi[A + NV + i]) - 1):
            res[(X + Int(mi[A + 2 * NV + adr])) * F + f] -= get(res, f, L + adr) * x_i
    for i in range(NV):  # x <- D^-1 x
        res[(X + i) * F + f] *= get(res, f, LI_ + i)
    for i in range(NV):  # x <- L^-1 x
        var start = Int(mi[A + i])
        var dot: Float32 = 0
        for adr in range(start, start + Int(mi[A + NV + i]) - 1):
            dot += get(res, f, L + adr) * get(res, f, X + Int(mi[A + 2 * NV + adr]))
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


@always_inline
def pow2(a: Float32) -> Float32:
    return a * a


@always_inline
def pow4(a: Float32) -> Float32:
    return (a * a) * (a * a)


@always_inline
def local_wrench(mi: TI, mf: TF, res: TO, f: Int, b: Int, p: V3, q: Q4) -> S6:
    """MuJoCo mj_objectVelocity(flg_local = 1) of the frame (p, q) fixed to body b, minus wind: mju_transformSpatial of cvel
    from the root's subtree_com to p, rotated into the frame (engine_support.c, engine_util_spatial.c:478)."""
    var cv = get6(res, f, OCV + 6 * (b - 1))
    var r = Int(mi[ROOT + b])
    var dif = p - V3(get(res, f, OSC + 3 * r), get(res, f, OSC + 3 * r + 1), get(res, f, OSC + 3 * r + 2), 0)
    var w = V3(cv[0], cv[1], cv[2], 0)
    var v = V3(cv[3], cv[4], cv[5], 0) - cross(dif, w) - m3(mf, FLF + 2)
    var qi = Q4(q[0], -q[1], -q[2], -q[3])
    var lw = rot(qi, w)
    var lv = rot(qi, v)
    return S6(lw[0], lw[1], lw[2], lv[0], lv[1], lv[2], 0, 0)


@always_inline
def add_wrench(mi: TI, res: TO, f: Int, b: Int, p: V3, q: Q4, lfrc: S6):
    """Fold mj_applyFT: the local frame wrench rotated to world, taken to the root's subtree_com, added to the weld body's
    row. qfrc[v] = cdof[v] . W is the J'F + J'T chain walk, because cdof is a motion vector about that same com."""
    var bw = b
    while bw > 0 and mi[JNTNUM + bw] == 0:
        bw = Int(mi[PARENT + bw])
    if bw == 0:
        return
    var r = Int(mi[ROOT + b])
    var tq = rot(q, V3(lfrc[0], lfrc[1], lfrc[2], 0))
    var fo = rot(q, V3(lfrc[3], lfrc[4], lfrc[5], 0))
    tq += cross(p - V3(get(res, f, OSC + 3 * r), get(res, f, OSC + 3 * r + 1), get(res, f, OSC + 3 * r + 2), 0), fo)
    put6(res, f, OFLW + 6 * bw, get6(res, f, OFLW + 6 * bw) + S6(tq[0], tq[1], tq[2], fo[0], fo[1], fo[2], 0, 0))


def inertia_box(mi: TI, mf: TF, res: TO, f: Int, b: Int):
    """MuJoCo mj_inertiaBoxFluidModel (engine_passive.c:1054): box from the principal inertia, viscous sphere drag plus
    quadratic drag, about the body's centre of mass in the body frame."""
    var rho = mf[FLF]
    var mu = mf[FLF + 1]
    var m = mf[MASS + b]
    var I = m3(mf, INERTIA + 3 * b)
    var bx = V3(sqrt(max(Float32(1e-15), I[1] + I[2] - I[0]) / m * 6), sqrt(max(Float32(1e-15), I[0] + I[2] - I[1]) / m * 6),
                sqrt(max(Float32(1e-15), I[0] + I[1] - I[2]) / m * 6), 0)
    var bd = body(res, f, b)
    var p = V3(get(res, f, OXI + 3 * (b - 1)), get(res, f, OXI + 3 * (b - 1) + 1), get(res, f, OXI + 3 * (b - 1) + 2), 0)
    var q = mul(bd[1], m4(mf, IQUAT + 4 * b))  # ximat is the inertial frame's rotation (mj_objectVelocity, engine_core_util.c:827)
    var lv = local_wrench(mi, mf, res, f, b, p, q)
    var fr = S6(0)
    if mu > 0:
        var diam = (bx[0] + bx[1] + bx[2]) / 3
        for k in range(3):
            fr[k] = -3.14159265358979 * diam * diam * diam * mu * lv[k]
            fr[3 + k] = -3 * 3.14159265358979 * diam * mu * lv[3 + k]
    if rho > 0:
        fr[3] -= 0.5 * rho * bx[1] * bx[2] * abs(lv[3]) * lv[3]
        fr[4] -= 0.5 * rho * bx[0] * bx[2] * abs(lv[4]) * lv[4]
        fr[5] -= 0.5 * rho * bx[0] * bx[1] * abs(lv[5]) * lv[5]
        fr[0] -= rho * bx[0] * (pow4(bx[1]) + pow4(bx[2])) * abs(lv[0]) * lv[0] / 64
        fr[1] -= rho * bx[1] * (pow4(bx[0]) + pow4(bx[2])) * abs(lv[1]) * lv[1] / 64
        fr[2] -= rho * bx[2] * (pow4(bx[0]) + pow4(bx[1])) * abs(lv[2]) * lv[2] / 64
    add_wrench(mi, res, f, b, p, q, fr)


def ellipsoid(mi: TI, mf: TF, res: TO, f: Int, row: Int):
    """MuJoCo mj_ellipsoidFluidModel for one geom (engine_passive.c:1114) with mj_addedMassForces (:1176) and
    mj_viscousForces (:1230): added mass, Magnus and Kutta lift, blunt/slender drag, viscous resistance."""
    var rho = mf[FLF]
    var mu = mf[FLF + 1]
    var c = FLF + 5 + 15 * row
    var s = m3(mf, c)
    var coef = mf[c + 3]
    var blunt = mf[c + 4]
    var slender = mf[c + 5]
    var angd = mf[c + 6]
    var kutta = mf[c + 7]
    var magnus = mf[c + 8]
    var vm = m3(mf, c + 9)
    var vi = m3(mf, c + 12)
    var b = Int(mi[FLI + 2 * row])
    var g = Int(mi[FLI + 2 * row + 1])
    var bd = body(res, f, b)
    var q = mul(bd[1], m4(mf, GQUAT + 4 * g))
    var p = V3(get(res, f, OG + 3 * g), get(res, f, OG + 3 * g + 1), get(res, f, OG + 3 * g + 2), 0)
    var lv = local_wrench(mi, mf, res, f, b, p, q)
    var w = V3(lv[0], lv[1], lv[2], 0)
    var v = V3(lv[3], lv[4], lv[5], 0)
    # mj_addedMassForces
    var lin_mom = rho * vm * v
    var ang_mom = rho * vi * w
    var tq = cross(lin_mom, v) + cross(ang_mom, w)
    var fo = cross(lin_mom, w)
    # mj_viscousForces
    var volume = 4.0 / 3.0 * 3.14159265358979 * s[0] * s[1] * s[2]
    var dmax = max(max(s[0], s[1]), s[2])
    var dmin = min(min(s[0], s[1]), s[2])
    var dmid = s[0] + s[1] + s[2] - dmax - dmin
    var amax = 3.14159265358979 * dmax * dmid
    fo += cross(w, v) * (magnus * rho * volume)
    var pden = pow4(s[1] * s[2]) * pow2(v[0]) + pow4(s[2] * s[0]) * pow2(v[1]) + pow4(s[0] * s[1]) * pow2(v[2])
    var pnum = pow2(s[1] * s[2] * v[0]) + pow2(s[2] * s[0] * v[1]) + pow2(s[0] * s[1] * v[2])
    var aproj = 3.14159265358979 * sqrt(pden / max(Float32(1e-15), pnum))
    var nrm = V3(pow2(s[1] * s[2]) * v[0], pow2(s[2] * s[0]) * v[1], pow2(s[0] * s[1]) * v[2], 0)
    var vn = sqrt((v * v).reduce_add())
    var cosa = pnum / max(Float32(1e-15), vn * pden)
    var circ = cross(nrm, v) * (kutta * rho * cosa * aproj)
    fo += cross(circ, v)
    var dsph = 2.0 / 3.0 * (s[0] + s[1] + s[2])
    var imax = 8.0 / 15.0 * 3.14159265358979 * dmid * pow4(dmax)
    var mom = V3(0)
    for k in range(3):
        var ii = 8.0 / 15.0 * 3.14159265358979 * s[k] * pow4(max(s[(k + 1) % 3], s[(k + 2) % 3]))
        mom[k] = w[k] * (angd * ii + slender * (imax - ii))
    var drag_lin = mu * 3 * 3.14159265358979 * dsph + rho * vn * (aproj * blunt + slender * (amax - aproj))
    var drag_ang = mu * 3.14159265358979 * dsph * dsph * dsph + rho * sqrt((mom * mom).reduce_add())
    tq -= drag_ang * w
    fo -= drag_lin * v
    add_wrench(mi, res, f, b, p, q, S6(tq[0], tq[1], tq[2], fo[0], fo[1], fo[2], 0, 0) * coef)


def fluid(mi: TI, mf: TF, res: TO, f: Int):
    """MuJoCo mj_fluid (engine_passive.c:842) into qfrc_passive: every body with mass takes the ellipsoid model if any of its
    geoms has one (the exported rows), else the inertia box. Wrenches go to one row per weld body, are summed leaf to root
    like mj_rne's cfrc, and qfrc_fluid[v] = cdof[v] . W[body of v]."""
    for b in range(NB):
        put6(res, f, OFLW + 6 * b, S6(0))
    for r in range(NFLROW):
        ellipsoid(mi, mf, res, f, r)
    for k in range(NFLBOX):
        inertia_box(mi, mf, res, f, Int(mi[FLI + 2 * NFLROW + k]))
    for i in range(NB - 1, 0, -1):
        var p = Int(mi[PARENT + i])
        if p > 0:
            put6(res, f, OFLW + 6 * p, get6(res, f, OFLW + 6 * p) + get6(res, f, OFLW + 6 * i))
    for v in range(NV):
        res[(OPAS + v) * F + f] += (get6(res, f, OCD + 6 * v) * get6(res, f, OFLW + 6 * Int(mi[DBODY + v]))).reduce_add()


@always_inline
def tendon(mi: TI, mf: TF, qpos: TQ, qvel: TV, f: Int, t: Int) -> Tuple[Float32, Float32]:
    """MuJoCo mj_tendon for a fixed tendon (engine_core_smooth.c): length = sum coef qpos, velocity = sum coef qvel."""
    var length: Float32 = 0
    var vel: Float32 = 0
    var a = Int(mi[TNI + t])
    for w in range(a, a + Int(mi[TNI + NTEN + t])):
        var c = mf[TNF + w]
        length += c * qpos[Int(mi[TNI + 2 * NTEN + NWRAP + w]) * F + f]
        vel += c * qvel[Int(mi[TNI + 2 * NTEN + w]) * F + f]
    return (length, vel)


def tendon_passive(mi: TI, mf: TF, qpos: TQ, qvel: TV, res: TO, f: Int):
    """MuJoCo mj_springdamper tendon part (engine_passive.c:775): spring about [lower, upper] of lengthspring, linear
    damper, along the constant tendon Jacobian (the wrap coefs) into qfrc_passive."""
    for t in range(NTEN):
        var lv = tendon(mi, mf, qpos, qvel, f, t)
        var lo = mf[TNF + NWRAP + 2 * NTEN + 2 * t]
        var hi = mf[TNF + NWRAP + 2 * NTEN + 2 * t + 1]
        var x = lv[0] - hi if lv[0] > hi else (lv[0] - lo if lv[0] < lo else Float32(0))
        var frc = -x * mf[TNF + NWRAP + t] - lv[1] * mf[TNF + NWRAP + NTEN + t]
        var a = Int(mi[TNI + t])
        for w in range(a, a + Int(mi[TNI + NTEN + t])):
            res[(OPAS + Int(mi[TNI + 2 * NTEN + w])) * F + f] += mf[TNF + w] * frc


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
    comptime if NTEN > 0:
        tendon_passive(mi, mf, qpos, qvel, res, f)
    comptime if NFLUID:
        fluid(mi, mf, res, f)
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
def sbase(mi: TI, p: Int) -> Int:
    return Int(mi[CPH + 11 * p + 8])


@always_inline
def nslot(mi: TI, p: Int) -> Int:
    return Int(mi[CPH + 11 * p + 9])


@always_inline
def jbase(mi: TI, p: Int) -> Int:
    return Int(mi[CPH + 11 * p + 10])


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
    mj_makeImpedance and mj_referenceConstraint, per exported plane-mesh pair; mjc_SphereSphere (one contact on the centre line, a two-body Jacobian) per sphere pair. A heightfield ground (HFGEOM) is not
    mjc_ConvexHField (GJK+EPA on cell prisms): the plane is the tangent plane of world.mojo's terrain function under
    the mesh centre, exact on linear terrain, wrong where the terrain bends inside one mesh."""
    var h = mf[TSTEP]
    var impratio = mf[COPT]
    for p in range(NCP):
        var g1 = Int(mi[CPH + 11 * p]); var g2 = Int(mi[CPH + 11 * p + 1]); var b = Int(mi[CPH + 11 * p + 2])
        var nvert = Int(mi[CPH + 11 * p + 3]); var voff = Int(mi[CPH + 11 * p + 4]); var eoff = Int(mi[CPH + 11 * p + 5])
        var nch = Int(mi[CPH + 11 * p + 6]); var choff = Int(mi[CPH + 11 * p + 7])
        var pf = CPF + 14 * p
        var margin = mf[pf]; var gap = mf[pf + 1]; var f0 = mf[pf + 2]; var f1 = mf[pf + 3]
        var b1 = body(res, f, Int(mi[GBODY + g1]))
        var q1 = mul(b1[1], m4(mf, GQUAT + 4 * g1))
        var pos1 = V3(get(res, f, OG + 3 * g1), get(res, f, OG + 3 * g1 + 1), get(res, f, OG + 3 * g1 + 2), 0)
        var b2 = body(res, f, b)
        var q2 = mul(b2[1], m4(mf, GQUAT + 4 * g2))
        var pos2 = V3(get(res, f, OG + 3 * g2), get(res, f, OG + 3 * g2 + 1), get(res, f, OG + 3 * g2 + 2), 0)
        var normal: V3
        var pnt: V3
        var dist: Float32
        var locdir = V3(0)
        var best = 0
        if nvert == 0:  # mjraw_SphereSphere: one contact on the centre line, normal from geom1 to geom2
            var r1 = mf[pf + 13]; var r2 = mf[pf + 12]
            var dif = pos2 - pos1
            var cd2 = (dif * dif).reduce_add()
            if cd2 > (margin + r1 + r2) * (margin + r1 + r2):
                for c in range(nslot(mi, p)):
                    res[(OVALID + sbase(mi, p) + c) * F + f] = 0
                continue
            var len = sqrt(cd2)
            dist = len - r1 - r2
            if len < 1e-15:  # same centre: the cross product of the z axes
                normal = cross(col(q1, 2), col(q2, 2))
                normal = normal / sqrt((normal * normal).reduce_add())
            else:
                normal = dif / len
            pnt = pos1 + normal * (r1 + dist)  # the loop's contact point pnt - dist/2 normal is MuJoCo's pos
        else:
            var o = origin(mi, qpos, f, b)  # the plane (a world geom, world coordinates) in the animal's frame
            pos1 = pos1 - o
            normal = rot(q1, basis(2))
            comptime if HFG >= 0:
                if g1 == HFG:
                    var w = pos2 + o
                    var hg = height_grad(w[0], w[1], terrain(mf))
                    normal = V3(-hg[1], -hg[2], 1, 0) / sqrt(hg[1] * hg[1] + hg[2] * hg[2] + 1)
                    pos1 = V3(pos2[0], pos2[1], hg[0] - o[2], 0)  # pos2's own x, y: no round trip through world coordinates
            if (normal * (pos2 - pos1)).reduce_add() - mf[pf + 12] > margin:  # bounding sphere clear of the plane: the
                for c in range(nslot(mi, p)):                                        # support vertex would be too (no contact)
                    res[(OVALID + sbase(mi, p) + c) * F + f] = 0
                continue
            locdir = rot(Q4(q2[0], -q2[1], -q2[2], -q2[3]), -normal)
            var bdot: Float32 = -3.4e38
            for v in range(nvert):
                var vd = (locdir * m3(mf, CVERT + 3 * (voff + v))).reduce_add()
                if vd > bdot:
                    bdot = vd
                    best = v
            pnt = rot(q2, m3(mf, CVERT + 3 * (voff + best))) + pos2
            dist = (normal * (pnt - pos1)).reduce_add()
        # contact frame (mju_makeFrame from the normal), shared by the pair's contacts
        var y = basis(1) if normal[1] < 0.5 and normal[1] > -0.5 else basis(2)
        y = y - normal * (normal * y).reduce_add()
        y = y / sqrt((y * y).reduce_add())
        var z = cross(normal, y)
        var com = V3(get(res, f, OSC + 3 * Int(mi[ROOT + b])), get(res, f, OSC + 3 * Int(mi[ROOT + b]) + 1), get(res, f, OSC + 3 * Int(mi[ROOT + b]) + 2), 0)
        var first = pnt - normal * (0.5 * dist)
        var threshold = (normal * (pos2 - pos1)).reduce_add() - margin
        var count = 0
        var e = Int(mi[CFIRST + eoff + best])
        while count < nslot(mi, p):
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
            var slot = sbase(mi, p) + count
            res[(OVALID + slot) * F + f] = 1
            # mj_jacDifPair over the chain: jacp = cdof_lin + cdof_ang x (pos - subtree_com[root]), signed (geom2's body
            # minus geom1's; a dof of both cancels exactly); rows to the contact frame
            var off = cpos - com
            for c in range(nch):
                var cd = OCD + 6 * Int(mi[CCHAIN + choff + c])
                var ang = V3(get(res, f, cd), get(res, f, cd + 1), get(res, f, cd + 2), 0)
                var jp = (V3(get(res, f, cd + 3), get(res, f, cd + 4), get(res, f, cd + 5), 0) + cross(ang, off)) * Float32(mi[CSIGN + choff + c])
                var jn = (normal * jp).reduce_add(); var j1 = (y * jp).reduce_add(); var j2 = (z * jp).reduce_add()
                var jadr = OJ + jbase(mi, p) + 4 * count * nch + c
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
                    vel += get(res, f, OJ + jbase(mi, p) + (4 * count + k) * nch + c) * qvel[Int(mi[CCHAIN + choff + c]) * F + f]
                res[(OAREF + row) * F + f] = -B * vel - K * imp * (dist - imargin)
            count += 1
        for c in range(count, nslot(mi, p)):
            res[(OVALID + sbase(mi, p) + c) * F + f] = 0


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
        elif mi[a] == 3:  # fixed tendon: moment = gear x the wrap coefs, length and velocity = gear x the tendon's
            var t = Int(mi[a + 1])
            var lv = tendon(mi, mf, qpos, qvel, f, t)
            var force = mf[af] * c + mf[af + 1] + mf[af + 2] * (lv[0] * gear) + mf[af + 3] * (lv[1] * gear)
            if mi[a + 3] != 0:
                force = min(max(force, mf[af + 6]), mf[af + 7])
            res[(OAF + u) * F + f] = force
            var w0 = Int(mi[TNI + t])
            for w in range(w0, w0 + Int(mi[TNI + NTEN + t])):
                res[(OQACT + Int(mi[TNI + 2 * NTEN + w])) * F + f] += gear * mf[TNF + w] * force
        else:  # body: adhesion over the body's contacts
            var body = Int(mi[a + 1])
            var counter = 0
            for p in range(NCP):
                if Int(mi[CPH + 11 * p + 2]) != body:
                    continue
                for sl in range(nslot(mi, p)):
                    counter += 1 if get(res, f, OVALID + sbase(mi, p) + sl) != 0 else 0
            var force = mf[af] * c + mf[af + 1]  # length and velocity of a body transmission are 0
            if mi[a + 3] != 0:
                force = min(max(force, mf[af + 6]), mf[af + 7])
            res[(OAF + u) * F + f] = force
            if counter == 0:
                continue
            var scale = -force / Float32(counter)
            for p in range(NCP):
                if Int(mi[CPH + 11 * p + 2]) != body:
                    continue
                var nch = Int(mi[CPH + 11 * p + 6]); var choff = Int(mi[CPH + 11 * p + 7])
                for sl in range(nslot(mi, p)):
                    if get(res, f, OVALID + sbase(mi, p) + sl) == 0:
                        continue
                    var jadr = OJ + jbase(mi, p) + 4 * sl * nch
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
        var nch = Int(mi[CPH + 11 * p + 6]); var choff = Int(mi[CPH + 11 * p + 7])
        for sl in range(nslot(mi, p)):
            if get(res, f, OVALID + sbase(mi, p) + sl) == 0:
                continue
            for k in range(4):
                var row = 4 * (sbase(mi, p) + sl) + k
                var jar = get(res, f, OJAR + row)
                var act = jar < 0
                res[(OST + row) * F + f] = 1 if act else 0
                if act:
                    var D = get(res, f, OD + row)
                    cost += 0.5 * D * jar * jar
                    var force = -D * jar
                    var jadr = OJ + jbase(mi, p) + (4 * sl + k) * nch
                    for c in range(nch):
                        var v = Int(mi[CCHAIN + choff + c])
                        res[(OQFC + v) * F + f] += get(res, f, jadr + c) * force
    for v in range(NV):
        cost += 0.5 * (get(res, f, OMA + v) - get(res, f, OQSM + v)) * (get(res, f, OCQACC + v) - get(res, f, OQACCS + v))
    return cost


def hessian(mi: TI, res: TO, f: Int):
    """MuJoCo MakeHessian + FactorizeHessian: H = M + J' D J over active rows, in H's pattern (model.py hpattern: M's
    plus the fill of pairs coupling two limbs; a pair's dof chain is exactly the pattern of its deepest row),
    factored L'DL with mj_factorI."""
    res[(ODIAG + 2) * F + f] += 1
    for c in range(NH):
        res[(OHN + c) * F + f] = 0
    for c in range(NC):
        res[(OHN + Int(mi[MTOH + c])) * F + f] = get(res, f, OM + c)
    sync()
    for p in range(NCP):
        var nch = Int(mi[CPH + 11 * p + 6]); var choff = Int(mi[CPH + 11 * p + 7])
        for sl in range(nslot(mi, p)):
            if get(res, f, OVALID + sbase(mi, p) + sl) == 0:
                continue
            for k in range(4):
                var row = 4 * (sbase(mi, p) + sl) + k
                if get(res, f, OST + row) == 0:
                    continue
                var D = get(res, f, OD + row)
                var jadr = OJ + jbase(mi, p) + (4 * sl + k) * nch
                for a in range(nch):  # lane owns H row d (d % LN): every row the same lane, no race across pairs
                    var d = Int(mi[CCHAIN + choff + a])
                    if d % LN != lane():
                        continue
                    var ja = get(res, f, jadr + a) * D
                    var ra = Int(mi[HADR + d])
                    for bb in range(a + 1):
                        res[(OHN + ra + bb) * F + f] += ja * get(res, f, jadr + bb)
    sync()
    factorI(mi, res, f, OHN, OHNI, HADR)


def lowrank(mi: TI, res: TO, f: Int) -> Bool:
    """Active-set change without refactoring: before update rewrites OST, every row turning active (inactive) is a
    rank-1 update (downdate) H +- D j j' of the factored L'DL in OHN (Gill-Golub-Murray-Saunders C1 in elimination
    order, k from the row's deepest dof back to the root). j lives on the pair's chain, which is the pattern of its
    deepest row, so the walk stays on that row's elimination path: no fill. w in OMV (free until the next search).
    False when a downdate's pivot collapses (float32): the caller rebuilds H from scratch."""
    for p in range(NCP):
        var nch = Int(mi[CPH + 11 * p + 6]); var choff = Int(mi[CPH + 11 * p + 7])
        var deep = 0
        for c in range(nch):
            deep = max(deep, Int(mi[CCHAIN + choff + c]))
        var ds = Int(mi[HADR + deep]); var dn = Int(mi[HADR + NV + deep])
        for sl in range(nslot(mi, p)):
            if get(res, f, OVALID + sbase(mi, p) + sl) == 0:
                continue
            for k in range(4):
                var row = 4 * (sbase(mi, p) + sl) + k
                var now = get(res, f, OJAR + row) < 0
                if now == (get(res, f, OST + row) != 0):
                    continue
                for t in range(dn):
                    res[(OMV + Int(mi[HADR + 2 * NV + ds + t])) * F + f] = 0
                var jadr = OJ + jbase(mi, p) + (4 * sl + k) * nch
                for c in range(nch):
                    res[(OMV + Int(mi[CCHAIN + choff + c])) * F + f] = get(res, f, jadr + c)
                var a = get(res, f, OD + row) if now else -get(res, f, OD + row)
                for t in range(dn - 1, -1, -1):
                    var kk = Int(mi[HADR + 2 * NV + ds + t])
                    var pk = get(res, f, OMV + kk)
                    if pk == 0:
                        continue
                    var start = Int(mi[HADR + kk]); var diag = Int(mi[HADR + NV + kk]) - 1
                    var dk = get(res, f, OHN + start + diag)
                    var dnew = dk + a * pk * pk
                    if dnew <= 1e-4 * dk:
                        return False
                    var beta = a * pk / dnew
                    a = a * dk / dnew
                    res[(OHN + start + diag) * F + f] = dnew
                    res[(OHNI + kk) * F + f] = 1 / dnew
                    for c in range(diag):
                        var i = Int(mi[HADR + 2 * NV + start + c])
                        var wi = get(res, f, OMV + i) - pk * get(res, f, OHN + start + c)
                        res[(OMV + i) * F + f] = wi
                        res[(OHN + start + c) * F + f] = get(res, f, OHN + start + c) + beta * wi
    return True


def gradient(mi: TI, res: TO, f: Int):
    """MuJoCo PrimalUpdateGradient (Newton): grad = M qacc - qfrc_smooth - qfrc_constraint, search = -H^-1 grad."""
    for v in range(NV):
        var g = get(res, f, OMA + v) - get(res, f, OQSM + v) - get(res, f, OQFC + v)
        res[(OGRAD + v) * F + f] = g
        res[(OSRCH + v) * F + f] = g
    solveLD(mi, res, f, OSRCH, OHN, OHNI, HADR)
    for v in range(NV):
        res[(OSRCH + v) * F + f] = -get(res, f, OSRCH + v)


comptime PNT = SIMD[DType.float32, 4]  # line-search point: alpha, cost, deriv0, deriv1


def peval(mi: TI, res: TO, f: Int, gauss: PNT, alpha: Float32) -> PNT:
    """MuJoCo PrimalEval for pyramidal rows: Gauss quadratic plus every row active at jar + alpha jv (valid rows split
    over lanes, partial sums in OPART)."""
    var q0: Float32 = 0; var q1: Float32 = 0; var q2: Float32 = 0
    for a in range(lane(), 4 * Int(get(res, f, OACT)), LN):
        var r = 4 * Int(get(res, f, OACT + 1 + a // 4)) + a % 4
        var jar = get(res, f, OJAR + r); var jv = get(res, f, OJV + r)
        if jar + alpha * jv < 0:
            var D = get(res, f, OD + r)
            q0 += 0.5 * D * jar * jar
            q1 += jv * D * jar
            q2 += 0.5 * jv * D * jv
    comptime if WAVE:
        res[(OPART + lane()) * F + f] = q0
        res[(OPART + LN + lane()) * F + f] = q1
        res[(OPART + 2 * LN + lane()) * F + f] = q2
        sync()
        q0 = 0; q1 = 0; q2 = 0
        for l in range(LN):
            q0 += get(res, f, OPART + l); q1 += get(res, f, OPART + LN + l); q2 += get(res, f, OPART + 2 * LN + l)
        sync()
    q0 += gauss[0]; q1 += gauss[1]; q2 += gauss[2]
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
        var nch = Int(mi[CPH + 11 * p + 6]); var choff = Int(mi[CPH + 11 * p + 7])
        for sl in range(nslot(mi, p)):
            if get(res, f, OVALID + sbase(mi, p) + sl) == 0:
                continue
            for k in range(4):
                res[(OJV + 4 * (sbase(mi, p) + sl) + k) * F + f] = jdot(mi, res, f, OJ + jbase(mi, p) + (4 * sl + k) * nch, nch, choff, OSRCH)
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


def start(mi: TI, res: TO, f: Int, src: Int) -> Float32:
    """Newton's starting point: qacc = rows src.., M qacc, jar on the valid rows, then PrimalUpdateConstraint's cost."""
    for v in range(NV):
        res[(OCQACC + v) * F + f] = get(res, f, src + v)
    mulM(mi, res, f, OMA, OCQACC)
    for p in range(NCP):
        var nch = Int(mi[CPH + 11 * p + 6]); var choff = Int(mi[CPH + 11 * p + 7])
        for sl in range(nslot(mi, p)):
            if get(res, f, OVALID + sbase(mi, p) + sl) == 0:
                continue
            for k in range(4):
                var row = 4 * (sbase(mi, p) + sl) + k
                res[(OJAR + row) * F + f] = jdot(mi, res, f, OJ + jbase(mi, p) + (4 * sl + k) * nch, nch, choff, OCQACC) - get(res, f, OAREF + row)
    return update(mi, res, f)


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
        scale += get(res, f, OM + Int(mi[MADR + v]) + Int(mi[MNNZ + v]) - 1)
    scale = 1 / scale
    res[ONCON * F + f] = get(res, f, OACT)
    var tol = mf[COPT + 1]
    res[ODIAG * F + f] = 0
    res[(ODIAG + 1) * F + f] = 0
    res[(ODIAG + 2) * F + f] = 0
    var cost = start(mi, res, f, OQACCS)
    comptime if K > 0:  # mj warmstart: the last step's qacc unless qacc_smooth costs less
        var cw = start(mi, res, f, OWARM)
        cost = start(mi, res, f, OQACCS) if cw > cost else cw
    hessian(mi, res, f)
    comptime if STOP == 9:
        return
    gradient(mi, res, f)
    comptime if STOP == 8:
        return
    for _ in range(Int(mi[CITER])):
        res[ODIAG * F + f] += 1
        var alpha = search(mi, res, f, scale, tol * mf[COPT + 2], Int(mi[CITER + 1]))
        if alpha == 0:
            break
        for v in range(NV):
            res[(OCQACC + v) * F + f] += alpha * get(res, f, OSRCH + v)
            res[(OMA + v) * F + f] += alpha * get(res, f, OMV + v)
        var changed = False
        for a in range(4 * Int(get(res, f, OACT))):
            var r = 4 * Int(get(res, f, OACT + 1 + a // 4)) + a % 4
            res[(OJAR + r) * F + f] += alpha * get(res, f, OJV + r)
            if (get(res, f, OJAR + r) < 0) != (get(res, f, OST + r) != 0):
                changed = True
        var cheap = True
        if changed:
            comptime if WAVE:  # one sequential walk per animal on lane 0; its verdict shared so all lanes agree
                sync()
                if lane() == 0:
                    res[OPART * F + f] = 1 if lowrank(mi, res, f) else 0
                sync()
                cheap = get(res, f, OPART) != 0
                sync()
            else:
                cheap = lowrank(mi, res, f)
        var oldcost = cost
        cost = update(mi, res, f)
        if changed and not cheap:
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


@always_inline
def ltsolve(mi: TI, res: TO, f: Int, W: Int, jadr: Int, nch: Int, choff: Int):
    """One constraint row's w = L^-T J' (mj_solveLD's first half on M's factor): nonzero on the row's chain only,
    which holds every M ancestor of its dofs, so rows W.. outside the chain are neither read nor written."""
    for c in range(nch):
        res[(W + Int(mi[CCHAIN + choff + c])) * F + f] = get(res, f, jadr + c)
    for c in range(nch - 1, -1, -1):
        var i = Int(mi[CCHAIN + choff + c])
        var start = Int(mi[MADR + i])
        var x_i = get(res, f, W + i)
        for adr in range(start, start + Int(mi[MNNZ + i]) - 1):
            res[(W + Int(mi[MCOL + adr])) * F + f] -= get(res, f, OLD + adr) * x_i


def noslip(mi: TI, mf: TF, res: TO, f: Int):
    """MuJoCo mj_solNoSlip for pyramidal contacts (no dry friction ported): after Newton, projected Gauss-Seidel over
    each contact's opposing pyramid-edge row pairs (their sum, what the normal sees, kept), noslip_iterations sweeps
    or until improvement / (meaninertia nv) < noslip_tolerance; then mj_dualFinish: qfrc_constraint = J' f, qacc =
    qacc_smooth + M^-1 qfrc_constraint. Matrix-free with M = L'DL: A(k, l) = J_k M^-1 J_l' = w_k' D^-1 w_l, w = L^-T J'
    on the row's chain, and the residual's J_k M^-1 J' f = w_k' D^-1 z with z = L^-T J' f kept current."""
    var nit = Int(mi[CITER + 2])
    if nit == 0:
        return
    var imp0: Float32 = 0
    for a in range(4 * Int(get(res, f, OACT))):  # Newton's forces: -D jar on active rows
        var r = 4 * Int(get(res, f, OACT + 1 + a // 4)) + a % 4
        var fr: Float32 = 0
        if get(res, f, OST + r) != 0:
            var D = get(res, f, OD + r)
            fr = -D * get(res, f, OJAR + r)
            imp0 += 0.5 * fr * fr / D
        res[(OFRC + r) * F + f] = fr
    for v in range(NV):
        res[(OZ + v) * F + f] = get(res, f, OQFC + v)
    for i in range(NV - 1, -1, -1):  # z = L^-T J' f
        var start = Int(mi[MADR + i])
        var x_i = get(res, f, OZ + i)
        for adr in range(start, start + Int(mi[MNNZ + i]) - 1):
            res[(OZ + Int(mi[MCOL + adr])) * F + f] -= get(res, f, OLD + adr) * x_i
    for p in range(NCP):  # w, A and J qacc_smooth - aref are fixed during the sweeps: once per step
        var nch = Int(mi[CPH + 11 * p + 6]); var choff = Int(mi[CPH + 11 * p + 7])
        for sl in range(nslot(mi, p)):
            if get(res, f, OVALID + sbase(mi, p) + sl) == 0:
                continue
            for t in range(2):
                var r0 = 4 * (sbase(mi, p) + sl) + 2 * t
                var jo = jbase(mi, p) + (4 * sl + 2 * t) * nch
                ltsolve(mi, res, f, OW, OJ + jo, nch, choff)
                ltsolve(mi, res, f, OW + NV, OJ + jo + nch, nch, choff)
                var a00: Float32 = 0; var a01: Float32 = 0; var a11: Float32 = 0
                for c in range(nch):
                    var i = Int(mi[CCHAIN + choff + c])
                    var w0 = get(res, f, OW + i); var w1 = get(res, f, OW + NV + i); var di = get(res, f, OLDI + i)
                    a00 += w0 * w0 * di; a01 += w0 * w1 * di; a11 += w1 * w1 * di
                    res[(OWC + jo + c) * F + f] = w0
                    res[(OWC + jo + nch + c) * F + f] = w1
                var ac = OAC + 5 * (r0 // 2)
                res[ac * F + f] = max(a00, Float32(1e-10))
                res[(ac + 1) * F + f] = a01
                res[(ac + 2) * F + f] = max(a11, Float32(1e-10))
                res[(ac + 3) * F + f] = jdot(mi, res, f, OJ + jo, nch, choff, OQACCS) - get(res, f, OAREF + r0)
                res[(ac + 4) * F + f] = jdot(mi, res, f, OJ + jo + nch, nch, choff, OQACCS) - get(res, f, OAREF + r0 + 1)
    var scale = 1 / (mf[COPT + 4] * Float32(NV))
    for it in range(nit):
        var improvement: Float32 = imp0 if it == 0 else 0
        for p in range(NCP):
            var nch = Int(mi[CPH + 11 * p + 6]); var choff = Int(mi[CPH + 11 * p + 7])
            for sl in range(nslot(mi, p)):
                if get(res, f, OVALID + sbase(mi, p) + sl) == 0:
                    continue
                for t in range(2):
                    var r0 = 4 * (sbase(mi, p) + sl) + 2 * t
                    var w = OWC + jbase(mi, p) + (4 * sl + 2 * t) * nch
                    var ac = OAC + 5 * (r0 // 2)
                    var a00 = get(res, f, ac); var a01 = get(res, f, ac + 1); var a11 = get(res, f, ac + 2)
                    var res0 = get(res, f, ac + 3); var res1 = get(res, f, ac + 4)
                    for c in range(nch):
                        var i = Int(mi[CCHAIN + choff + c])
                        var zd = get(res, f, OZ + i) * get(res, f, OLDI + i)
                        res0 += get(res, f, w + c) * zd; res1 += get(res, f, w + nch + c) * zd
                    var o0 = get(res, f, OFRC + r0); var o1 = get(res, f, OFRC + r0 + 1)
                    var mid = 0.5 * (o0 + o1)
                    var k1 = a00 + a11 - 2 * a01
                    var k0 = mid * (a00 - a11) + res0 - (a00 * o0 + a01 * o1) - res1 + (a01 * o0 + a11 * o1)
                    var n0 = mid; var n1 = mid
                    if k1 >= 1e-15:
                        var y = -k0 / k1
                        if y < -mid:
                            n0 = 0; n1 = 2 * mid
                        elif y > mid:
                            n0 = 2 * mid; n1 = 0
                        else:
                            n0 = mid + y; n1 = mid - y
                    var d0 = n0 - o0; var d1 = n1 - o1
                    # d1 = -d0 (the pair's sum is kept): the exact form; the general one cancels ulp(f) x |res| ~ 0.2 in float32
                    var change = 0.5 * k1 * d0 * d0 + d0 * (res0 - res1)
                    if change > 1e-10:  # no improvement: keep the old forces
                        continue
                    improvement -= change
                    res[(OFRC + r0) * F + f] = n0
                    res[(OFRC + r0 + 1) * F + f] = n1
                    for c in range(nch):
                        var i = Int(mi[CCHAIN + choff + c])
                        res[(OZ + i) * F + f] += get(res, f, w + c) * d0 + get(res, f, w + nch + c) * d1
        if improvement * scale < mf[COPT + 3]:
            break
    for v in range(NV):  # mj_dualFinish as a change on Newton's result: qfrc_constraint += J' (f - f_newton), qacc +=
        res[(OW + v) * F + f] = 0  # M^-1 of it (qacc_smooth + M^-1 J'f cancels ~140x in float32: qacc_smooth >> qacc)
    for p in range(NCP):
        var nch = Int(mi[CPH + 11 * p + 6]); var choff = Int(mi[CPH + 11 * p + 7])
        for sl in range(nslot(mi, p)):
            if get(res, f, OVALID + sbase(mi, p) + sl) == 0:
                continue
            for k in range(4):
                var row = 4 * (sbase(mi, p) + sl) + k
                var df = get(res, f, OFRC + row)
                if get(res, f, OST + row) != 0:
                    df += get(res, f, OD + row) * get(res, f, OJAR + row)
                var jadr = OJ + jbase(mi, p) + (4 * sl + k) * nch
                for c in range(nch):
                    res[(OW + Int(mi[CCHAIN + choff + c])) * F + f] += get(res, f, jadr + c) * df
    for v in range(NV):
        res[(OQFC + v) * F + f] += get(res, f, OW + v)
    solveLD(mi, res, f, OW, OLD, OLDI)
    for v in range(NV):
        res[(OCQACC + v) * F + f] += get(res, f, OW + v)

def euler(mi: TI, mf: TF, qpos: TQ, qvel: TV, res: TO, f: Int):
    """MuJoCo mj_Euler (or, INTEGRATOR 3, mj_implicitSkip implicitfast) with implicit dof damping (contact-free: qfrc = qfrc_passive - qfrc_bias): qH = M + h diag(damping),
    factor, solve qH qacc = qfrc (mj_solveLD), qvel += h qacc, mj_integratePos. qvel is read, not written back."""
    var h = mf[TSTEP]
    for c in range(NC):
        res[(OQH + c) * F + f] = get(res, f, OM + c)
    for v in range(NV):
        res[(OQH + Int(mi[MADR + v]) + Int(mi[MNNZ + v]) - 1) * F + f] += h * mf[DAMP + v]
    comptime if INTEGRATOR == 3 and STAGE >= 7:
        # mj_implicitSkip (engine_forward.c:1864) implicitfast: qH = M - h qDeriv. The dof damping above is -qDeriv of mjd_passive_vel;
        # mjd_actuator_vel adds moment' bias_vel moment per joint actuator (diagonal: gear^2 biasprm[2]), skipped where forcerange clamps
        for u in range(NU):
            var a = CACT + 5 * u
            if mi[a] != 0:
                continue
            var af = CACF + 9 * u
            var force = get(res, f, OAF + u)
            if mi[a + 3] != 0 and (force <= mf[af + 6] or force >= mf[af + 7]):
                continue
            var dof = Int(mi[a + 1])
            res[(OQH + Int(mi[MADR + dof]) + Int(mi[MNNZ + dof]) - 1) * F + f] -= h * mf[af + 8] * mf[af + 8] * mf[af + 3]
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
    var f = block_idx.x if WAVE else block_idx.x * block_dim.x + thread_idx.x
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
        var na = 0
        for sl in range(NSLOT):
            if get(res, f, OVALID + sl) != 0:
                res[(OACT + 1 + na) * F + f] = Float32(sl)
                na += 1
        res[OACT * F + f] = Float32(na)
        comptime if STAGE >= 7:
            actuation(mi, mf, qpos, qvel, ctrl, res, f)
        comptime if STOP == 5:
            return
        newton(mi, mf, res, f)
        comptime if STOP == 8 or STOP == 9:
            return
        noslip(mi, mf, res, f)
        comptime if K > 0:
            for v in range(NV):
                res[(OWARM + v) * F + f] = get(res, f, OCQACC + v)
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
    o_b.enqueue_fill(0)  # warm start reads OWARM before the first step writes it
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
    ctx.enqueue_function(kernel, TileTensor(mi_b, LI), TileTensor(mf_b, LF), TileTensor(q_b, LQ), TileTensor(v_b, LV), TileTensor(u_b, LU), TileTensor(o_b, LO), grid_dim=GRID, block_dim=BLOCK)
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
                ctx.enqueue_function(kernel, TileTensor(mi_b, LI), TileTensor(mf_b, LF), TileTensor(q_b, LQ), TileTensor(v_b, LV), TileTensor(u_b, LU), TileTensor(o_b, LO), grid_dim=GRID, block_dim=BLOCK)
            ctx.enqueue_copy(state.unsafe_ptr(), q_b)
            ctx.enqueue_copy(state.unsafe_ptr().unsafe_offset(NQ * F), v_b)
            ctx.synchronize()
            with open(dir + "state10.bin", "w") as file:
                file.write_bytes(Span(unsafe_ptr=state.unsafe_ptr().unsafe_bitcast[UInt8](), length=(NQ + NV) * F * 4))
    var t0 = perf_counter_ns()
    for _ in range(100):
        ctx.enqueue_function(kernel, TileTensor(mi_b, LI), TileTensor(mf_b, LF), TileTensor(q_b, LQ), TileTensor(v_b, LV), TileTensor(u_b, LU), TileTensor(o_b, LO), grid_dim=GRID, block_dim=BLOCK)
    ctx.synchronize()
    var us = Float64(perf_counter_ns() - t0) / 100 / 1000
    var host = List[Float32](length=NROW * F, fill=0)
    ctx.enqueue_copy(host.unsafe_ptr(), o_b)
    ctx.synchronize()
    with open(dir + "smooth.bin", "w") as file:
        file.write_bytes(Span(unsafe_ptr=host.unsafe_ptr().unsafe_bitcast[UInt8](), length=NOUT * F * 4))
    print(F, "animals, stage", STAGE, ", K", max(K, 1), ":", us, "us per launch,", Float64(F * max(K, 1)) / us * 1e6, "animal-steps/s")
    comptime if STAGE >= 6:
        var it: Float64 = 0; var ev: Float64 = 0; var hb: Float64 = 0; var slots: Float64 = 0; var itmax: Float32 = 0
        for a in range(F):
            it += Float64(host[ODIAG * F + a]); ev += Float64(host[(ODIAG + 1) * F + a]); hb += Float64(host[(ODIAG + 2) * F + a]); slots += Float64(host[ONCON * F + a]); itmax = max(itmax, host[ODIAG * F + a])
        print("Newton iterations per animal", it / Float64(F), "max", itmax, "; line-search evaluations", ev / Float64(F))
        print("Hessian rebuilds per animal", hb / Float64(F), "; active slots", slots / Float64(F))
