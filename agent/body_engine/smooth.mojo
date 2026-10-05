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
from convex import Sp, collide, collide_hfield, Height, ELLIPSOID, MESH, BOX, T as CT, S as CS, V as CVEC  # convex.mojo's scalar type (float64 with -DCCD64=1, which the export sets)

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
comptime DUMP = get_defined_int["DUMP", 0]() == 1  # diagnostics (row_diag.py): write every scratch row and their offsets
comptime STOP = get_defined_int["STOP", 99]()
comptime INTEGRATOR = get_defined_int["INTEGRATOR", 0]()  # opt.integrator: 0 Euler, 1 RK4, 3 implicitfast (model.py export; absent = Euler)
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
# model.py fluid() then tendons(), appended after the actuators: fluid ints per ellipsoid row (body, geom) then the inertia-box bodies
comptime NFLROW = get_defined_int["NFLROW", -1]()  # -1: no fluid block (walking models: the code below folds out)
comptime NFLBOX = get_defined_int["NFLBOX", 0]()
comptime NFLUID = NFLROW >= 0
comptime FLI = NI1
comptime NTEN = get_defined_int["NTEN", 0]()  # fixed tendons; tendon ints: wrap start and count per tendon, then each wrap's dof, then its qpos address
comptime NWRAP = get_defined_int["NWRAP", 0]()
comptime TNI = FLI + (2 * NFLROW + NFLBOX if NFLUID else 0)
comptime NCV = get_defined_int["NCV", 0]()  # convex pairs (box or ellipsoid vs mesh, convex.mojo): per pair its kind, between the tendon and equality blocks
comptime NI2 = TNI + 2 * NTEN + 2 * NWRAP
# equality constraints (model.py eq_block, last block): per constraint 12 ints, then every chain, then its masks
comptime NEQ = get_defined_int["NEQ", 0]()
comptime NEQR = get_defined_int["NEQR", 0]()  # rows: connect 3, weld 6, joint 1
comptime NEQCH = get_defined_int["NEQCH", 0]()
comptime NEQJ = get_defined_int["NEQJ", 0]()  # Jacobian entries: rows x chain length, summed
comptime EQH = 12
comptime EQF = 31
comptime CEQH = NI2 + (NCP if NCV > 0 else 0)  # the convex block (kinds per pair) sits between the tendon and equality blocks
comptime CEQC = CEQH + EQH * NEQ
comptime CEQM = CEQC + NEQCH
# mocap bodies (model.py mocap_block, after the equality block): body_mocapid per body; each animal's mocap pose (pos 3,
# quat 4 per mocap body) rides in rows NU.. of ctrl.bin
comptime NMOCAP = get_defined_int["NMOCAP", 0]()
comptime NUC = NU + 7 * NMOCAP
comptime CMOC = CEQM + NEQCH
# gravcomp (model.py gravcomp_block, last): ints jnt_actgravcomp per joint, floats body_gravcomp per body
comptime GRAVCOMP = get_defined_int["GRAVCOMP", 0]() == 1
comptime CGC = CMOC + (NB if NMOCAP > 0 else 0)
comptime NI = CGC + (NJ if GRAVCOMP else 0)
comptime CKIND = NI2
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
comptime NFL2 = TNF + NWRAP + 4 * NTEN
comptime CCV = NFL2  # NCV: opt.ccd_tolerance, heightfield size 0-3 and origin z (HFEXACT), then per pair geom1's size (3)
comptime CEQF = NFL2 + (6 + 3 * NCP if NCV > 0 else 0)  # per equality: anchors 0 and 1, quaternions 0 and 1, torquescale, solref, solimp, invweight 2, joint data 5, qpos0 2
comptime CGF = CEQF + EQF * NEQ
comptime NFL = CGF + (NB if GRAVCOMP else 0)
comptime HFNR = get_defined_int["HFNR", 2]()  # HFEXACT: grid rows and columns of the MuJoCo hfield (the engine's grid is virtual and endless, same step)
comptime HFNC = get_defined_int["HFNC", 2]()
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
comptime NPOOL = get_defined_int["NPOOL", 0]()  # NCV: slots shared by the convex and exact-heightfield pairs of one animal (the last NPOOL of NSLOT); each such pair takes what it needs per step
comptime JPOOL = get_defined_int["JPOOL", 0]()  # their Jacobian entries (the last JPOOL of NJE), bump-allocated per step
comptime NSTATIC = NSLOT - NPOOL
comptime JSTATIC = NJE - JPOOL
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
comptime OPA = OWARM + NV  # active pairs (some valid slot) of this step: count, then ids ascending: the solver loops visit these only
comptime OSD = OPA + 1 + NCP  # NCV: per pair this step's slot base, slot count and Jacobian base (pooled pairs only)
comptime OPP = OSD + 3 * NCP  # NCV: pool slots and pool Jacobian entries taken so far this step
comptime NROW1 = OPP + 2 if STAGE >= 6 else (OVALID if STAGE >= 4 else NOUT)
comptime OEQD = NROW1  # equality rows: D, aref, jar, jv, then the Jacobian entries (like OJ)
comptime OEQA = OEQD + NEQR
comptime OEQJAR = OEQA + NEQR
comptime OEQJV = OEQJAR + NEQR
comptime OEQJ = OEQJV + NEQR
comptime OGC = OEQJ + NEQJ  # qfrc_gravcomp (GRAVCOMP)
comptime NROW0 = OGC + (NV if GRAVCOMP else 0)
comptime ORKQ = NROW0  # RK4 scratch (INTEGRATOR 1 only): X0 qpos, then per stage velocity v_s and acceleration F_s, then one scaled velocity
comptime ORKV = ORKQ + NQ
comptime ORKF = ORKV + 4 * NV
comptime ORKW = ORKF + 4 * NV
comptime NROW = ORKW + NV if INTEGRATOR == 1 else NROW0
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
comptime LU = row_major[max(NUC, 1) * F]()
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


def gravcomp(mi: TI, mf: TF, res: TO, f: Int):
    """MuJoCo mj_gravcomp (engine_passive.c:817) and its hand-off (:1012): per body with body_gravcomp the force
    -gravity mass gravcomp at xipos through mj_applyFT's point Jacobian (cdof_lin + cdof_ang x (xipos - subtree_com of the
    root), over the body's own and its ancestors' dofs) into qfrc_gravcomp (rows OGC); the dofs of joints without
    actgravcomp are added to qfrc_passive, the others wait for actuation() (engine_forward.c:770)."""
    for v in range(NV):
        res[(OGC + v) * F + f] = 0
    for i in range(1, NB):
        var gc = mf[CGF + i]
        if gc == 0:
            continue
        var force = m3(mf, GRAV) * (-(mf[MASS + i] * gc))
        var r = 3 * Int(mi[ROOT + i])
        var off = V3(get(res, f, OXI + 3 * (i - 1)), get(res, f, OXI + 3 * (i - 1) + 1), get(res, f, OXI + 3 * (i - 1) + 2), 0) - V3(get(res, f, OSC + r), get(res, f, OSC + r + 1), get(res, f, OSC + r + 2), 0)
        var b = i
        while b > 0:
            for d in range(Int(mi[BDOFADR + b]), Int(mi[BDOFADR + b]) + Int(mi[BDOFNUM + b])):
                var cd = OCD + 6 * d
                var ang = V3(get(res, f, cd), get(res, f, cd + 1), get(res, f, cd + 2), 0)
                var lin = V3(get(res, f, cd + 3), get(res, f, cd + 4), get(res, f, cd + 5), 0)
                res[(OGC + d) * F + f] += ((lin + cross(ang, off)) * force).reduce_add()
            b = Int(mi[PARENT + b])
    for j in range(NJ):
        if mi[CGC + j] != 0:
            continue
        var t = Int(mi[JTYPE + j])
        var da = Int(mi[DOFADR + j])
        for d in range(da, da + (6 if t == 0 else (3 if t == 1 else 1))):
            res[(OPAS + d) * F + f] += get(res, f, OGC + d)


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
    comptime if GRAVCOMP:
        gravcomp(mi, mf, res, f)
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
def jdot(mi: TI, res: TO, f: Int, jadr: Int, nch: Int, choff: Int, X: Int, ch: Int = CCHAIN) -> Float32:
    """Sparse Jacobian row (chain order) dotted with the dof vector in rows X.."""
    var acc: Float32 = 0
    for c in range(nch):
        acc += get(res, f, jadr + c) * get(res, f, X + Int(mi[ch + choff + c]))
    return acc


@always_inline
def sbase(mi: TI, res: TO, f: Int, p: Int) -> Int:
    comptime if NCV > 0:
        if Int(mi[CKIND + p]) > 0:  # pooled pair: this step's allocation
            return Int(get(res, f, OSD + 3 * p))
    return Int(mi[CPH + 11 * p + 8])


@always_inline
def nslot(mi: TI, res: TO, f: Int, p: Int) -> Int:
    comptime if NCV > 0:
        if Int(mi[CKIND + p]) > 0:
            return Int(get(res, f, OSD + 3 * p + 1))
    return Int(mi[CPH + 11 * p + 9])


@always_inline
def jbase(mi: TI, res: TO, f: Int, p: Int) -> Int:
    comptime if NCV > 0:
        if Int(mi[CKIND + p]) > 0:
            return Int(get(res, f, OSD + 3 * p + 2))
    return Int(mi[CPH + 11 * p + 10])


@always_inline
def powi(a: Float32, b: Float32) -> Float32:
    """MuJoCo power() for a whole exponent (export refuses others): amdgpu links no powf here."""
    var r: Float32 = 1
    for _ in range(Int(b)):
        r *= a
    return r


@always_inline
def impedance(s0: Float32, s1: Float32, s2: Float32, s3: Float32, s4: Float32, pos: Float32) -> Float32:
    """MuJoCo getimpedance (engine_core_constraint.c:2017), solimp already clamped, pos = violation minus margin."""
    if s0 == s1 or s2 <= 1e-15:
        return 0.5 * (s0 + s1)
    var x = abs(pos / s2)
    if x >= 1 or x <= 0:
        return s1 if x >= 1 else s0
    var yy: Float32
    if s4 == 1:
        yy = x
    elif x <= s3:
        yy = powi(x, s4) / powi(s3, s4 - 1)
    else:
        yy = 1 - powi(1 - x, s4) / powi(1 - s3, s4 - 1)
    return s0 + yy * (s1 - s0)


@always_inline
def terrain(mf: TF) -> PW:
    var p = PW()
    for k in range(NP):
        p[k] = mf[HFO + k]
    return p


@always_inline
def emit_contact(mi: TI, mf: TF, qvel: TV, res: TO, f: Int, p: Int, count: Int, cpos: V3, dist: Float32, normal: V3, y: V3, z: V3, com: V3):
    """The rest of mj_instantiateContact for contact `count` of pair p at cpos with the frame (normal, y, z): the pyramidal
    Jacobian rows over the pair's dof chain, getsolparam / getimpedance / makeImpedance, referenceConstraint."""
    var h = mf[TSTEP]
    var impratio = mf[COPT]
    var nch = Int(mi[CPH + 11 * p + 6]); var choff = Int(mi[CPH + 11 * p + 7])
    var pf = CPF + 14 * p
    var margin = mf[pf]; var gap = mf[pf + 1]; var f0 = mf[pf + 2]; var f1 = mf[pf + 3]
    var slot = sbase(mi, res, f, p) + count
    # mj_jacDifPair over the chain: jacp = cdof_lin + cdof_ang x (pos - subtree_com[root]), signed (geom2's body
    # minus geom1's; a dof of both cancels exactly); rows to the contact frame
    var off = cpos - com
    for c in range(nch):
        var cd = OCD + 6 * Int(mi[CCHAIN + choff + c])
        var ang = V3(get(res, f, cd), get(res, f, cd + 1), get(res, f, cd + 2), 0)
        var jp = (V3(get(res, f, cd + 3), get(res, f, cd + 4), get(res, f, cd + 5), 0) + cross(ang, off)) * Float32(mi[CSIGN + choff + c])
        var jn = (normal * jp).reduce_add(); var j1 = (y * jp).reduce_add(); var j2 = (z * jp).reduce_add()
        var jadr = OJ + jbase(mi, res, f, p) + 4 * count * nch + c
        res[jadr * F + f] = jn + f0 * j1
        res[(jadr + nch) * F + f] = jn - f0 * j1
        res[(jadr + 2 * nch) * F + f] = jn + f1 * j2
        res[(jadr + 3 * nch) * F + f] = jn - f1 * j2
    # getsolparam, getimpedance, makeImpedance (pyramidal), referenceConstraint
    var ref0 = max(mf[pf + 4], 2 * h); var ref1 = mf[pf + 5]
    var s0 = min(Float32(0.9999), max(Float32(1e-4), mf[pf + 6])); var s1 = min(Float32(0.9999), max(Float32(1e-4), mf[pf + 7]))
    var s2 = max(Float32(0), mf[pf + 8]); var s3 = min(Float32(0.9999), max(Float32(1e-4), mf[pf + 9])); var s4 = max(Float32(1), mf[pf + 10])
    var imargin = margin - gap
    var imp = impedance(s0, s1, s2, s3, s4, dist - imargin)
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
            vel += get(res, f, OJ + jbase(mi, res, f, p) + (4 * count + k) * nch + c) * qvel[Int(mi[CCHAIN + choff + c]) * F + f]
        res[(OAREF + row) * F + f] = -B * vel - K * imp * (dist - imargin)



def rows3(q: Q4) -> Tuple[V3, V3, V3]:
    """Rotation matrix rows of the quaternion (convex.mojo's Shape takes rows)."""
    var c0 = col(q, 0)
    var c1 = col(q, 1)
    var c2 = col(q, 2)
    return (V3(c0[0], c1[0], c2[0], 0), V3(c0[1], c1[1], c2[1], 0), V3(c0[2], c1[2], c2[2], 0))


def pool_alloc(mi: TI, res: TO, f: Int, p: Int, want: Int) -> Int:
    """Take up to `want` contact slots (and their Jacobian entries) of the animal's pool for pooled pair p this step; returns
    how many it got (fewer only when the pool is exhausted: those contacts are dropped, the gate's ncon shows it)."""
    var nch = Int(mi[CPH + 11 * p + 6])
    var used = Int(get(res, f, OPP))
    var jused = Int(get(res, f, OPP + 1))
    var n = want
    while n > 0 and (used + n > NPOOL or jused + 4 * n * nch > JPOOL):
        n -= 1
    res[(OSD + 3 * p) * F + f] = Float32(NSTATIC + used)
    res[(OSD + 3 * p + 1) * F + f] = Float32(n)
    res[(OSD + 3 * p + 2) * F + f] = Float32(JSTATIC + jused)
    res[OPP * F + f] = Float32(used + n)
    res[(OPP + 1) * F + f] = Float32(jused + 4 * n * nch)
    return n


struct TerrainH(Height):
    """The world's terrain function sampled at the virtual grid's nodes (x = dx c - size0, y = dx r - size1; the hfield
    geom sits at world (0, 0, lo)): the node heights MuJoCo's hfield holds, from world.mojo's function instead of a table."""
    var params: PW
    var lo: Float32
    var dx: Float32
    var size0: Float32
    var size1: Float32

    def __init__(out self, params: PW, lo: Float32, dx: Float32, size0: Float32, size1: Float32):
        self.params = params
        self.lo = lo
        self.dx = dx
        self.size0 = size0
        self.size1 = size1

    def h(self, r: Int, c: Int) -> CS:
        return (height_grad(self.dx * Float32(c) - self.size0, self.dx * Float32(r) - self.size1, self.params)[0] - self.lo).cast[CT]()


def hfield_pair(mi: TI, mf: TF, qpos: TQ, qvel: TV, res: TO, f: Int, p: Int, q2: Q4, pos2: V3, b: Int):
    """Port of mjc_ConvexHField (engine_collision_convex.c:1120), in convex.mojo: one GJK + EPA per cell-triangle prism under
    the mesh, on world.mojo's terrain function at the grid nodes (the grid is MuJoCo's, extended without limit). The mesh
    pose goes to the hfield frame in float64 (animal frame + floating origin, minus the hfield's origin), the contacts
    come back to the animal frame. At most nslot contacts, in prism order; MuJoCo keeps up to 50 (check.sh terrain_exact reports it)."""
    var g2 = Int(mi[CPH + 11 * p + 1])
    var voff = Int(mi[CPH + 11 * p + 4]); var eoff = Int(mi[CPH + 11 * p + 5])
    var pf = CPF + 14 * p
    var margin = mf[pf] + mf[pf + 1]
    var rb2 = mf[pf + 12]
    var o = origin(mi, qpos, f, b)
    var size0 = mf[CCV + 1]; var size1 = mf[CCV + 2]; var size2 = mf[CCV + 3]; var size3 = mf[CCV + 4]; var lo = mf[CCV + 5]
    var dx = 2 * size0 / Float32(HFNC - 1)
    var mesh = Sp(mf.ptr.unsafe_offset(CVERT + 3 * voff), mi.ptr.unsafe_offset(CFIRST + eoff), mi.ptr.unsafe_offset(CEDGE))
    mesh.kind = MESH
    var org = (o.cast[CT]() + CVEC(0, 0, -lo.cast[CT]()[0], 0))  # animal frame -> hfield frame
    mesh.pos = pos2.cast[CT]() + org
    var c = rows3(q2)
    mesh.r0 = c[0].cast[CT](); mesh.r1 = c[1].cast[CT](); mesh.r2 = c[2].cast[CT]()
    var cs = collide_hfield(TerrainH(terrain(mf), lo, dx, size0, size1), mesh, HFNR, HFNC, size0.cast[CT]()[0], size1.cast[CT]()[0], size2.cast[CT]()[0],
                            size3.cast[CT]()[0], margin.cast[CT]()[0], mf[CCV].cast[CT]()[0], rb2.cast[CT]()[0], True, CVEC(0))
    var com = V3(get(res, f, OSC + 3 * Int(mi[ROOT + b])), get(res, f, OSC + 3 * Int(mi[ROOT + b]) + 1), get(res, f, OSC + 3 * Int(mi[ROOT + b]) + 2), 0)
    var count = 0
    var take = pool_alloc(mi, res, f, p, min(cs.n, Int(mi[CPH + 11 * p + 9])))
    while count < take:
        var normal = cs.normal[count].cast[DType.float32]()
        var y = basis(1) if normal[1] < 0.5 and normal[1] > -0.5 else basis(2)  # mju_makeFrame
        y = y - normal * (normal * y).reduce_add()
        y = y / sqrt((y * y).reduce_add())
        var z = cross(normal, y)
        res[(OVALID + sbase(mi, res, f, p) + count) * F + f] = 1
        emit_contact(mi, mf, qvel, res, f, p, count, (cs.pos[count] - org).cast[DType.float32](), cs.dist[count].cast[DType.float32](), normal, y, z, com)
        count += 1


def convex_pair(mi: TI, mf: TF, qpos: TQ, qvel: TV, res: TO, f: Int, p: Int, kind: Int, q1: Q4, pos1_: V3, q2: Q4, pos2: V3, b: Int):
    """Port of mjc_Convex for a box-mesh (kind 2, up to 5 contacts) or ellipsoid-mesh (kind 1, one) pair, ported in convex.mojo (GJK +
    EPA, perturbation search); the contacts take the shared mj_instantiateContact tail with their own frame. A world geom1
    is moved into the animal's frame. The bounding spheres skip pairs that cannot be within the margin (MuJoCo's pair
    loop applies the same bound); the box is tested as its exact distance to the mesh's bounding sphere."""
    var g1 = Int(mi[CPH + 11 * p]); var g2 = Int(mi[CPH + 11 * p + 1])
    var nvert = Int(mi[CPH + 11 * p + 3]); var voff = Int(mi[CPH + 11 * p + 4]); var eoff = Int(mi[CPH + 11 * p + 5])
    var pf = CPF + 14 * p
    var margin = mf[pf] + mf[pf + 1]
    var rb2 = mf[pf + 12]; var rb1 = mf[pf + 13]
    var pos1 = pos1_
    if Int(mi[GBODY + g1]) == 0:
        pos1 = pos1 - origin(mi, qpos, f, b)
    var far = False
    var d12 = pos2 - pos1
    if kind == 1:
        far = sqrt((d12 * d12).reduce_add()) > rb1 + rb2 + margin
    else:  # distance from the mesh centre to the box (in the box frame), against the mesh's bounding radius
        var r1 = rows3(q1)
        var sz = m3(mf, CCV + 6 + 3 * p)
        var l = V3((r1[0] * d12).reduce_add(), (r1[1] * d12).reduce_add(), (r1[2] * d12).reduce_add(), 0)
        var e = V3(max(abs(l[0]) - sz[0], 0), max(abs(l[1]) - sz[1], 0), max(abs(l[2]) - sz[2], 0), 0)
        far = sqrt((e * e).reduce_add()) > rb2 + margin
    var count = 0
    res[(OSD + 3 * p + 1) * F + f] = 0  # no slots unless the pair takes some below
    if not far:
        var s1 = Sp(mf.ptr.unsafe_offset(CVERT + 3 * voff), mi.ptr.unsafe_offset(CFIRST + eoff), mi.ptr.unsafe_offset(CEDGE))
        var s2 = Sp(mf.ptr.unsafe_offset(CVERT + 3 * voff), mi.ptr.unsafe_offset(CFIRST + eoff), mi.ptr.unsafe_offset(CEDGE))
        s1.kind = BOX if kind == 2 else ELLIPSOID
        s2.kind = MESH
        s1.size = m3(mf, CCV + 6 + 3 * p).cast[CT]()
        s1.pos = pos1.cast[CT]()
        var a = rows3(q1)
        s1.r0 = a[0].cast[CT](); s1.r1 = a[1].cast[CT](); s1.r2 = a[2].cast[CT]()
        s2.pos = pos2.cast[CT]()
        var c = rows3(q2)
        s2.r0 = c[0].cast[CT](); s2.r1 = c[1].cast[CT](); s2.r2 = c[2].cast[CT]()
        var cs = collide(s1, s2, margin.cast[CT](), mf[CCV].cast[CT](), min(rb1, rb2).cast[CT]())
        var com = V3(get(res, f, OSC + 3 * Int(mi[ROOT + b])), get(res, f, OSC + 3 * Int(mi[ROOT + b]) + 1), get(res, f, OSC + 3 * Int(mi[ROOT + b]) + 2), 0)
        var take = pool_alloc(mi, res, f, p, min(cs.n, Int(mi[CPH + 11 * p + 9])))
        while count < take:
            var normal = cs.normal[count].cast[DType.float32]()
            var y = basis(1) if normal[1] < 0.5 and normal[1] > -0.5 else basis(2)  # mju_makeFrame
            y = y - normal * (normal * y).reduce_add()
            y = y / sqrt((y * y).reduce_add())
            var z = cross(normal, y)
            res[(OVALID + sbase(mi, res, f, p) + count) * F + f] = 1
            emit_contact(mi, mf, qvel, res, f, p, count, cs.pos[count].cast[DType.float32](), cs.dist[count].cast[DType.float32](), normal, y, z, com)
            count += 1


def contacts(mi: TI, mf: TF, qpos: TQ, qvel: TV, res: TO, f: Int):
    """MuJoCo mjc_PlaneConvex (support vertex + hull neighbours below the margin, at most 3, none within 0.3 rbound of
    the first), mj_instantiateContact (pyramidal condim 3, Jacobian over the body's dof chain), mj_diagApprox,
    mj_makeImpedance and mj_referenceConstraint, per exported plane-mesh pair; mjc_SphereSphere (one contact on the centre line, a two-body Jacobian) per sphere pair. A heightfield ground (HFGEOM) is not
    mjc_ConvexHField (GJK+EPA on cell prisms): the plane is the tangent plane of world.mojo's terrain function under
    the mesh centre, exact on linear terrain, wrong where the terrain bends inside one mesh."""
    comptime if NCV > 0:  # the pool of the convex and exact-heightfield pairs starts empty each step
        res[OPP * F + f] = 0
        res[(OPP + 1) * F + f] = 0
        for sp in range(NPOOL):
            res[(OVALID + NSTATIC + sp) * F + f] = 0
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
        comptime if NCV > 0:
            var kind = Int(mi[CKIND + p])
            if kind == 3:
                hfield_pair(mi, mf, qpos, qvel, res, f, p, q2, pos2, b)
                continue
            if kind > 0:
                convex_pair(mi, mf, qpos, qvel, res, f, p, kind, q1, pos1, q2, pos2, b)
                continue
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
                for c in range(nslot(mi, res, f, p)):
                    res[(OVALID + sbase(mi, res, f, p) + c) * F + f] = 0
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
                for c in range(nslot(mi, res, f, p)):                                        # support vertex would be too (no contact)
                    res[(OVALID + sbase(mi, res, f, p) + c) * F + f] = 0
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
        while count < nslot(mi, res, f, p):
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
            var slot = sbase(mi, res, f, p) + count
            res[(OVALID + slot) * F + f] = 1
            emit_contact(mi, mf, qvel, res, f, p, count, cpos, dist, normal, y, z, com)
            count += 1
        for c in range(count, nslot(mi, res, f, p)):
            res[(OVALID + sbase(mi, res, f, p) + c) * F + f] = 0


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
            for ip in range(Int(get(res, f, OPA))):
                var p = Int(get(res, f, OPA + 1 + ip))
                if Int(mi[CPH + 11 * p + 2]) != body:
                    continue
                for sl in range(nslot(mi, res, f, p)):
                    counter += 1 if get(res, f, OVALID + sbase(mi, res, f, p) + sl) != 0 else 0
            var force = mf[af] * c + mf[af + 1]  # length and velocity of a body transmission are 0
            if mi[a + 3] != 0:
                force = min(max(force, mf[af + 6]), mf[af + 7])
            res[(OAF + u) * F + f] = force
            if counter == 0:
                continue
            var scale = -force / Float32(counter)
            for ip in range(Int(get(res, f, OPA))):
                var p = Int(get(res, f, OPA + 1 + ip))
                if Int(mi[CPH + 11 * p + 2]) != body:
                    continue
                var nch = Int(mi[CPH + 11 * p + 6]); var choff = Int(mi[CPH + 11 * p + 7])
                for sl in range(nslot(mi, res, f, p)):
                    if get(res, f, OVALID + sbase(mi, res, f, p) + sl) == 0:
                        continue
                    var jadr = OJ + jbase(mi, res, f, p) + 4 * sl * nch
                    for k in range(nch):  # normal row = mean of the pyramid's first pair
                        var jn = 0.5 * (get(res, f, jadr + k) + get(res, f, jadr + nch + k))
                        res[(OQACT + Int(mi[CCHAIN + choff + k])) * F + f] += scale * jn

    comptime if GRAVCOMP and NU > 0:  # actuator-level gravcomp (engine_forward.c:770), only where mj_fwdActuation runs (nu > 0)
        for j in range(NJ):
            if mi[CGC + j] == 0:
                continue
            var t = Int(mi[JTYPE + j])
            var da = Int(mi[DOFADR + j])
            for d in range(da, da + (6 if t == 0 else (3 if t == 1 else 1))):
                res[(OQACT + d) * F + f] += get(res, f, OGC + d)


def equality(mi: TI, mf: TF, qpos: TQ, qvel: TV, res: TO, f: Int):
    """MuJoCo mj_instantiateEquality (engine_core_constraint.c:513) for connect, weld and joint constraints, mj_diagApprox
    (:1637), mj_makeImpedance (:2069) and mj_referenceConstraint (:3048) with its mj_Jdotv correction (:973): per constraint
    the rows' residual, Jacobian over the constraint's chain (mj_jacDifPair, anchors 0 minus 1, weld rotation corrected
    by 0.5 neg(q1) J q0 relpose), D and aref. Positions are relative to the floating origin of the animal (a body of the
    world tree is shifted into it); the impedance of a connect or weld is a function of the norm of its residual."""
    var h = mf[TSTEP]
    for e in range(NEQ):
        var eh = CEQH + EQH * e
        var typ = Int(mi[eh]); var b0 = Int(mi[eh + 1]); var b1 = Int(mi[eh + 2]); var size = Int(mi[eh + 3])
        var nch = Int(mi[eh + 4]); var choff = Int(mi[eh + 5]); var rbase = Int(mi[eh + 6]); var jb = OEQJ + Int(mi[eh + 7])
        var ef = CEQF + EQF * e
        var cpos = S6(0)
        var jdv = S6(0)  # Jdot qvel of the rows, subtracted from aref
        var posn: Float32
        if typ == 2:  # joint: q0 - qpos0 - c0 - poly(q1 - qpos1_0) (engine_core_constraint.c:808)
            var d0 = Int(mi[eh + 8]); var d1 = Int(mi[eh + 9])
            cpos[0] = qpos[Int(mi[eh + 10]) * F + f] - mf[ef + 29] - mf[ef + 24]
            var deriv: Float32 = 0
            if d1 >= 0:
                var dif = qpos[Int(mi[eh + 11]) * F + f] - mf[ef + 30]
                cpos[0] -= mf[ef + 25] * dif + mf[ef + 26] * dif * dif + mf[ef + 27] * dif * dif * dif + mf[ef + 28] * dif * dif * dif * dif
                deriv = mf[ef + 25] + 2 * mf[ef + 26] * dif + 3 * mf[ef + 27] * dif * dif + 4 * mf[ef + 28] * dif * dif * dif
            for c in range(nch):
                var d = Int(mi[CEQC + choff + c])
                res[(jb + c) * F + f] = (Float32(1) if d == d0 else Float32(0)) - (deriv if d == d1 else Float32(0))
            posn = cpos[0]
        else:
            var weld = typ == 1
            var o0 = origin(mi, qpos, f, b0); var o1 = origin(mi, qpos, f, b1)
            var frame = origin(mi, qpos, f, Int(mi[eh + 8]))
            var sh0 = o0 - frame; var sh1 = o1 - frame
            var B0 = body(res, f, b0); var B1 = body(res, f, b1)
            var p0 = rot(B0[1], m3(mf, ef)) + B0[0] + sh0
            var p1 = rot(B1[1], m3(mf, ef + 3)) + B1[0] + sh1
            var r0 = 3 * Int(mi[ROOT + b0]); var r1 = 3 * Int(mi[ROOT + b1])
            var off0 = p0 - V3(get(res, f, OSC + r0), get(res, f, OSC + r0 + 1), get(res, f, OSC + r0 + 2), 0) - sh0
            var off1 = p1 - V3(get(res, f, OSC + r1), get(res, f, OSC + r1 + 1), get(res, f, OSC + r1 + 2), 0) - sh1
            var d01 = p0 - p1
            for k in range(3):
                cpos[k] = d01[k]
            var qa = mul(B0[1], m4(mf, ef + 6)); var qb = mul(B1[1], m4(mf, ef + 10))
            var nq1 = Q4(qb[0], -qb[1], -qb[2], -qb[3])
            var ts = mf[ef + 14]
            if weld:  # neg(q1) q0 relpose, axis components scaled by torquescale
                var qq = mul(nq1, qa)
                for k in range(3):
                    cpos[3 + k] = qq[1 + k] * ts
            var cv0 = get6(res, f, OCV + 6 * (b0 - 1)) if b0 > 0 else S6(0)
            var cv1 = get6(res, f, OCV + 6 * (b1 - 1)) if b1 > 0 else S6(0)
            var om0 = V3(cv0[0], cv0[1], cv0[2], 0); var om1 = V3(cv1[0], cv1[1], cv1[2], 0)
            # point velocity (mju_transformSpatial from the root's com to the anchor)
            var pv0 = V3(cv0[3], cv0[4], cv0[5], 0) - cross(off0, om0)
            var pv1 = V3(cv1[3], cv1[4], cv1[5], 0) - cross(off1, om1)
            var jd0 = V3(0); var jd1 = V3(0); var jr0 = V3(0); var jr1 = V3(0)
            for c in range(nch):
                var d = Int(mi[CEQC + choff + c]); var mk = Int(mi[CEQM + choff + c])
                var cd = OCD + 6 * d
                var ang = V3(get(res, f, cd), get(res, f, cd + 1), get(res, f, cd + 2), 0)
                var lin = V3(get(res, f, cd + 3), get(res, f, cd + 4), get(res, f, cd + 5), 0)
                var jp = V3(0); var jr = V3(0)
                var qv = qvel[d * F + f]
                var cdot = get6(res, f, OCDD + 6 * d)
                if mk & 4 != 0:  # mj_jacDot: a quaternion dof takes its body's final velocity
                    cdot = cross_motion(get6(res, f, OCV + 6 * (Int(mi[DBODY + d]) - 1)), get6(res, f, cd))
                var cda = V3(cdot[0], cdot[1], cdot[2], 0); var cdl = V3(cdot[3], cdot[4], cdot[5], 0)
                if mk & 1 != 0:
                    jp += lin + cross(ang, off0)
                    jr += ang
                    jd0 += (cdl + cross(cda, off0) + cross(ang, pv0)) * qv
                    jr0 += cda * qv
                if mk & 2 != 0:
                    jp -= lin + cross(ang, off1)
                    jr -= ang
                    jd1 += (cdl + cross(cda, off1) + cross(ang, pv1)) * qv
                    jr1 += cda * qv
                for k in range(3):
                    res[(jb + k * nch + c) * F + f] = jp[k]
                if weld:  # 0.5 neg(q1) (J0 - J1) q0 relpose, axis components
                    var q3 = mul(mul(nq1, Q4(0, jr[0], jr[1], jr[2])), qa)
                    for k in range(3):
                        res[(jb + (3 + k) * nch + c) * F + f] = 0.5 * q3[1 + k] * ts
            for k in range(3):
                jdv[k] = jd0[k] - jd1[k]
            if weld:  # mj_Jdotv rotational part: product rule on 0.5 neg(q1) (omega0 - omega1) q0 relpose
                var dom = om0 - om1
                var qd0 = Q4(0.5 * (-om0[0] * qa[1] - om0[1] * qa[2] - om0[2] * qa[3]), 0.5 * (om0[0] * qa[0] + om0[1] * qa[3] - om0[2] * qa[2]),
                             0.5 * (-om0[0] * qa[3] + om0[1] * qa[0] + om0[2] * qa[1]), 0.5 * (om0[0] * qa[2] - om0[1] * qa[1] + om0[2] * qa[0]))
                var qd1 = Q4(0.5 * (-om1[0] * qb[1] - om1[1] * qb[2] - om1[2] * qb[3]), 0.5 * (om1[0] * qb[0] + om1[1] * qb[3] - om1[2] * qb[2]),
                             0.5 * (-om1[0] * qb[3] + om1[1] * qb[0] + om1[2] * qb[1]), 0.5 * (om1[0] * qb[2] - om1[1] * qb[1] + om1[2] * qb[0]))
                var dj = jr0 - jr1
                var t1 = mul(mul(Q4(qd1[0], -qd1[1], -qd1[2], -qd1[3]), Q4(0, dom[0], dom[1], dom[2])), qa)
                var t2 = mul(mul(nq1, Q4(0, dj[0], dj[1], dj[2])), qa)
                var t3 = mul(mul(nq1, Q4(0, dom[0], dom[1], dom[2])), qd0)
                for k in range(3):
                    jdv[3 + k] = 0.5 * (t1[1 + k] + t2[1 + k] + t3[1 + k]) * ts
            var sq = cpos[0] * cpos[0] + cpos[1] * cpos[1] + cpos[2] * cpos[2]
            if weld:
                sq += cpos[3] * cpos[3] + cpos[4] * cpos[4] + cpos[5] * cpos[5]
            posn = sqrt(sq)
        # mj_diagApprox, getsolparam, getimpedance, makeImpedance, referenceConstraint
        var ref0 = max(mf[ef + 15], 2 * h); var ref1 = mf[ef + 16]
        var s0 = min(Float32(0.9999), max(Float32(1e-4), mf[ef + 17])); var s1 = min(Float32(0.9999), max(Float32(1e-4), mf[ef + 18]))
        var s2 = max(Float32(0), mf[ef + 19]); var s3 = min(Float32(0.9999), max(Float32(1e-4), mf[ef + 20])); var s4 = max(Float32(1), mf[ef + 21])
        var imp = impedance(s0, s1, s2, s3, s4, posn)
        var K = 1 / max(Float32(1e-15), s1 * s1 * ref0 * ref0 * ref1 * ref1)
        var B = 2 / max(Float32(1e-15), s1 * ref0)
        for k in range(size):
            var row = rbase + k
            var dA = mf[ef + 23] if typ == 1 and k >= 3 else mf[ef + 22]
            res[(OEQD + row) * F + f] = 1 / max(Float32(1e-15), (1 - imp) * dA / imp)
            var vel: Float32 = 0
            for c in range(nch):
                vel += get(res, f, jb + k * nch + c) * qvel[Int(mi[CEQC + choff + c]) * F + f]
            res[(OEQA + row) * F + f] = -B * vel - K * imp * cpos[k] - jdv[k]


def update(mi: TI, res: TO, f: Int) -> Float32:
    """MuJoCo PrimalUpdateConstraint, pyramidal rows: state = jar < 0, force = -D jar, qfrc_constraint = J' force,
    returns cost = sum 0.5 D jar^2 over active rows + Gauss."""
    var cost: Float32 = 0
    for v in range(NV):
        res[(OQFC + v) * F + f] = 0
    for ip in range(Int(get(res, f, OPA))):
        var p = Int(get(res, f, OPA + 1 + ip))
        var nch = Int(mi[CPH + 11 * p + 6]); var choff = Int(mi[CPH + 11 * p + 7])
        for sl in range(nslot(mi, res, f, p)):
            if get(res, f, OVALID + sbase(mi, res, f, p) + sl) == 0:
                continue
            for k in range(4):
                var row = 4 * (sbase(mi, res, f, p) + sl) + k
                var jar = get(res, f, OJAR + row)
                var act = jar < 0
                res[(OST + row) * F + f] = 1 if act else 0
                if act:
                    var D = get(res, f, OD + row)
                    cost += 0.5 * D * jar * jar
                    var force = -D * jar
                    var jadr = OJ + jbase(mi, res, f, p) + (4 * sl + k) * nch
                    for c in range(nch):
                        var v = Int(mi[CCHAIN + choff + c])
                        res[(OQFC + v) * F + f] += get(res, f, jadr + c) * force
    comptime if NEQ > 0:  # equality rows are always active (quadratic)
        for e in range(NEQ):
            var eh = CEQH + EQH * e
            var nch = Int(mi[eh + 4]); var choff = Int(mi[eh + 5]); var rbase = Int(mi[eh + 6]); var jb = OEQJ + Int(mi[eh + 7])
            for k in range(Int(mi[eh + 3])):
                var jar = get(res, f, OEQJAR + rbase + k)
                var D = get(res, f, OEQD + rbase + k)
                cost += 0.5 * D * jar * jar
                var force = -D * jar
                for c in range(nch):
                    res[(OQFC + Int(mi[CEQC + choff + c])) * F + f] += get(res, f, jb + k * nch + c) * force
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
    for ip in range(Int(get(res, f, OPA))):
        var p = Int(get(res, f, OPA + 1 + ip))
        var nch = Int(mi[CPH + 11 * p + 6]); var choff = Int(mi[CPH + 11 * p + 7])
        for sl in range(nslot(mi, res, f, p)):
            if get(res, f, OVALID + sbase(mi, res, f, p) + sl) == 0:
                continue
            for k in range(4):
                var row = 4 * (sbase(mi, res, f, p) + sl) + k
                if get(res, f, OST + row) == 0:
                    continue
                var D = get(res, f, OD + row)
                var jadr = OJ + jbase(mi, res, f, p) + (4 * sl + k) * nch
                for a in range(nch):  # lane owns H row d (d % LN): every row the same lane, no race across pairs
                    var d = Int(mi[CCHAIN + choff + a])
                    if d % LN != lane():
                        continue
                    var ja = get(res, f, jadr + a) * D
                    var ra = Int(mi[HADR + d])
                    for bb in range(a + 1):
                        res[(OHN + ra + bb) * F + f] += ja * get(res, f, jadr + bb)
    comptime if NEQ > 0:  # equality rows: always in H
        for e in range(NEQ):
            var eh = CEQH + EQH * e
            var nch = Int(mi[eh + 4]); var choff = Int(mi[eh + 5]); var rbase = Int(mi[eh + 6]); var jb = OEQJ + Int(mi[eh + 7])
            for k in range(Int(mi[eh + 3])):
                var D = get(res, f, OEQD + rbase + k)
                var jadr = jb + k * nch
                for a in range(nch):
                    var d = Int(mi[CEQC + choff + a])
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
    for ip in range(Int(get(res, f, OPA))):
        var p = Int(get(res, f, OPA + 1 + ip))
        var nch = Int(mi[CPH + 11 * p + 6]); var choff = Int(mi[CPH + 11 * p + 7])
        var deep = 0
        for c in range(nch):
            deep = max(deep, Int(mi[CCHAIN + choff + c]))
        var ds = Int(mi[HADR + deep]); var dn = Int(mi[HADR + NV + deep])
        for sl in range(nslot(mi, res, f, p)):
            if get(res, f, OVALID + sbase(mi, res, f, p) + sl) == 0:
                continue
            for k in range(4):
                var row = 4 * (sbase(mi, res, f, p) + sl) + k
                var now = get(res, f, OJAR + row) < 0
                if now == (get(res, f, OST + row) != 0):
                    continue
                for t in range(dn):
                    res[(OMV + Int(mi[HADR + 2 * NV + ds + t])) * F + f] = 0
                var jadr = OJ + jbase(mi, res, f, p) + (4 * sl + k) * nch
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
    comptime if NEQ > 0:  # equality rows: active at every alpha
        for r in range(lane(), NEQR, LN):
            var jar = get(res, f, OEQJAR + r); var jv = get(res, f, OEQJV + r); var D = get(res, f, OEQD + r)
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
    for ip in range(Int(get(res, f, OPA))):
        var p = Int(get(res, f, OPA + 1 + ip))
        var nch = Int(mi[CPH + 11 * p + 6]); var choff = Int(mi[CPH + 11 * p + 7])
        for sl in range(nslot(mi, res, f, p)):
            if get(res, f, OVALID + sbase(mi, res, f, p) + sl) == 0:
                continue
            for k in range(4):
                res[(OJV + 4 * (sbase(mi, res, f, p) + sl) + k) * F + f] = jdot(mi, res, f, OJ + jbase(mi, res, f, p) + (4 * sl + k) * nch, nch, choff, OSRCH)
    comptime if NEQ > 0:
        for e in range(NEQ):
            var eh = CEQH + EQH * e
            var nch = Int(mi[eh + 4]); var choff = Int(mi[eh + 5]); var rbase = Int(mi[eh + 6]); var jb = OEQJ + Int(mi[eh + 7])
            for k in range(Int(mi[eh + 3])):
                res[(OEQJV + rbase + k) * F + f] = jdot(mi, res, f, jb + k * nch, nch, choff, OSRCH, CEQC)
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
    for ip in range(Int(get(res, f, OPA))):
        var p = Int(get(res, f, OPA + 1 + ip))
        var nch = Int(mi[CPH + 11 * p + 6]); var choff = Int(mi[CPH + 11 * p + 7])
        for sl in range(nslot(mi, res, f, p)):
            if get(res, f, OVALID + sbase(mi, res, f, p) + sl) == 0:
                continue
            for k in range(4):
                var row = 4 * (sbase(mi, res, f, p) + sl) + k
                res[(OJAR + row) * F + f] = jdot(mi, res, f, OJ + jbase(mi, res, f, p) + (4 * sl + k) * nch, nch, choff, OCQACC) - get(res, f, OAREF + row)
    comptime if NEQ > 0:
        for e in range(NEQ):
            var eh = CEQH + EQH * e
            var nch = Int(mi[eh + 4]); var choff = Int(mi[eh + 5]); var rbase = Int(mi[eh + 6]); var jb = OEQJ + Int(mi[eh + 7])
            for k in range(Int(mi[eh + 3])):
                res[(OEQJAR + rbase + k) * F + f] = jdot(mi, res, f, jb + k * nch, nch, choff, OCQACC, CEQC) - get(res, f, OEQA + rbase + k)
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
        comptime if NEQ > 0:
            for r in range(NEQR):
                res[(OEQJAR + r) * F + f] += alpha * get(res, f, OEQJV + r)
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
    comptime if NEQ > 0:  # solNoSlip's iteration-0 correction sums every row, equality rows included
        for r in range(NEQR):
            imp0 += 0.5 * get(res, f, OEQD + r) * get(res, f, OEQJAR + r) * get(res, f, OEQJAR + r)
    for v in range(NV):
        res[(OZ + v) * F + f] = get(res, f, OQFC + v)
    for i in range(NV - 1, -1, -1):  # z = L^-T J' f
        var start = Int(mi[MADR + i])
        var x_i = get(res, f, OZ + i)
        for adr in range(start, start + Int(mi[MNNZ + i]) - 1):
            res[(OZ + Int(mi[MCOL + adr])) * F + f] -= get(res, f, OLD + adr) * x_i
    for ip in range(Int(get(res, f, OPA))):
        var p = Int(get(res, f, OPA + 1 + ip))  # w, A and J qacc_smooth - aref are fixed during the sweeps: once per step
        var nch = Int(mi[CPH + 11 * p + 6]); var choff = Int(mi[CPH + 11 * p + 7])
        for sl in range(nslot(mi, res, f, p)):
            if get(res, f, OVALID + sbase(mi, res, f, p) + sl) == 0:
                continue
            for t in range(2):
                var r0 = 4 * (sbase(mi, res, f, p) + sl) + 2 * t
                var jo = jbase(mi, res, f, p) + (4 * sl + 2 * t) * nch
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
        for ip in range(Int(get(res, f, OPA))):
            var p = Int(get(res, f, OPA + 1 + ip))
            var nch = Int(mi[CPH + 11 * p + 6]); var choff = Int(mi[CPH + 11 * p + 7])
            for sl in range(nslot(mi, res, f, p)):
                if get(res, f, OVALID + sbase(mi, res, f, p) + sl) == 0:
                    continue
                for t in range(2):
                    var r0 = 4 * (sbase(mi, res, f, p) + sl) + 2 * t
                    var w = OWC + jbase(mi, res, f, p) + (4 * sl + 2 * t) * nch
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
    for ip in range(Int(get(res, f, OPA))):
        var p = Int(get(res, f, OPA + 1 + ip))
        var nch = Int(mi[CPH + 11 * p + 6]); var choff = Int(mi[CPH + 11 * p + 7])
        for sl in range(nslot(mi, res, f, p)):
            if get(res, f, OVALID + sbase(mi, res, f, p) + sl) == 0:
                continue
            for k in range(4):
                var row = 4 * (sbase(mi, res, f, p) + sl) + k
                var df = get(res, f, OFRC + row)
                if get(res, f, OST + row) != 0:
                    df += get(res, f, OD + row) * get(res, f, OJAR + row)
                var jadr = OJ + jbase(mi, res, f, p) + (4 * sl + k) * nch
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


def rkintegrate(mi: TI, res: TO, f: Int, h: Float32):
    """MuJoCo mj_integratePos (engine_support.c) of X0 = ORKQ along the velocity in ORKW over h, into OQPOS: free and ball joints by
    mju_quatIntegrate (unit(q) * quat(axis, h |w|), w in the body frame), slide and hinge by q += h v."""
    for a in range(NQ):
        res[(OQPOS + a) * F + f] = get(res, f, ORKQ + a)
    for j in range(NJ):
        var a = Int(mi[QADR + j])
        var da = Int(mi[DOFADR + j])
        var t = Int(mi[JTYPE + j])
        if t == 0:
            for c in range(3):
                res[(OQPOS + a + c) * F + f] += h * get(res, f, ORKW + da + c)
            a += 3
            da += 3
        if t <= 1:
            var w = V3(get(res, f, ORKW + da), get(res, f, ORKW + da + 1), get(res, f, ORKW + da + 2), 0)
            var n = sqrt((w * w).reduce_add())
            var axis = w / n if n >= 1e-15 else basis(0)
            var half = 0.5 * h * n
            var s = sin(half)
            var q = unit(Q4(get(res, f, OQPOS + a), get(res, f, OQPOS + a + 1), get(res, f, OQPOS + a + 2), get(res, f, OQPOS + a + 3)))
            put(res, f, OQPOS + a, mul(q, Q4(cos(half), axis[0] * s, axis[1] * s, axis[2] * s)), 4)
        else:
            res[(OQPOS + a) * F + f] += h * get(res, f, ORKW + da)


def rkstage(mi: TI, mf: TF, qpos: TQ, qvel: TV, res: TO, f: Int, s: Int):
    """MuJoCo mj_RungeKutta N=4 (engine_forward.c:1290), one stage per call: the forward pass just ran at X_s (held in the
    qpos/qvel tensors), so F_s = its constrained qacc. Stages 0..2 write X_{s+1} = X0 '+' (A v_s, h) into the tensors
    (A = 1/2, 1/2, 1: qvel_{s+1} = v0 + h A F_s, qpos_{s+1} = integratePos(q0, A v_s, h)); stage 3 forms the B = 1/6, 1/3, 1/3, 1/6
    sums, puts them in OQACC (dX acc: qvel_next = v0 + h OQACC) and OQPOS (integratePos(q0, dX vel, h)) and restores X0 in
    the tensors, so smooth()'s write-back and K=0 re-launches see the original state."""
    var h = mf[TSTEP]
    sync()  # every lane is past its reads of qpos/qvel from the forward pass before they are overwritten
    comptime if STAGE >= 6:
        for v in range(NV):
            res[(ORKF + s * NV + v) * F + f] = get(res, f, OCQACC + v)
    else:  # contact-free: qacc_smooth = M^-1 (qfrc_passive - qfrc_bias)
        for v in range(NV):
            res[(OQACC + v) * F + f] = get(res, f, OPAS + v) - get(res, f, OBIAS + v)
        solveLD(mi, res, f, OQACC, OLD, OLDI)
        for v in range(NV):
            res[(ORKF + s * NV + v) * F + f] = get(res, f, OQACC + v)
    for v in range(NV):
        res[(ORKV + s * NV + v) * F + f] = qvel[v * F + f]
    if s == 0:
        for a in range(NQ):
            res[(ORKQ + a) * F + f] = qpos[a * F + f]
    if s < 3:
        var c: Float32 = 0.5 if s < 2 else 1.0
        for v in range(NV):
            res[(ORKW + v) * F + f] = c * get(res, f, ORKV + s * NV + v)
        rkintegrate(mi, res, f, h)
        for v in range(NV):
            qvel[v * F + f] = get(res, f, ORKV + v) + h * c * get(res, f, ORKF + s * NV + v)
        for a in range(NQ):
            qpos[a * F + f] = get(res, f, OQPOS + a)
    else:
        absolute(mi, qpos, res, f)  # the rows hold the last forward pass (MuJoCo's d after mj_step): world positions of X_3, whose origin is in the tensors now
        for v in range(NV):
            res[(OQACC + v) * F + f] = (get(res, f, ORKF + v) + 2 * get(res, f, ORKF + NV + v) + 2 * get(res, f, ORKF + 2 * NV + v) + get(res, f, ORKF + 3 * NV + v)) / 6
            res[(ORKW + v) * F + f] = (get(res, f, ORKV + v) + 2 * get(res, f, ORKV + NV + v) + 2 * get(res, f, ORKV + 2 * NV + v) + get(res, f, ORKV + 3 * NV + v)) / 6
        rkintegrate(mi, res, f, h)
        for v in range(NV):
            qvel[v * F + f] = get(res, f, ORKV + v)
        for a in range(NQ):
            qpos[a * F + f] = get(res, f, ORKQ + a)
        comptime if K > 0 and STAGE >= 6:  # mj_advance: qacc_warmstart = d->qacc, the last forward pass's
            for v in range(NV):
                res[(OWARM + v) * F + f] = get(res, f, OCQACC + v)
    sync()


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
        comptime if INTEGRATOR == 1:  # RK4: four forward passes, X_1..3 held in qpos/qvel, X0 back after the last
            for rks in range(4):
                step(mi, mf, qpos, qvel, ctrl, res, f, rks)
        else:
            step(mi, mf, qpos, qvel, ctrl, res, f)
            absolute(mi, qpos, res, f)
        comptime if K > 0:  # mj_Euler's write-back: qvel += h qacc, qpos = integrated qpos
            for v in range(NV):
                qvel[v * F + f] += mf[TSTEP] * get(res, f, OQACC + v)
            for a in range(NQ):
                qpos[a * F + f] = get(res, f, OQPOS + a)


def step(mi: TI, mf: TF, qpos: TQ, qvel: TV, ctrl: TU, res: TO, f: Int, rks: Int = 0):

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
            comptime if NMOCAP > 0:  # mj_kinematics (engine_core_smooth.c:86): a mocap body takes its pose from mocap_pos/quat, a world child
                var mid = Int(mi[CMOC + i])
                if mid >= 0:
                    var mrow = NU + 7 * mid
                    xpos = V3(ctrl[mrow * F + f], ctrl[(mrow + 1) * F + f], ctrl[(mrow + 2) * F + f], 0)
                    xquat = Q4(ctrl[(mrow + 3) * F + f], ctrl[(mrow + 4) * F + f], ctrl[(mrow + 5) * F + f], ctrl[(mrow + 6) * F + f])
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
        var npa = 0  # pairs with a valid slot, ascending: every solver loop below visits these
        for p in range(NCP):
            var live = False
            for sl in range(nslot(mi, res, f, p)):
                live = live or get(res, f, OVALID + sbase(mi, res, f, p) + sl) != 0
            if live:
                res[(OPA + 1 + npa) * F + f] = Float32(p)
                npa += 1
        res[OPA * F + f] = Float32(npa)
        comptime if NEQ > 0:
            equality(mi, mf, qpos, qvel, res, f)
        comptime if STAGE >= 7:
            actuation(mi, mf, qpos, qvel, ctrl, res, f)
        comptime if STOP == 5:
            return
        newton(mi, mf, res, f)
        comptime if STOP == 8 or STOP == 9:
            return
        noslip(mi, mf, res, f)
        comptime if K > 0 and INTEGRATOR != 1:  # RK4 saves it after its last forward pass (rkstage)
            for v in range(NV):
                res[(OWARM + v) * F + f] = get(res, f, OCQACC + v)
        comptime if STOP == 6:
            return
    comptime if STAGE >= 5:
        comptime if INTEGRATOR == 1:
            rkstage(mi, mf, qpos, qvel, res, f, rks)
        else:
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
    var u_b = ctx.enqueue_create_buffer[DType.float32](max(NUC, 1) * F)
    var o_b = ctx.enqueue_create_buffer[DType.float32](NROW * F)
    o_b.enqueue_fill(0)  # warm start reads OWARM before the first step writes it
    var mi = load(dir + "mi.bin", NI)
    var mf = load(dir + "mf.bin", NFL)
    var qp = load(dir + "qpos.bin", NQ * F)
    var qv = load(dir + "qvel.bin", NV * F)
    var uv = load(dir + "ctrl.bin", NUC * F) if NUC > 0 else List[UInt8](length=4 * F, fill=0)
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
    comptime if DUMP and STAGE >= 6:
        with open(dir + "dump.bin", "w") as file:
            file.write_bytes(Span(unsafe_ptr=host.unsafe_ptr().unsafe_bitcast[UInt8](), length=NROW * F * 4))
        with open(dir + "dump.txt", "w") as file:
            file.write(String(NROW, " ", F, " ", OVALID, " ", OD, " ", OAREF, " ", OJ, " ", NSLOT, " ", CPH, " ", NCP, " ", CKIND if NCV > 0 else -1, " ", OSD, "\n"))
    print(F, "animals, stage", STAGE, ", K", max(K, 1), ":", us, "us per launch,", Float64(F * max(K, 1)) / us * 1e6, "animal-steps/s")
    comptime if STAGE >= 6:
        var it: Float64 = 0; var ev: Float64 = 0; var hb: Float64 = 0; var slots: Float64 = 0; var itmax: Float32 = 0
        for a in range(F):
            it += Float64(host[ODIAG * F + a]); ev += Float64(host[(ODIAG + 1) * F + a]); hb += Float64(host[(ODIAG + 2) * F + a]); slots += Float64(host[ONCON * F + a]); itmax = max(itmax, host[ODIAG * F + a])
        print("Newton iterations per animal", it / Float64(F), "max", itmax, "; line-search evaluations", ev / Float64(F))
        print("Hessian rebuilds per animal", hb / Float64(F), "; active slots", slots / Float64(F))
