"""Smooth dynamics, ported from MuJoCo 3.9.0 src/engine/engine_core_smooth.c (Apache-2.0), batched:
one GPU thread per animal. STAGE 1: mj_kinematics1 + the inertia and geom part of mj_kinematics2.
STAGE 2: + mj_comPos (subtree_com, cinert, cdof) + mj_crb (sparse M in MuJoCo's own M_rowadr layout).

Model arrays come from agent/body_engine/model.py export (mi.bin int32, mf.bin float32, field order
INTS / FLOATS there); sizes are -D constants, so any MJCF body (fly, spider) builds its own binary.
Orientation is kept as quaternions only (MuJoCo also builds 3x3 xmat; rotating by the quaternion is
the same map and stores 4 floats, not 9). Mocap bodies and the sleep filter are not ported.

Layout: every per-fly array is [row][fly], flies fastest, so a warp reads 32 neighbouring flies
in one transaction. Model arrays are shared by all flies and stay in cache.

usage: smooth DIR/      reads DIR/mi.bin, mf.bin, qpos.bin (NQ x FLIES); writes DIR/smooth.bin, rows:
       xpos, xquat, xipos (bodies 1..), geom_xpos, xanchor, xaxis; STAGE 2 adds subtree_com,
       cinert (bodies 1..), cdof, crb (bodies 1..), M; then prints us per launch
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
comptime NI = MNNZ + NV
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
comptime NFL = ARMATURE + NV
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
comptime NOUT = OSC if STAGE == 1 else OM + NC
comptime LI = row_major[NI]()
comptime LF = row_major[NFL]()
comptime LQ = row_major[NQ * F]()
comptime LO = row_major[NOUT * F]()
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


def smooth(mi: TI, mf: TF, qpos: TQ, res: TO):
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
    var o_b = ctx.enqueue_create_buffer[DType.float32](NOUT * F)
    var mi = load(dir + "mi.bin", NI)
    var mf = load(dir + "mf.bin", NFL)
    var qp = load(dir + "qpos.bin", NQ * F)
    ctx.enqueue_copy(mi_b, mi.unsafe_ptr().unsafe_bitcast[Int32]())
    ctx.enqueue_copy(mf_b, mf.unsafe_ptr().unsafe_bitcast[Float32]())
    ctx.enqueue_copy(q_b, qp.unsafe_ptr().unsafe_bitcast[Float32]())
    var kernel = ctx.compile_function[smooth]()
    var grid = (F + 63) // 64
    ctx.enqueue_function(kernel, TileTensor(mi_b, LI), TileTensor(mf_b, LF), TileTensor(q_b, LQ), TileTensor(o_b, LO), grid_dim=grid, block_dim=64)
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(100):
        ctx.enqueue_function(kernel, TileTensor(mi_b, LI), TileTensor(mf_b, LF), TileTensor(q_b, LQ), TileTensor(o_b, LO), grid_dim=grid, block_dim=64)
    ctx.synchronize()
    var us = Float64(perf_counter_ns() - t0) / 100 / 1000
    var host = List[Float32](length=NOUT * F, fill=0)
    ctx.enqueue_copy(host.unsafe_ptr(), o_b)
    ctx.synchronize()
    with open(dir + "smooth.bin", "w") as file:
        file.write_bytes(Span(unsafe_ptr=host.unsafe_ptr().unsafe_bitcast[UInt8](), length=len(host) * 4))
    print(F, "animals, stage", STAGE, ":", us, "us per launch,", Float64(F) / us * 1e6, "animals/s")
