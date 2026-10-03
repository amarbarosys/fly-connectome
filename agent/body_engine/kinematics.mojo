"""Forward kinematics, a port of MuJoCo 3.9.0 mj_kinematics1 + the inertia and geom part of
mj_kinematics2 (src/engine/engine_core_smooth.c, Apache-2.0), batched: one GPU thread per fly.

Model arrays come from agent/body_engine/model.py export (mi.bin int32, mf.bin float32, field order
INTS / FLOATS there); sizes are -D constants, so any MJCF body (fly, spider) builds its own binary.
Orientation is kept as quaternions only (MuJoCo also builds 3x3 xmat; rotating by the quaternion is
the same map and stores 4 floats, not 9). Mocap bodies and the sleep filter are not ported.

Layout: every per-fly array is [row][fly], flies fastest, so a warp reads 32 neighbouring flies
in one transaction. Model arrays are shared by all flies and stay in cache.

usage: kinematics DIR/      reads DIR/mi.bin, mf.bin, qpos.bin (NQ x FLIES); writes DIR/kinematics.bin
       rows: xpos, xquat, xipos (bodies 1..), geom_xpos, xanchor, xaxis; then prints us per launch
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
comptime F = get_defined_int["FLIES"]()
# mi.bin: body_parentid, body_jntadr, body_jntnum, jnt_type, jnt_qposadr, geom_bodyid
comptime PARENT = 0
comptime JNTADR = NB
comptime JNTNUM = 2 * NB
comptime JTYPE = 3 * NB
comptime QADR = 3 * NB + NJ
comptime GBODY = 3 * NB + 2 * NJ
comptime NI = 3 * NB + 2 * NJ + NG
# mf.bin: body_pos, body_quat, jnt_pos, jnt_axis, qpos0, body_ipos, body_iquat, geom_pos, geom_quat
comptime BPOS = 0
comptime BQUAT = 3 * NB
comptime JPOS = 7 * NB
comptime JAXIS = 7 * NB + 3 * NJ
comptime QPOS0 = 7 * NB + 6 * NJ
comptime IPOS = 7 * NB + 6 * NJ + NQ
comptime GPOS = 14 * NB + 6 * NJ + NQ
comptime NFL = 14 * NB + 6 * NJ + NQ + 7 * NG
# kinematics.bin rows (per fly): xpos, xquat, xipos of bodies 1.., geom_xpos, xanchor, xaxis
comptime OXQ = 3 * (NB - 1)
comptime OXI = 7 * (NB - 1)
comptime OG = 10 * (NB - 1)
comptime OA = 10 * (NB - 1) + 3 * NG
comptime OX = 10 * (NB - 1) + 3 * NG + 3 * NJ
comptime NOUT = 10 * (NB - 1) + 3 * NG + 6 * NJ
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
def body(res: TO, f: Int, i: Int) -> Tuple[V3, Q4]:
    """World frame of body i (0 = world), already written by this thread."""
    if i == 0:
        return (V3(0), Q4(1, 0, 0, 0))
    var r = 3 * (i - 1)
    var s = OXQ + 4 * (i - 1)
    return (V3(res[r * F + f], res[(r + 1) * F + f], res[(r + 2) * F + f], 0),
            Q4(res[s * F + f], res[(s + 1) * F + f], res[(s + 2) * F + f], res[(s + 3) * F + f]))


def kinematics(mi: TI, mf: TF, qpos: TQ, res: TO):
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
    var kernel = ctx.compile_function[kinematics]()
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
    with open(dir + "kinematics.bin", "w") as file:
        file.write_bytes(Span(unsafe_ptr=host.unsafe_ptr().unsafe_bitcast[UInt8](), length=len(host) * 4))
    print(F, "flies:", us, "us per kinematics launch,", Float64(F) / us * 1e6, "fly-kinematics/s")
