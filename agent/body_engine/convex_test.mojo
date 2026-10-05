"""CPU driver of convex.mojo for convex_check.py: DIR/cases.f64 (CW doubles per case), hull tables, writes DIR/out.f64.

usage: convex_test DIR/   (build with -DCCD64=1 for the double-precision parity run, without for the engine's float32)"""
from std.sys import argv
from std.memory import UnsafePointer
from convex import *

comptime CW = 40  # case: kind1 kind2, size1 3, pos1 3, mat1 9, pos2 3, mat2 9, hull, margin, tol, maxcon
comptime OW = 32  # out: ncon, dist, 5 x (pos 3, normal 3)


def read_f64(path: String) raises -> List[Float64]:
    var b = open(path, "r").read_bytes()
    var n = len(b) // 8
    var out = List[Float64](length=n, fill=0)
    var p = b.unsafe_ptr().unsafe_bitcast[Float64]()
    for i in range(n):
        out[i] = p[i]
    return out^


def read_f32(path: String) raises -> List[Float32]:
    var b = open(path, "r").read_bytes()
    var n = len(b) // 4
    var out = List[Float32](length=n, fill=0)
    var p = b.unsafe_ptr().unsafe_bitcast[Float32]()
    for i in range(n):
        out[i] = p[i]
    return out^


def read_i32(path: String) raises -> List[Int32]:
    var b = open(path, "r").read_bytes()
    var n = len(b) // 4
    var out = List[Int32](length=n, fill=0)
    var p = b.unsafe_ptr().unsafe_bitcast[Int32]()
    for i in range(n):
        out[i] = p[i]
    return out^


@always_inline
def v3(c: List[Float64], k: Int) -> V:
    return V(S(c[k]), S(c[k + 1]), S(c[k + 2]), S(0))


def main() raises:
    var dir = String(argv()[1])
    var cases = read_f64(dir + "cases.f64")
    var hverts = read_f32(dir + "hverts.f32")
    var hfirst = read_i32(dir + "hfirst.i32")
    var hedges = read_i32(dir + "hedges.i32")
    var hmeta = read_i32(dir + "hmeta.i32")  # per hull: vertex offset, first-edge offset, edge offset
    var n = len(cases) // CW
    var out = List[Float64](length=n * OW, fill=0)
    for c in range(n):
        var b = c * CW
        var h = Int(cases[b + 29])
        var pv = hverts.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin]() + 3 * Int(hmeta[3 * h])
        var pf = hfirst.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin]() + Int(hmeta[3 * h + 1])
        var pe = hedges.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin]()  # first[] holds absolute edge addresses
        var o1 = Sp(pv, pf, pe)
        var o2 = Sp(pv, pf, pe)
        o1.kind = Int(cases[b])
        o2.kind = Int(cases[b + 1])
        o1.size = v3(cases, b + 2)
        o1.pos = v3(cases, b + 5)
        o1.r0 = v3(cases, b + 8)
        o1.r1 = v3(cases, b + 11)
        o1.r2 = v3(cases, b + 14)
        o2.pos = v3(cases, b + 17)
        var org = o2.pos  # work relative to the mesh centre: positions are mm-sized, float32 resolves the 1e-6 tolerance
        o1.pos = o1.pos - org
        o2.pos = V(0)
        o2.r0 = v3(cases, b + 20)
        o2.r1 = v3(cases, b + 23)
        o2.r2 = v3(cases, b + 26)
        var margin = S(cases[b + 30])
        o1.margin = margin
        o2.margin = margin
        var st = Status()
        st.tolerance = S(cases[b + 31])
        st.max_contacts = Int(cases[b + 32])
        var d = ccd(st, o1, o2)
        var r = c * OW
        if d < 0:
            var nrm = contact_normal(st)
            var pos = (st.x1 + st.x2) * S(0.5) + org
            out[r] = 1
            out[r + 1] = Float64(margin + d)
            for k in range(3):
                out[r + 2 + k] = Float64(pos[k])
                out[r + 5 + k] = Float64(nrm[k])
    with open(dir + "out.f64", "w") as f:
        f.write_bytes(Span(unsafe_ptr=out.unsafe_ptr().unsafe_bitcast[UInt8](), length=len(out) * 8))
