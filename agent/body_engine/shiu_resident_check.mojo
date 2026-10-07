"""Harness for shiu_resident.mojo (agent/verify_shiu_resident.py): Resident on a DeviceContext it does not own, advanced in two calls, then counts and
membrane voltages written to WORK/resident_counts.bin and resident_u.bin for a bit-exact comparison with ShiuLive.advance + snapshot.
usage: shiu_resident_check GRAPH_DIR/ WORK_DIR/ SEED REFRACTORY STEPS_A STEPS_B"""
import shiu_resident as sr
import shiu_worker as sw
from std.sys import argv
from max.gpu.host import DeviceContext


def main() raises:
    var a = argv()
    var work = String(a[2])
    var ctx = DeviceContext(0)
    var f_int = ctx.compile_function[sw.integrate]()
    var f_del = ctx.compile_function[sw.deliver]()
    var f_kick = ctx.compile_function[sw.kick]()
    var r = sr.Resident(ctx, String(a[1]), work, UInt32(Int(String(a[3]))), Int(String(a[4])))
    r.advance(ctx, Int(String(a[5])), f_int, f_del, f_kick)
    r.advance(ctx, Int(String(a[6])), f_int, f_del, f_kick)
    var hc = List[Int32](length=sw.S, fill=0)
    var hu = List[Float32](length=sw.S, fill=0.0)
    ctx.enqueue_copy(hc.unsafe_ptr(), r.counts)
    ctx.enqueue_copy(hu.unsafe_ptr(), r.u)
    ctx.synchronize()
    with open(work + "resident_counts.bin", "w") as f:
        f.write_bytes(Span(unsafe_ptr=hc.unsafe_ptr().unsafe_bitcast[UInt8](), length=len(hc) * 4))
    with open(work + "resident_u.bin", "w") as f:
        f.write_bytes(Span(unsafe_ptr=hu.unsafe_ptr().unsafe_bitcast[UInt8](), length=len(hu) * 4))
    print("shiu_resident_check ok", r.t, "ticks")
