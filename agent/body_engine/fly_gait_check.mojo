"""Harness for fly_gait.mojo (fly_gait.py check): runs the gait kernel for N control steps on the card from DIR/gait.tab, gait.idx, gait.state (float64 /
int32 files), drives DIR/drive_seq.bin (float64, N x 2), ctrl start values DIR/ctrl.bin, and writes DIR/gait_trace.bin (float32, N x NU).
usage: fly_gait_check DIR/ N DT"""
import smooth as eng
import fly_gait as fg
from std.ffi import external_call
from std.sys import argv
from max.gpu.host import DeviceContext
from layout import TileTensor


def main() raises:
    var a = argv()
    var dir = String(a[1])
    var n = Int(String(a[2]))
    var dt = Float64(atof(String(a[3])))
    _ = external_call["setenv", Int32]("MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT".unsafe_ptr(), "0".unsafe_ptr(), Int32(0))
    var ctx = DeviceContext(0)
    var tab = eng.load(dir + "gait.tab", 2 * fg.NLEG * fg.LEGSZ)
    var idx = eng.load(dir + "gait.idx", fg.NLEG * fg.NDOF + fg.NLEG)
    var st = eng.load(dir + "gait.state", 2 * fg.NST)
    var seq = eng.load(dir + "drive_seq.bin", 2 * 2 * n)
    var u0 = eng.load(dir + "ctrl.bin", max(eng.NU, 1))
    var tab_b = ctx.enqueue_create_buffer[DType.float64](fg.NLEG * fg.LEGSZ)
    var idx_b = ctx.enqueue_create_buffer[DType.int32](fg.NLEG * fg.NDOF + fg.NLEG)
    var st_b = ctx.enqueue_create_buffer[DType.float64](fg.NST)
    var dr_b = ctx.enqueue_create_buffer[DType.float64](2)
    var u_b = ctx.enqueue_create_buffer[DType.float32](max(eng.NU, 1) * eng.F)
    ctx.enqueue_copy(tab_b, tab.unsafe_ptr().unsafe_bitcast[Float64]())
    ctx.enqueue_copy(idx_b, idx.unsafe_ptr().unsafe_bitcast[Int32]())
    ctx.enqueue_copy(st_b, st.unsafe_ptr().unsafe_bitcast[Float64]())
    ctx.enqueue_copy(u_b, u0.unsafe_ptr().unsafe_bitcast[Float32]())
    var k = ctx.compile_function[fg.gait]()
    var trace = List[Float32](length=n * max(eng.NU, 1), fill=0)
    var host = ctx.enqueue_create_host_buffer[DType.float32](max(eng.NU, 1) * eng.F)
    var dh = ctx.enqueue_create_host_buffer[DType.float64](2)
    for i in range(n):
        dh[0] = seq.unsafe_ptr().unsafe_bitcast[Float64]()[2 * i]
        dh[1] = seq.unsafe_ptr().unsafe_bitcast[Float64]()[2 * i + 1]
        ctx.enqueue_copy(dr_b, dh)
        ctx.enqueue_function(k, TileTensor(u_b, eng.LU), TileTensor(st_b, fg.LST), TileTensor(dr_b, fg.LDR), TileTensor(tab_b, fg.LTAB), TileTensor(idx_b, fg.LIDX), dt,
            grid_dim=1, block_dim=1)
        ctx.enqueue_copy(host, u_b)
        ctx.synchronize()
        for j in range(eng.NU):
            trace[i * eng.NU + j] = host[j * eng.F]
    with open(dir + "gait_trace.bin", "w") as file:
        file.write_bytes(Span(unsafe_ptr=trace.unsafe_ptr().unsafe_bitcast[UInt8](), length=len(trace) * 4))
    print("fly_gait_check", n, "steps")
