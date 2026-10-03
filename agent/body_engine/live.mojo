"""Live view (docs/plans/BRAIN_BODY_LOOP.md step 2): the body engine and the renderer in one process and one
DeviceContext. Per frame: one smooth launch (K steps, state kept on the card), then the renderer's clear, raster
and shade passes read the poses straight from the engine's output buffer (rows xpos, xquat of bodies 1.., the
layout render.mojo already reads), and one write(2) puts the RGBA frame on stdout. No pose is copied.

build: mojo build $(cat DIR/sizes.txt) $(cat DIR/rsizes.txt) -DFLIES=1 -DSTAGE=7 -DK=k -DW=w -DH=h
       -DPROWS=7*(NBODY-1) live.mojo
usage: live DIR/ FRAMES ex ey ez tx ty tz fovy_deg [OUT]
       DIR holds the engine files (mi.bin, mf.bin, qpos.bin, qvel.bin, ctrl.bin) and the mesh tables (rmi.bin ...);
       FRAMES 0 = until the reader closes the pipe; OUT = a .rgba file gets frame FRAMES-1 instead of the stream
       (the check's path); stderr gets ms per frame and simulated time.
       live DIR/ 0 ... | mpv --demuxer=rawvideo --demuxer-rawvideo-w=W --demuxer-rawvideo-h=H
                             --demuxer-rawvideo-mp-format=rgba -
"""
import smooth as eng
import render as ren
from std.ffi import external_call
from std.math import tan
from std.sys import argv, stderr
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from layout import TileTensor


def main() raises:
    var a = argv()
    var dir = String(a[1])
    var frames = Int(String(a[2]))
    _ = external_call["setenv", Int32]("MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT".unsafe_ptr(), "0".unsafe_ptr(), Int32(0))
    var ctx = DeviceContext(0)
    # engine: smooth.main's buffers
    var mi_b = ctx.enqueue_create_buffer[DType.int32](eng.NI)
    var mf_b = ctx.enqueue_create_buffer[DType.float32](eng.NFL)
    var q_b = ctx.enqueue_create_buffer[DType.float32](eng.NQ * eng.F)
    var v_b = ctx.enqueue_create_buffer[DType.float32](eng.NV * eng.F)
    var u_b = ctx.enqueue_create_buffer[DType.float32](max(eng.NU, 1) * eng.F)
    var o_b = ctx.enqueue_create_buffer[DType.float32](eng.NROW * eng.F)
    var mi = eng.load(dir + "mi.bin", eng.NI)
    var mf = eng.load(dir + "mf.bin", eng.NFL)
    var qp = eng.load(dir + "qpos.bin", eng.NQ * eng.F)
    var qv = eng.load(dir + "qvel.bin", eng.NV * eng.F)
    var uv = eng.load(dir + "ctrl.bin", eng.NU * eng.F)
    ctx.enqueue_copy(mi_b, mi.unsafe_ptr().unsafe_bitcast[Int32]())
    ctx.enqueue_copy(mf_b, mf.unsafe_ptr().unsafe_bitcast[Float32]())
    ctx.enqueue_copy(q_b, qp.unsafe_ptr().unsafe_bitcast[Float32]())
    ctx.enqueue_copy(v_b, qv.unsafe_ptr().unsafe_bitcast[Float32]())
    ctx.enqueue_copy(u_b, uv.unsafe_ptr().unsafe_bitcast[Float32]())
    var step = ctx.compile_function[eng.smooth]()
    var grid = (eng.F + 63) // 64
    # renderer: render.main's buffers, the pose tensor over the engine's output
    var pi_h = ren.load(dir + "rmi.bin", 20 * ren.NPART)
    var pf_h = ren.load(dir + "rmf.bin", 4 * ren.PFS * ren.NPART)
    var rv_h = ren.load(dir + "rv.bin", 6 * ren.NRV)
    var rt_h = ren.load(dir + "rt.bin", 2 * ren.NRT)
    var rp_h = ren.load(dir + "rp.bin", ren.NTRI)
    var pi_b = ctx.enqueue_create_buffer[DType.int32](5 * ren.NPART)
    var pf_b = ctx.enqueue_create_buffer[DType.float32](ren.PFS * ren.NPART)
    var rv_b = ctx.enqueue_create_buffer[DType.int16](3 * ren.NRV)
    var rt_b = ctx.enqueue_create_buffer[DType.uint16](ren.NRT)
    var rp_b = ctx.enqueue_create_buffer[DType.uint8](ren.NTRI)
    var cam_b = ctx.enqueue_create_buffer[DType.float32](16)
    var vis_b = ctx.enqueue_create_buffer[DType.uint64](ren.W * ren.H)
    var img_b = ctx.enqueue_create_buffer[DType.uint32](ren.W * ren.H)
    var out_h = ctx.enqueue_create_host_buffer[DType.uint32](ren.W * ren.H)
    var camh = List[Float32](length=16, fill=0)
    for k in range(6):
        camh[k] = Float32(atof(String(a[3 + k])))
    var fovy = Float32(atof(String(a[9]))) * 3.14159265 / 180
    camh[6] = fovy
    camh[7] = Float32(ren.H) * 0.5 / tan(fovy * 0.5)
    ctx.enqueue_copy(pi_b, pi_h.unsafe_ptr().unsafe_bitcast[Int32]())
    ctx.enqueue_copy(pf_b, pf_h.unsafe_ptr().unsafe_bitcast[Float32]())
    ctx.enqueue_copy(rv_b, rv_h.unsafe_ptr().unsafe_bitcast[Int16]())
    ctx.enqueue_copy(rt_b, rt_h.unsafe_ptr().unsafe_bitcast[UInt16]())
    ctx.enqueue_copy(rp_b, rp_h.unsafe_ptr().unsafe_bitcast[UInt8]())
    ctx.enqueue_copy(cam_b, camh.unsafe_ptr())
    var kc = ctx.compile_function[ren.clear]()
    var kr = ctx.compile_function[ren.raster]()
    var ks = ctx.compile_function[ren.shade]()
    var vis = TileTensor(vis_b, ren.LVIS)
    var img = TileTensor(img_b, ren.LVIS)
    var tpi = TileTensor(pi_b, ren.LPI)
    var tpf = TileTensor(pf_b, ren.LPF)
    var trv = TileTensor(rv_b, ren.LRV)
    var trt = TileTensor(rt_b, ren.LRT)
    var trp = TileTensor(rp_b, ren.LRP)
    var tpose = TileTensor(o_b, ren.LPOSE)  # the engine's own rows; PROWS covers xpos + xquat only
    var tcam = TileTensor(cam_b, ren.LCAM)
    var to_file = len(a) > 10
    var t0 = perf_counter_ns()
    var fr = 0
    while frames == 0 or fr < frames:
        ctx.enqueue_function(step, TileTensor(mi_b, eng.LI), TileTensor(mf_b, eng.LF), TileTensor(q_b, eng.LQ), TileTensor(v_b, eng.LV),
            TileTensor(u_b, eng.LU), TileTensor(o_b, eng.LO), grid_dim=grid, block_dim=64)
        ctx.enqueue_function(kc, vis, grid_dim=(ren.W * ren.H + 255) // 256, block_dim=256)
        ctx.enqueue_function(kr, vis, tpi, tpf, trv, trt, trp, tpose, tcam, grid_dim=(ren.F * ren.NTRI + 255) // 256, block_dim=256)
        ctx.enqueue_function(ks, vis, img, tpi, tpf, trv, trt, trp, tpose, tcam, grid_dim=(ren.W * ren.H + 255) // 256, block_dim=256)
        ctx.enqueue_copy(out_h, img_b)
        ctx.synchronize()
        fr += 1
        if to_file:
            continue
        var left = ren.W * ren.H * 4
        var p = out_h.unsafe_ptr().unsafe_bitcast[UInt8]()
        while left > 0:
            var n = external_call["write", Int](Int(1), p.bitcast[NoneType](), left)
            if n <= 0:
                return  # reader gone
            p = p.unsafe_offset(n)
            left -= n
        if fr % 100 == 0:
            var ms = Float64(perf_counter_ns() - t0) / Float64(fr) / 1e6
            print(fr, "frames,", ms, "ms per frame,", Float64(fr * max(eng.K, 1)) * 0.1, "ms simulated", file=stderr)
    if to_file:
        with open(String(a[10]), "w") as file:
            file.write_bytes(Span(unsafe_ptr=out_h.unsafe_ptr().unsafe_bitcast[UInt8](), length=ren.W * ren.H * 4))
        print(fr, "frames,", Float64(perf_counter_ns() - t0) / Float64(fr) / 1e6, "ms per frame (K", max(eng.K, 1), "steps + render)")
