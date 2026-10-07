"""Live view (docs/plans/BRAIN_BODY_LOOP.md step 2): the body engine and the renderer in one process and one
DeviceContext. Per frame: one smooth launch (K steps, state kept on the card), then the renderer's clear, raster
and shade passes read the poses straight from the engine's output buffer (rows xpos, xquat of bodies 1.., the
layout render.mojo already reads), and one write(2) puts the RGBA frame on stdout. No pose is copied.

build: mojo build $(cat DIR/sizes.txt) $(cat DIR/rsizes.txt) -DFLIES=1 -DSTAGE=7 -DK=k -DW=w -DH=h
       -DPROWS=7*(NBODY-1) live.mojo
brain link (-DBRAIN=1 or 2, docs/plans/SPIDER_BRAIN.md unit 1; BRAIN=0 compiles none of it): each frame becomes one control
       step. BRAIN=1: the ctrl for the step arrives on stdin (NU x F float32, [servo][animal]) and goes to the engine's
       ctrl buffer; BRAIN=2: spider_brain.mojo's gait kernel writes it on the card (drives from DIR/drive.bin, 2 x F float32:
       speed, turn). After the launch a sensor packet goes to stdout every PACKET-th step (arg 11, 0 = never, and at the last step):
       ctrl, qpos, qvel, valid contact slots (OVALID.., NSLOT rows), under -DGAME=1 the 4 game rows, each float32 [row][animal]; with -DEYES=1 then the retinas, uint8
       [animal][eye][H x W] (eyes.mojo; DIR/eyes.bin the eye table, DIR/env.bin the prey's path). Frames (arg 10 = every n-th
       step, 0 = never) go to fd 4 instead of stdout. No target state anywhere in the packet.

spectator (docs/plans/RENDERER.md; each flag compiles none of its block when 0): -DFOLLOW=1 follows the root bodies of DIR/roots.bin
       (body, radius per root; spectate.follow, on the card, same formula as follow_camera.fit_camera) instead of the fixed camera;
       -DHUD=1 draws the energy/health/stamina bars of animal 0 from DIR/game.bin (4 x F float32, [row][animal]), or from the engine's own rows
       under -DGAME=1 (smooth.mojo: work drain, stamina, death gates the servos; rules at the end of mf.bin); -DPROFILE=1 syncs
       between phases and, with arg 12 = FILE, writes the root trace FILE (float32 frame, root, xyz; arg 13 = game rows + effort sums per frame under GAME) and prints one COST line
       (ms per frame: physics, neural_control, render, copy, display; the syncs cost, the run is for the split, not for fps).

usage: live DIR/ FRAMES ex ey ez tx ty tz fovy_deg [OUT]
       DIR holds the engine files (mi.bin, mf.bin, qpos.bin, qvel.bin, ctrl.bin) and the mesh tables (rmi.bin ...);
       FRAMES 0 = until the reader closes the pipe; OUT = a .rgba file gets frame FRAMES-1 instead of the stream
       and OUT.qpos the state after it
       (the check's path); stderr gets ms per frame and simulated time.
       live DIR/ 0 ... | mpv --demuxer=rawvideo --demuxer-rawvideo-w=W --demuxer-rawvideo-h=H
                             --demuxer-rawvideo-mp-format=rgba -
"""
import smooth as eng
import render as ren
from std.ffi import external_call
from std.math import tan
from std.sys import argv, get_defined_int, stderr
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from layout import TileTensor
import spider_brain as brain
import spectate as spec
import eyes

comptime BRAIN = get_defined_int["BRAIN", 0]()
comptime NVALID = eng.NSLOT
comptime FOLLOW = get_defined_int["FOLLOW", 0]()
comptime HUD = get_defined_int["HUD", 0]()
comptime PROFILE = get_defined_int["PROFILE", 0]()
comptime WORLD = get_defined_int["WORLD", 0]()
comptime ORBIT = get_defined_int["ORBIT", 0]() == 1  # FOLLOW camera orbits by DIR/orbit.bin (play)
comptime GAZE = get_defined_int["GAZE", 0]() == 1  # BRAIN = 1 and EYES: F gaze yaws (deg, AM eyes) follow the ctrl on stdin
comptime EYES = get_defined_int["EYES", 0]() == 1  # BRAIN > 0: eyes.mojo retinas in the packet, prey moved by DIR/env.bin
comptime NRET = eyes.F * eyes.NE * eyes.W * eyes.H if EYES else 1
comptime EYEROW = get_defined_int["EYEROW", 0]() == 1  # EYES: render AM_L only and send its darkest pixel per column (F x W bytes)
comptime NPKR = eyes.F * eyes.W if EYEROW else NRET


def write_all(fd: Int, p: Pointer[UInt8, MutUntrackedOrigin], n: Int) -> Bool:
    var left = n
    var q = p
    while left > 0:
        var w = external_call["write", Int](fd, q.unsafe_bitcast[NoneType](), left)
        if w <= 0:
            return False
        q = q.unsafe_offset(w)
        left -= w
    return True


def read_all(fd: Int, p: Pointer[UInt8, MutUntrackedOrigin], n: Int) -> Bool:
    var left = n
    var q = p
    while left > 0:
        var r = external_call["read", Int](fd, q.unsafe_bitcast[NoneType](), left)
        if r <= 0:
            return False
        q = q.unsafe_offset(r)
        left -= r
    return True


def main() raises:
    comptime assert not (eng.CPU and EYES), "CPU build: the eye kernels read the device state, which a CPU step only refreshes per frame"
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
    o_b.enqueue_fill(0)  # warm start reads OWARM before the first step writes it
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
    # -DCPU=1: the step runs on the host over these (one interactive animal: GPU latency is 4.4 ms per 0.1 ms step, host 53 us);
    # the device copies are refreshed after each step, so render, packets and HUD read them unchanged
    var ob = List[UInt8](length=4 * eng.NROW * eng.F, fill=0)
    var hi = eng.host[DType.int32](mi, eng.LI); var hf = eng.host[DType.float32](mf, eng.LF); var hq = eng.host[DType.float32](qp, eng.LQ)
    var hv = eng.host[DType.float32](qv, eng.LV); var hu = eng.host[DType.float32](uv, eng.LU); var ho = eng.host[DType.float32](ob, eng.LO)
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
    var cam_b = ctx.enqueue_create_buffer[DType.float32](ren.NCAM)
    var vis_b = ctx.enqueue_create_buffer[DType.uint64](ren.W * ren.H)
    var img_b = ctx.enqueue_create_buffer[DType.uint32](ren.W * ren.H)
    var out_h = ctx.enqueue_create_host_buffer[DType.uint32](ren.W * ren.H)
    var camh = List[Float32](length=ren.NCAM, fill=0)
    for k in range(6):
        camh[k] = Float32(atof(String(a[3 + k])))
    var fovy = Float32(atof(String(a[9]))) * 3.14159265 / 180
    camh[6] = fovy
    camh[7] = Float32(ren.H) * 0.5 / tan(fovy * 0.5)
    comptime if eng.HFG >= 0:  # the renderer draws the engine's own terrain
        var mfp = mf.unsafe_ptr().unsafe_bitcast[Float32]()
        for k in range(eng.NP):
            camh[16 + k] = mfp[eng.HFO + k]
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
    var to_file = len(a) > 10 and BRAIN == 0
    var render_every = Int(String(a[10])) if BRAIN > 0 else 1
    var packet_every = Int(String(a[11])) if BRAIN > 0 else 0
    var ctrl_h = ctx.enqueue_create_host_buffer[DType.float32](max(eng.NU, 1) * eng.F)
    var pkt_u = ctx.enqueue_create_host_buffer[DType.float32](max(eng.NU, 1) * eng.F)
    var pkt_q = ctx.enqueue_create_host_buffer[DType.float32](eng.NQ * eng.F)
    var pkt_v = ctx.enqueue_create_host_buffer[DType.float32](eng.NV * eng.F)
    var pkt_s = ctx.enqueue_create_host_buffer[DType.float32](max(NVALID, 1) * eng.F)
    var pkt_g = ctx.enqueue_create_host_buffer[DType.float32](4 * eng.F)  # GAME: the game rows ride after the contacts
    var kb = ctx.compile_function[brain.gait]()  # launched only when BRAIN == 2
    var drive_b = ctx.enqueue_create_buffer[DType.float32](2 * eng.F)
    var k_ec = ctx.compile_function[eyes.eye_cam]()  # the eye kernels launch only when EYES
    var k_ecl = ctx.compile_function[eyes.eye_clear]()
    var k_er = ctx.compile_function[eyes.eye_raster]()
    var k_es = ctx.compile_function[eyes.eye_shade]()
    var k_pr = ctx.compile_function[eyes.prey_set]()
    var ecam_b = ctx.enqueue_create_buffer[DType.float32](eyes.F * ren.NCAM if EYES else 1)
    var evis_b = ctx.enqueue_create_buffer[DType.uint64](eyes.F * eyes.W * eyes.H if EYES else 1)
    var eyet_b = ctx.enqueue_create_buffer[DType.float32](eyes.NE * 8)
    var env_b = ctx.enqueue_create_buffer[DType.float32](6 * eng.F)
    var gaze_b = ctx.enqueue_create_buffer[DType.float32](eng.F)
    gaze_b.enqueue_fill(0)
    var gaze_h = ctx.enqueue_create_host_buffer[DType.float32](eng.F)
    var ret_b = ctx.enqueue_create_buffer[DType.uint8](NRET)
    var row_b = ctx.enqueue_create_buffer[DType.uint8](NPKR if EYEROW else 1)
    var pkt_r = ctx.enqueue_create_host_buffer[DType.uint8](NPKR)
    var k_rm = ctx.compile_function[eyes.eye_rowmin]()
    comptime if EYES:
        var et = eng.load(dir + "eyes.bin", eyes.NE * 8)
        var ev = eng.load(dir + "env.bin", 6 * eng.F)
        ctx.enqueue_copy(eyet_b, et.unsafe_ptr().unsafe_bitcast[Float32]())
        ctx.enqueue_copy(env_b, ev.unsafe_ptr().unsafe_bitcast[Float32]())
    comptime if BRAIN == 2:
        var dr = eng.load(dir + "drive.bin", 2 * eng.F)
        ctx.enqueue_copy(drive_b, dr.unsafe_ptr().unsafe_bitcast[Float32]())
    var kf = ctx.compile_function[spec.follow]()  # launched only when FOLLOW
    var kh = ctx.compile_function[spec.hud]()  # launched only when HUD
    var ksn = ctx.compile_function[spec.sense]()  # launched only when WORLD
    var kht = ctx.compile_function[spec.hud_text]()
    var kmm = ctx.compile_function[spec.minimap]()
    var src_b = ctx.enqueue_create_buffer[DType.float32](5 * spec.NSRC)
    var site_b = ctx.enqueue_create_buffer[DType.float32](4 * spec.NSITE)
    var plume_b = ctx.enqueue_create_buffer[DType.float32](2 * spec.NSITE)
    var plume_h = ctx.enqueue_create_host_buffer[DType.float32](2 * spec.NSITE)
    var ptrace = List[Float32]()
    comptime if WORLD > 0:
        var sr = eng.load(dir + "src.bin", 5 * spec.NSRC)
        ctx.enqueue_copy(src_b, sr.unsafe_ptr().unsafe_bitcast[Float32]())
        var st = eng.load(dir + "sites.bin", 4 * spec.NSITE)
        ctx.enqueue_copy(site_b, st.unsafe_ptr().unsafe_bitcast[Float32]())
        plume_b.enqueue_fill(0)
    var tsrc = TileTensor(src_b, spec.LSRC)
    var tsite = TileTensor(site_b, spec.LSITE)
    var tplume = TileTensor(plume_b, spec.LPLUME)
    var root_b = ctx.enqueue_create_buffer[DType.float32](2 * spec.NROOT)
    var gs_b = ctx.enqueue_create_buffer[DType.float32](4 * eng.F)
    var rs_h = List[Float32]()
    comptime if FOLLOW > 0 or PROFILE > 0:
        var rs = eng.load(dir + "roots.bin", 2 * spec.NROOT)
        ctx.enqueue_copy(root_b, rs.unsafe_ptr().unsafe_bitcast[Float32]())
        for k in range(2 * spec.NROOT):
            rs_h.append(rs.unsafe_ptr().unsafe_bitcast[Float32]()[k])
    comptime if HUD > 0 and not eng.GAME:
        var gs = eng.load(dir + "game.bin", 4 * eng.F)
        ctx.enqueue_copy(gs_b, gs.unsafe_ptr().unsafe_bitcast[Float32]())
    var troot = TileTensor(root_b, spec.LROOT)
    var gs_src = gs_b  # the HUD reads the engine's own game rows under GAME, else DIR/game.bin
    comptime if eng.GAME:
        gs_src = o_b.create_sub_buffer[DType.float32](eng.OGAME * eng.F, 4 * eng.F)
    var tgs = TileTensor(gs_src, spec.LGS)
    var game_h = ctx.enqueue_create_host_buffer[DType.float32](4 * eng.F)  # PROFILE + GAME trace: game rows, then the effort rows
    var act_h = ctx.enqueue_create_host_buffer[DType.float32]((eng.NU + eng.NV) * eng.F)
    var gtrace = List[Float32]()
    var pose_h = ctx.enqueue_create_host_buffer[DType.float32](ren.PROWS * ren.F)  # PROFILE: the root trace reads xpos from here
    var trace = List[Float32]()
    var ms = Array[Float64, 5](fill=0)  # PROFILE: neural_control, physics, render, copy, display
    var t0 = perf_counter_ns()
    var fr = 0
    while frames == 0 or fr < frames:
        var tp = perf_counter_ns()
        comptime if EYES:
            ctx.enqueue_function(k_pr, TileTensor(q_b, eng.LQ), TileTensor(v_b, eng.LV), TileTensor(env_b, eyes.LENV), Float32(fr * max(eng.K, 1)) * Float32(1e-4),
                grid_dim=(eng.F + 63) // 64, block_dim=64)
        comptime if BRAIN == 1:
            if not read_all(0, ctrl_h.unsafe_ptr().unsafe_bitcast[UInt8](), max(eng.NU, 1) * eng.F * 4):
                return  # brain gone
            ctx.enqueue_copy(u_b, ctrl_h)
            comptime if eng.CPU:
                for k in range(eng.NU * eng.F):
                    uv.unsafe_ptr().unsafe_bitcast[Float32]()[k] = ctrl_h[k]
            comptime if GAZE:
                if not read_all(0, gaze_h.unsafe_ptr().unsafe_bitcast[UInt8](), eng.F * 4):
                    return
                ctx.enqueue_copy(gaze_b, gaze_h)
        elif BRAIN == 2:
            ctx.enqueue_function(kb, TileTensor(u_b, eng.LU), TileTensor(drive_b, brain.LD), Float32(fr * max(eng.K, 1)) * Float32(1e-4),
                grid_dim=(eng.NU * eng.F + 63) // 64, block_dim=64)
        comptime if PROFILE > 0:
            ctx.synchronize()
            ms[0] += Float64(perf_counter_ns() - tp)
            tp = perf_counter_ns()
        comptime if eng.CPU:
            for f in range(eng.F):
                eng.smooth_at(hi, hf, hq, hv, hu, ho, f)
        else:
            ctx.enqueue_function(step, TileTensor(mi_b, eng.LI), TileTensor(mf_b, eng.LF), TileTensor(q_b, eng.LQ), TileTensor(v_b, eng.LV),
                TileTensor(u_b, eng.LU), TileTensor(o_b, eng.LO), grid_dim=eng.GRID, block_dim=eng.BLOCK)
        comptime if PROFILE > 0:
            ctx.synchronize()
            ms[1] += Float64(perf_counter_ns() - tp)
            tp = perf_counter_ns()
        comptime if EYES:
            var tecam = TileTensor(ecam_b, eyes.LECAM)
            var tevis = TileTensor(evis_b, eyes.LEVIS)
            var tret = TileTensor(ret_b, eyes.LRET)
            for e in range(1 if EYEROW else eyes.NE):
                ctx.enqueue_function(k_ec, tecam, tpose, TileTensor(eyet_b, eyes.LEYE), TileTensor(gaze_b, eyes.LGAZE), Int32(e), grid_dim=(eng.F + 63) // 64, block_dim=64)
                ctx.enqueue_function(k_ecl, tevis, grid_dim=(eyes.F * eyes.W * eyes.H + 255) // 256, block_dim=256)
                ctx.enqueue_function(k_er, tevis, tecam, tpi, tpf, trv, trt, trp, tpose, tcam, grid_dim=(eyes.F * ren.NTRI + 255) // 256, block_dim=256)
                ctx.enqueue_function(k_es, tevis, tret, Int32(e), tpi, tpf, trv, trt, trp, tpose, tcam, grid_dim=(eyes.F * eyes.W * eyes.H + 255) // 256, block_dim=256)
            comptime if EYEROW:
                ctx.enqueue_function(k_rm, tret, TileTensor(row_b, eyes.LROW), grid_dim=(eyes.F * eyes.W + 255) // 256, block_dim=256)
        var draw = render_every > 0 and fr % render_every == 0 if BRAIN > 0 else True
        comptime if WORLD > 0:  # the plume at the odor sites of this step's pose: the HUD text, the trace and a brain's sense packet read it
            ctx.enqueue_function(ksn, tplume, tpose, tsite, tsrc, grid_dim=1, block_dim=64)
        comptime if eng.CPU:  # the device rows only for a frame (pose, game rows); packets read the host arrays below
            if draw:
                ctx.enqueue_copy(o_b, ob.unsafe_ptr().unsafe_bitcast[Float32]())
        if draw:
            comptime if FOLLOW > 0:
                comptime if ORBIT:  # DIR/orbit.bin: on, azimuth, elevation, zoom (play writes it on mouse drag / wheel)
                    try:
                        var ob4 = open(dir + "orbit.bin", "r").read_bytes()
                        if len(ob4) == 16:
                            var p4 = ob4.unsafe_ptr().unsafe_bitcast[Float32]()
                            for k in range(4):
                                camh[11 + k] = p4[k]
                            ctx.enqueue_copy(cam_b, camh.unsafe_ptr())
                    except:
                        pass
                ctx.enqueue_function(kf, tcam, tpose, troot, grid_dim=1, block_dim=1)
            ctx.enqueue_function(kc, vis, grid_dim=(ren.W * ren.H + 255) // 256, block_dim=256)
            ctx.enqueue_function(kr, vis, tpi, tpf, trv, trt, trp, tpose, tcam, grid_dim=(ren.F * ren.NTRI + 255) // 256, block_dim=256)
            ctx.enqueue_function(ks, vis, img, tpi, tpf, trv, trt, trp, tpose, tcam, grid_dim=(ren.W * ren.H + 255) // 256, block_dim=256)
            comptime if HUD > 0:
                ctx.enqueue_function(kh, img, tgs, grid_dim=(spec.PANEL_W * spec.PANEL_H + 63) // 64, block_dim=64)
                comptime if WORLD > 0:
                    ctx.enqueue_function(kht, img, tplume, grid_dim=(spec.PANEL_W * spec.TXT_H + 63) // 64, block_dim=64)
                    ctx.enqueue_function(kmm, img, tpose, troot, tsrc, grid_dim=(spec.MAP * spec.MAP + 63) // 64, block_dim=64)
            comptime if PROFILE > 0:
                ctx.synchronize()
                ms[2] += Float64(perf_counter_ns() - tp)
                tp = perf_counter_ns()
            ctx.enqueue_copy(out_h, img_b)
        var send = False
        comptime if BRAIN > 0:
            send = packet_every > 0 and fr % packet_every == 0 or fr + 1 == frames
            comptime if eng.CPU:
                if send:
                    var o = ob.unsafe_ptr().unsafe_bitcast[Float32]()
                    for k in range(eng.NU * eng.F):
                        pkt_u[k] = uv.unsafe_ptr().unsafe_bitcast[Float32]()[k]
                    for k in range(eng.NQ * eng.F):
                        pkt_q[k] = qp.unsafe_ptr().unsafe_bitcast[Float32]()[k]
                    for k in range(eng.NV * eng.F):
                        pkt_v[k] = qv.unsafe_ptr().unsafe_bitcast[Float32]()[k]
                    for k in range(NVALID * eng.F):
                        pkt_s[k] = o[eng.OVALID * eng.F + k]
                    comptime if eng.GAME:
                        for k in range(4 * eng.F):
                            pkt_g[k] = o[eng.OGAME * eng.F + k]
            else:
                if send:
                    ctx.enqueue_copy(pkt_u, u_b)
                    ctx.enqueue_copy(pkt_q, q_b)
                    ctx.enqueue_copy(pkt_v, v_b)
                    comptime if EYEROW:
                        ctx.enqueue_copy(pkt_r, row_b)
                    elif EYES:
                        ctx.enqueue_copy(pkt_r, ret_b)
                    comptime if NVALID > 0:
                        ctx.enqueue_copy(pkt_s, o_b.create_sub_buffer[DType.float32](eng.OVALID * eng.F, NVALID * eng.F))
                    comptime if eng.GAME:
                        ctx.enqueue_copy(pkt_g, o_b.create_sub_buffer[DType.float32](eng.OGAME * eng.F, 4 * eng.F))
        ctx.synchronize()
        comptime if PROFILE > 0:
            ms[3] += Float64(perf_counter_ns() - tp)
            tp = perf_counter_ns()
            if len(a) > 12:  # outside the timed phases: the root rows of this frame, read back
                ctx.enqueue_copy(pose_h, o_b.create_sub_buffer[DType.float32](0, ren.PROWS * ren.F))
                ctx.synchronize()
                for r in range(spec.NROOT):
                    var b = Int(rs_h[2 * r]) - 1
                    for c in range(3):
                        trace.append(pose_h[(3 * b + c) * ren.F])
                comptime if WORLD > 0:
                    if len(a) > 14:  # concentration food, decoy per site after this frame's steps
                        ctx.enqueue_copy(plume_h, plume_b)
                        ctx.synchronize()
                        for k in range(2 * spec.NSITE):
                            ptrace.append(plume_h[k])
                comptime if eng.GAME:
                    if len(a) > 13:  # energy, health, stamina, dead, sum |qfrc_actuator|, sum |actuator_force| of animal 0 after this frame's steps
                        ctx.enqueue_copy(game_h, gs_src)
                        ctx.enqueue_copy(act_h, o_b.create_sub_buffer[DType.float32](eng.OAF * eng.F, (eng.NU + eng.NV) * eng.F))
                        ctx.synchronize()
                        var qf = Float32(0)
                        var af = Float32(0)
                        for k in range(eng.NV):
                            qf += abs(act_h[(eng.NU + k) * eng.F])
                        for k in range(eng.NU):
                            af += abs(act_h[k * eng.F])
                        for k in range(4):
                            gtrace.append(game_h[k * eng.F])
                        gtrace.append(qf)
                        gtrace.append(af)
                tp = perf_counter_ns()
        fr += 1
        if to_file:
            continue
        comptime if BRAIN > 0:
            if send:
                if not write_all(1, pkt_u.unsafe_ptr().unsafe_bitcast[UInt8](), max(eng.NU, 1) * eng.F * 4) \
                    or not write_all(1, pkt_q.unsafe_ptr().unsafe_bitcast[UInt8](), eng.NQ * eng.F * 4) \
                    or not write_all(1, pkt_v.unsafe_ptr().unsafe_bitcast[UInt8](), eng.NV * eng.F * 4) \
                    or not write_all(1, pkt_s.unsafe_ptr().unsafe_bitcast[UInt8](), NVALID * eng.F * 4):
                    return
                comptime if eng.GAME:
                    if not write_all(1, pkt_g.unsafe_ptr().unsafe_bitcast[UInt8](), 4 * eng.F * 4):
                        return
                comptime if EYES:
                    if not write_all(1, pkt_r.unsafe_ptr(), NPKR):
                        return
        if draw:
            if not write_all(4 if BRAIN > 0 else 1, out_h.unsafe_ptr().unsafe_bitcast[UInt8](), ren.W * ren.H * 4):
                return  # reader gone
        comptime if PROFILE > 0:
            ms[4] += Float64(perf_counter_ns() - tp)
        if fr % 100 == 0:
            var ms_frame = Float64(perf_counter_ns() - t0) / Float64(fr) / 1e6
            print(fr, "frames,", ms_frame, "ms per frame,", Float64(fr * max(eng.K, 1)) * 0.1, "ms simulated", file=stderr)
            comptime if PROFILE > 0:  # running per-phase split, so a long or killed run still reports where the time goes
                var per = 1e-6 / Float64(max(fr, 1))
                print("COST@" + String(fr), "physics", ms[1] * per, "neural_control", ms[0] * per, "render", ms[2] * per, "copy", ms[3] * per, "display", ms[4] * per, file=stderr)
    comptime if PROFILE > 0:
        var per = 1e-6 / Float64(max(fr, 1))
        print("COST physics", ms[1] * per, "neural_control", ms[0] * per, "render", ms[2] * per, "copy", ms[3] * per, "display", ms[4] * per, "frames", fr, file=stderr)
        if len(a) > 12:
            with open(String(a[12]), "w") as file:
                file.write_bytes(Span(unsafe_ptr=trace.unsafe_ptr().unsafe_bitcast[UInt8](), length=len(trace) * 4))
        comptime if eng.GAME:
            if len(a) > 13:
                with open(String(a[13]), "w") as file:
                    file.write_bytes(Span(unsafe_ptr=gtrace.unsafe_ptr().unsafe_bitcast[UInt8](), length=len(gtrace) * 4))
        comptime if WORLD > 0:
            if len(a) > 14:
                with open(String(a[14]), "w") as file:
                    file.write_bytes(Span(unsafe_ptr=ptrace.unsafe_ptr().unsafe_bitcast[UInt8](), length=len(ptrace) * 4))
    if to_file:
        with open(String(a[10]), "w") as file:
            file.write_bytes(Span(unsafe_ptr=out_h.unsafe_ptr().unsafe_bitcast[UInt8](), length=ren.W * ren.H * 4))
        var qh = ctx.enqueue_create_host_buffer[DType.float32](eng.NQ * eng.F)  # the final state, for the stand gate
        ctx.enqueue_copy(qh, q_b)
        ctx.synchronize()
        with open(String(a[10]) + ".qpos", "w") as file:
            file.write_bytes(Span(unsafe_ptr=qh.unsafe_ptr().unsafe_bitcast[UInt8](), length=eng.NQ * eng.F * 4))
        print(fr, "frames,", Float64(perf_counter_ns() - t0) / Float64(fr) / 1e6, "ms per frame (K", max(eng.K, 1), "steps + render)")
