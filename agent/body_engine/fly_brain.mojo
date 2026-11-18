"""Brain of the fly on the card (live.mojo -DBRAIN=3; private note BRAIN_BODY_LOOP step 4): the resident Shiu brain (agent/body_engine/shiu_resident.mojo), the descending-neuron
readout, the two drives and the FlyGym gait (fly_gait.mojo) in the SAME DeviceContext as the body and the renderer. Per control step, with no host copy between them:
  shiu_resident.advance(K ticks)  ->  dn_sum: spikes of the descending neurons per side (cumulative counts, atomics into 2 int32)
  ->  fly_drive: rate over the last WINDOW control steps / the side's reference rate, clipped to [0, 1]  ->  fly_gait: the ctrl buffer the engine reads.
fly_link.py's host loop is the oracle for the readout (same window, same warm-up: drive 0 until the first window has filled, reference rate = the
symmetric run's own DN rate after that window, measured on the card when cfg[0] = 0).
Files (env FLY3_WORK: stim/consts files of shiu_resident, seed.bin int32 seed + refractory steps, fly3.bin float64 cfg = ref_left, ref_right, silence, freeze_step,
window, and fly3_dn.bin int32 DN neuron ids, left then right; env FLY3_GRAPH: the CSR graph dir; DIR/gait.tab, gait.idx, gait.state as fly_gait.py exports).
-DFLY_NDN= (all DNs) -DFLY_NDNL= (left ones).
"""
from std.collections import Optional
from std.atomic import Atomic
from max.gpu import block_idx, block_dim, thread_idx
from std.os import getenv
from std.sys import get_defined_int
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major
import shiu_resident as sr
import shiu_worker as sw
import fly_gait as fg
from smooth import NU, F, TU, LU, load as load_words

comptime NDN = get_defined_int["FLY_NDN", 1]()
comptime NDNL = get_defined_int["FLY_NDNL", 1]()
comptime WINMAX = 128
comptime LDN = row_major[NDN]()
comptime LCUM = row_major[2]()
comptime LHIST = row_major[2 * (WINMAX + 1)]()
comptime LCFG = row_major[5]()
comptime TDN = TileTensor[DType.int32, type_of(LDN), MutAnyOrigin]
comptime TCUM = TileTensor[DType.int32, type_of(LCUM), MutAnyOrigin]
comptime THIST = TileTensor[DType.int32, type_of(LHIST), MutAnyOrigin]
comptime TCFG = TileTensor[DType.float64, type_of(LCFG), MutAnyOrigin]
comptime TCNT = TileTensor[DType.int32, type_of(sw.LS), MutAnyOrigin]


def dn_sum(counts: TCNT, idx: TDN, cum: TCUM):
    """Cumulative spikes of the descending neurons, left ids first: cum[0] left, cum[1] right (cum zeroed before the launch)."""
    var k = Int(block_idx.x * block_dim.x + thread_idx.x)
    if k < NDN:
        _ = Atomic.fetch_add(cum.ptr.unsafe_offset(0 if k < NDNL else 1), counts[Int(idx[k])])


def fly_drive(cum: TCUM, hist: THIST, cfg: TCFG, drive: fg.TDR, step: Int32, dt_k: Float64):
    """Drive per side from the DN spike counts of the last `window` control steps (fly_link.run's readout): cfg = ref_left, ref_right (0: take this run's own
    rate at the end of the first window), silence (1: the readout is zeroed), freeze_step, window."""
    if thread_idx.x != 0:
        return
    var i = Int(step)
    var win = Int(cfg[4])
    var ring = win + 1
    hist[(i + 1) % ring] = cum[0]
    hist[WINMAX + 1 + (i + 1) % ring] = cum[1]
    var w = min(i + 1, win)
    var old = (i + 1 - w) % ring
    var rl = Float64(cum[0] - hist[old]) / (Float64(w) * dt_k)
    var rr = Float64(cum[1] - hist[WINMAX + 1 + old]) / (Float64(w) * dt_k)
    if i + 1 == win and cfg[0] == 0:
        cfg[0] = max(rl, 1.0)
        cfg[1] = max(rr, 1.0)
    var active = i + 1 >= win and cfg[0] != 0
    if cfg[2] != 0 or not active:
        rl = 0
        rr = 0
    drive[0] = min(max(rl / (cfg[0] if cfg[0] != 0 else 1.0), 0.0), 1.0)
    drive[1] = min(max(rr / (cfg[1] if cfg[1] != 0 else 1.0), 0.0), 1.0)


struct FlyBrain[enabled: Bool](Movable):
    var res: Optional[sr.Resident]
    var dn_b: DeviceBuffer[DType.int32]
    var cum_b: DeviceBuffer[DType.int32]
    var hist_b: DeviceBuffer[DType.int32]
    var cfg_b: DeviceBuffer[DType.float64]
    var drive_b: DeviceBuffer[DType.float64]
    var tab_b: DeviceBuffer[DType.float64]
    var idx_b: DeviceBuffer[DType.int32]
    var st_b: DeviceBuffer[DType.float64]
    var freeze: Int

    def __init__(out self, ctx: DeviceContext, dir: String) raises:
        self.dn_b = ctx.enqueue_create_buffer[DType.int32](NDN)
        self.cum_b = ctx.enqueue_create_buffer[DType.int32](2)
        self.hist_b = ctx.enqueue_create_buffer[DType.int32](2 * (WINMAX + 1))
        self.cfg_b = ctx.enqueue_create_buffer[DType.float64](5)
        self.drive_b = ctx.enqueue_create_buffer[DType.float64](2)
        self.tab_b = ctx.enqueue_create_buffer[DType.float64](fg.NLEG * fg.LEGSZ)
        self.idx_b = ctx.enqueue_create_buffer[DType.int32](fg.NLEG * fg.NDOF + fg.NLEG)
        self.st_b = ctx.enqueue_create_buffer[DType.float64](fg.NST)
        self.res = Optional[sr.Resident]()
        self.freeze = -1
        comptime if Self.enabled:
            var work = getenv("FLY3_WORK", "")
            var graph = getenv("FLY3_GRAPH", "")
            if work == "" or graph == "":
                raise Error("BRAIN=3 needs FLY3_WORK and FLY3_GRAPH")
            var seed_raw = load_words(work + "seed.bin", 2)
            var seed = seed_raw.unsafe_ptr().unsafe_bitcast[Int32]()
            self.res = Optional(sr.Resident(ctx, graph, work, UInt32(seed.unsafe_offset(0)[]), Int(seed.unsafe_offset(1)[])))
            var dn = load_words(work + "fly3_dn.bin", NDN)
            var cfg = load_words(work + "fly3.bin", 10)
            var tab = load_words(dir + "gait.tab", 2 * fg.NLEG * fg.LEGSZ)
            var idx = load_words(dir + "gait.idx", fg.NLEG * fg.NDOF + fg.NLEG)
            var st = load_words(dir + "gait.state", 2 * fg.NST)
            ctx.enqueue_copy(self.dn_b, dn.unsafe_ptr().unsafe_bitcast[Int32]())
            ctx.enqueue_copy(self.cfg_b, cfg.unsafe_ptr().unsafe_bitcast[Float64]())
            ctx.enqueue_copy(self.tab_b, tab.unsafe_ptr().unsafe_bitcast[Float64]())
            ctx.enqueue_copy(self.idx_b, idx.unsafe_ptr().unsafe_bitcast[Int32]())
            ctx.enqueue_copy(self.st_b, st.unsafe_ptr().unsafe_bitcast[Float64]())
            self.cum_b.enqueue_fill(0)
            self.hist_b.enqueue_fill(0)
            self.drive_b.enqueue_fill(0.0)
            self.freeze = Int(cfg.unsafe_ptr().unsafe_bitcast[Float64]().unsafe_offset(3)[])
            ctx.synchronize()  # the host lists above are freed at the end of this block

    def step(mut self, ctx: DeviceContext, ctrl: TU, fr: Int, k: Int) raises:
        """One control step's brain: advance k ticks, read the DNs, make the drives, write the ctrl the engine reads. From `freeze` on the ctrl is held (the brain runs on)."""
        comptime if Self.enabled:
            # compile_function is cached by the runtime (measured 70 us for compile + launch): struct fields cannot hold a DeviceFunction, so each call asks for it
            var f_int = ctx.compile_function[sw.integrate]()
            var f_del = ctx.compile_function[sw.deliver]()
            var f_kick = ctx.compile_function[sw.kick]()
            self.res.value().advance(ctx, k, f_int, f_del, f_kick)
            if self.freeze >= 0 and fr >= self.freeze:
                return
            self.cum_b.enqueue_fill(0)
            ctx.enqueue_function(ctx.compile_function[dn_sum](), TileTensor(self.res.value().counts, sw.LS), TileTensor(self.dn_b, LDN), TileTensor(self.cum_b, LCUM),
                grid_dim=(NDN + 63) // 64, block_dim=64)
            ctx.enqueue_function(ctx.compile_function[fly_drive](), TileTensor(self.cum_b, LCUM), TileTensor(self.hist_b, LHIST), TileTensor(self.cfg_b, LCFG),
                TileTensor(self.drive_b, fg.LDR), Int32(fr), Float64(k) * 1e-4, grid_dim=1, block_dim=1)
            ctx.enqueue_function(ctx.compile_function[fg.gait](), ctrl, TileTensor(self.st_b, fg.LST), TileTensor(self.drive_b, fg.LDR), TileTensor(self.tab_b, fg.LTAB),
                TileTensor(self.idx_b, fg.LIDX), Float64(k) * 1e-4, grid_dim=1, block_dim=1)
