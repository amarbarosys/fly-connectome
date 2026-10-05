"""The resident Shiu brain as an importable API (live.mojo -DBRAIN=3; docs/plans/BRAIN_BODY_LOOP.md step 4): the kernels of shiu_worker.mojo (imported, not
copied: integrate, deliver, kick) launched on the CALLER's DeviceContext, so the brain, the sampler, the gait and the body share one context and nothing round-trips
through the host. State stays on the card across `advance` calls with shiu_worker's resident "advance" semantics (no reset, one absolute tick, seed + 0).

Inputs (what ShiuLive.start writes, agent/shiu.py): GRAPH_DIR row_ptr.bin, col_idx.bin, weights.bin; WORK_DIR stim_index.bin (int32 entries, column x N + neuron),
stim_key.bin (int32 Poisson stream per entry), stim_p.bin (float32 probability per step), consts.bin (float32 em, eg, c, v_th, kick). The plain model only: one edge
group at scale 1, no plasticity, no slow modulation (those stay in shiu_worker's own resident loop). Build with -DSHIU_N= -DSHIU_E= -DSHIU_C= like shiu_worker.
"""
from std.ffi import external_call
from std.os.path import exists
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor
import shiu_worker as sw

comptime I32 = DeviceBuffer[DType.int32]
comptime F32 = DeviceBuffer[DType.float32]
comptime U8 = DeviceBuffer[DType.uint8]
comptime FInt = type_of(DeviceContext(0).compile_function[sw.integrate]())
comptime FDel = type_of(DeviceContext(0).compile_function[sw.deliver]())
comptime FKick = type_of(DeviceContext(0).compile_function[sw.kick]())


struct Resident(Movable):
    var rows: I32
    var cols: I32
    var wts: F32
    var grp: U8
    var scale: F32
    var rank: I32
    var factor: F32
    var u: F32
    var g: F32
    var last: I32
    var rfc: U8
    var counts: I32
    var stim: I32
    var key: I32
    var p: F32
    var amt: F32
    var w0: I32
    var w1: I32
    var ring: I32
    var fill: I32
    var m: F32
    var modp: I32
    var mk: F32
    var t: Int
    var seed: UInt32
    var stim_count: Int
    var em: Float32
    var eg: Float32
    var c: Float32
    var vth: Float32

    def __init__(out self, ctx: DeviceContext, graph: String, work: String, seed: UInt32, refractory: Int) raises:
        var rows = sw.load(graph + "row_ptr.bin", sw.N + 1)
        var cols = sw.load(graph + "col_idx.bin", sw.E)
        var wts = sw.load(graph + "weights.bin", sw.E)
        var stim_raw = sw.load(work + "stim_index.bin", -1)
        var count = len(stim_raw) // 4
        var p_raw = sw.load(work + "stim_p.bin", count)
        var k_raw = sw.load(work + "consts.bin", 5)
        var key_raw = sw.load(work + "stim_key.bin", count) if exists(work + "stim_key.bin") else stim_raw.copy()
        var k = k_raw.unsafe_ptr().unsafe_bitcast[Float32]()
        if count > sw.S or refractory > 255:
            raise Error("invalid run inputs")
        var stim = List[Int32](length=sw.S, fill=0)
        var p = List[Float32](length=sw.S, fill=0.0)
        var rfc = List[UInt8](length=sw.S, fill=UInt8(refractory))
        for i in range(count):
            stim[i] = stim_raw.unsafe_ptr().unsafe_bitcast[Int32]().unsafe_offset(i)[]
            p[i] = p_raw.unsafe_ptr().unsafe_bitcast[Float32]().unsafe_offset(i)[]
            rfc[Int(stim[i])] = 0  # a Poisson-kick entry drops its cell's refractory period (shiu_worker main)
        var rank = List[Int32](length=sw.N + 1, fill=0)
        var modp = List[Int32](length=sw.GROUPS, fill=-1)
        var mk = List[Float32](length=3 * sw.MP, fill=0)
        var amt = List[Float32](length=max(count, 1), fill=k.unsafe_offset(4)[])
        var w0 = List[Int32](length=max(count, 1), fill=0)
        var w1 = List[Int32](length=max(count, 1), fill=Int32.MAX)
        self.rows = ctx.enqueue_create_buffer[DType.int32](sw.N + 1)
        self.cols = ctx.enqueue_create_buffer[DType.int32](sw.E)
        self.wts = ctx.enqueue_create_buffer[DType.float32](sw.E)
        self.grp = ctx.enqueue_create_buffer[DType.uint8](sw.E)
        self.scale = ctx.enqueue_create_buffer[DType.float32](sw.C * sw.GROUPS)
        self.rank = ctx.enqueue_create_buffer[DType.int32](sw.N + 1)
        self.factor = ctx.enqueue_create_buffer[DType.float32](2 * sw.C * sw.KMAX)
        self.u = ctx.enqueue_create_buffer[DType.float32](sw.S)
        self.g = ctx.enqueue_create_buffer[DType.float32](sw.S)
        self.last = ctx.enqueue_create_buffer[DType.int32](sw.S)
        self.rfc = ctx.enqueue_create_buffer[DType.uint8](sw.S)
        self.counts = ctx.enqueue_create_buffer[DType.int32](sw.S)
        self.stim = ctx.enqueue_create_buffer[DType.int32](sw.S)
        self.key = ctx.enqueue_create_buffer[DType.int32](sw.S)
        self.p = ctx.enqueue_create_buffer[DType.float32](sw.S)
        self.amt = ctx.enqueue_create_buffer[DType.float32](max(count, 1))
        self.w0 = ctx.enqueue_create_buffer[DType.int32](max(count, 1))
        self.w1 = ctx.enqueue_create_buffer[DType.int32](max(count, 1))
        self.ring = ctx.enqueue_create_buffer[DType.int32](sw.SLOTS * sw.CAP)
        self.fill = ctx.enqueue_create_buffer[DType.int32](sw.SLOTS + 1)
        self.m = ctx.enqueue_create_buffer[DType.float32](1)
        self.modp = ctx.enqueue_create_buffer[DType.int32](sw.GROUPS)
        self.mk = ctx.enqueue_create_buffer[DType.float32](3 * sw.MP)
        ctx.enqueue_copy(self.rank, rank.unsafe_ptr())
        ctx.enqueue_copy(self.modp, modp.unsafe_ptr())
        ctx.enqueue_copy(self.mk, mk.unsafe_ptr())
        self.factor.enqueue_fill(1.0)
        self.m.enqueue_fill(0.0)
        ctx.enqueue_copy(self.rows, rows.unsafe_ptr().unsafe_bitcast[Int32]())
        ctx.enqueue_copy(self.cols, cols.unsafe_ptr().unsafe_bitcast[Int32]())
        ctx.enqueue_copy(self.wts, wts.unsafe_ptr().unsafe_bitcast[Float32]())
        self.grp.enqueue_fill(0)
        self.scale.enqueue_fill(1.0)
        ctx.enqueue_copy(self.rfc, rfc.unsafe_ptr())
        ctx.enqueue_copy(self.stim, stim.unsafe_ptr())
        if count > 0:
            ctx.enqueue_copy(self.key.create_sub_buffer[DType.int32](0, count), key_raw.unsafe_ptr().unsafe_bitcast[Int32]())
        ctx.enqueue_copy(self.p, p.unsafe_ptr())
        ctx.enqueue_copy(self.amt, amt.unsafe_ptr())
        ctx.enqueue_copy(self.w0, w0.unsafe_ptr())
        ctx.enqueue_copy(self.w1, w1.unsafe_ptr())
        self.u.enqueue_fill(0.0)
        self.g.enqueue_fill(0.0)
        self.last.enqueue_fill(-1000000000)
        self.counts.enqueue_fill(0)
        self.fill.enqueue_fill(0)
        ctx.synchronize()  # the host lists above are freed at the end of this call
        self.t = 0
        self.seed = seed
        self.stim_count = count
        self.em = k.unsafe_offset(0)[]
        self.eg = k.unsafe_offset(1)[]
        self.c = k.unsafe_offset(2)[]
        self.vth = k.unsafe_offset(3)[]

    def advance(mut self, ctx: DeviceContext, steps: Int, f_int: FInt, f_del: FDel, f_kick: FKick) raises:
        """`steps` ticks, enqueued only: integrate + threshold + reset, push delivery of the spikes from t - DELAY, Poisson kicks (shiu_worker's per-step order)."""
        var rows_t = TileTensor(self.rows, sw.LR)
        var cols_t = TileTensor(self.cols, sw.LE)
        var w_t = TileTensor(self.wts, sw.LE)
        var grp_t = TileTensor(self.grp, sw.LE)
        var scale_t = TileTensor(self.scale, sw.LSCALE)
        var rank_t = TileTensor(self.rank, sw.LR)
        var factor_t = TileTensor(self.factor, sw.LF)
        var u_t = TileTensor(self.u, sw.LS)
        var g_t = TileTensor(self.g, sw.LS)
        var last_t = TileTensor(self.last, sw.LS)
        var rfc_t = TileTensor(self.rfc, sw.LS)
        var counts_t = TileTensor(self.counts, sw.LS)
        var stim_t = TileTensor(self.stim, sw.LS)
        var key_t = TileTensor(self.key, sw.LS)
        var p_t = TileTensor(self.p, sw.LS)
        var amt_t = TileTensor(self.amt, sw.LS)
        var w0_t = TileTensor(self.w0, sw.LS)
        var w1_t = TileTensor(self.w1, sw.LS)
        var m_t = TileTensor(self.m, sw.LM)
        var modp_t = TileTensor(self.modp, sw.LMP)
        var mk_t = TileTensor(self.mk, sw.LMK)
        var ring_t = TileTensor(self.ring, sw.LRING)
        var fill_t = TileTensor(self.fill, sw.LSLOT)
        var kick_grid = max((self.stim_count + sw.BLOCK - 1) // sw.BLOCK, 1)
        for _ in range(steps):
            var slot = Int32(self.t % sw.SLOTS)
            var due = Int32((self.t + 1) % sw.SLOTS)
            ctx.enqueue_function(f_int, u_t, g_t, last_t, rfc_t, counts_t, ring_t, fill_t, m_t, mk_t, self.em, self.eg, self.c, self.vth, Int32(self.t), slot, Int32(0),
                grid_dim=(sw.S + sw.BLOCK - 1) // sw.BLOCK, block_dim=sw.BLOCK)
            ctx.enqueue_function(f_del, rows_t, cols_t, w_t, grp_t, scale_t, rank_t, factor_t, g_t, ring_t, fill_t, m_t, modp_t,
                Int32(self.t), due, Int32(1), Int32(-1), Int32(-1), Int32(0), grid_dim=sw.DGRID, block_dim=sw.BLOCK)
            ctx.enqueue_function(f_kick, stim_t, key_t, p_t, u_t, last_t, fill_t, amt_t, w0_t, w1_t, Int32(self.stim_count), self.seed, Int32(self.t),
                due, grid_dim=kick_grid, block_dim=sw.BLOCK)
            self.t += 1
