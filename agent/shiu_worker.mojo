"""Shiu et al. 2024 whole-brain LIF on the GPU; twin of ShiuBrain in agent/shiu.py.

usage: shiu_worker GRAPH_DIR/ WORK_DIR/ SEED DELAY_STEPS REFRACTORY_STEPS [resident]
GRAPH_DIR holds the out-edge CSR (row_ptr, col_idx int32; weights float32 mV). WORK_DIR holds
stim_index.bin (int32 entries, column * N + neuron), stim_p.bin (float32 Poisson probability
per step), windows.bin (int32 ascending step counts) and consts.bin (float32 em, eg, c, v_th,
poisson kick); counts.bin (int32, windows x C x N) is written there.

resident: the graph stays on the device and each stdin line "sniff" resets the state, rereads
stim_p.bin, runs to the last window and writes counts at every window for the entries in
readout.bin only (gather.bin, int32 windows x R), then prints "ok"; "q" ends. Sniff k uses
seed + k, so repeated sniffs draw fresh Poisson events.

Per step t, three launches in ShiuBrain's order: integrate + threshold + reset, push delivery
of the spikes from t - DELAY (atomic adds into g, refractory and just-spiked targets skipped),
Poisson kicks. Spikes wait in a ring of DELAY + 1 slots so step t never overwrites the slot it
delivers. A full slot is counted and fails the run; nothing is dropped silently.
"""
from std.atomic import Atomic
from std.sys import argv, get_defined_int
from std.gpu import block_idx, block_dim, thread_idx
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major

comptime N = get_defined_int["SHIU_N", 138639]()
comptime E = get_defined_int["SHIU_E", 15091983]()
comptime C = get_defined_int["SHIU_C", 1]()
comptime S = N * C
comptime SLOTS = 19
comptime CAP = 1 << 16
comptime BLOCK = 256
comptime LANES = 32  # gfx1100 wave32
comptime DGRID = 1024
comptime LS = row_major[S]()
comptime LR = row_major[N + 1]()
comptime LE = row_major[E]()
comptime LRING = row_major[SLOTS * CAP]()
comptime LSLOT = row_major[SLOTS + 1]()  # last entry counts overflowed spikes


def integrate(
    u: TileTensor[DType.float32, type_of(LS), MutAnyOrigin],
    g: TileTensor[DType.float32, type_of(LS), MutAnyOrigin],
    last: TileTensor[DType.int32, type_of(LS), MutAnyOrigin],
    rfc: TileTensor[DType.int32, type_of(LS), MutAnyOrigin],
    counts: TileTensor[DType.int32, type_of(LS), MutAnyOrigin],
    ring: TileTensor[DType.int32, type_of(LRING), MutAnyOrigin],
    fill: TileTensor[DType.int32, type_of(LSLOT), MutAnyOrigin],
    em: Float32, eg: Float32, c: Float32, v_th: Float32, t: Int32, slot: Int32,
):
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i >= S:
        return
    var since = t - last[i]
    var r = rfc[i]
    var uu = u[i]
    var gg = g[i]
    if since - 1 >= r:
        uu = uu * em + gg * c
        gg = gg * eg
    if since >= r and uu > v_th:
        uu = 0.0
        gg = 0.0
        last[i] = t
        counts[i] += 1
        var k = Atomic.fetch_add(fill.ptr.unsafe_offset(Int(slot)), Int32(1))
        if k < CAP:
            ring[Int(slot) * CAP + Int(k)] = Int32(i)
        else:
            _ = Atomic.fetch_add(fill.ptr.unsafe_offset(SLOTS), Int32(1))
    u[i] = uu
    g[i] = gg


def deliver(
    rows: TileTensor[DType.int32, type_of(LR), MutAnyOrigin],
    cols: TileTensor[DType.int32, type_of(LE), MutAnyOrigin],
    weights: TileTensor[DType.float32, type_of(LE), MutAnyOrigin],
    g: TileTensor[DType.float32, type_of(LS), MutAnyOrigin],
    last: TileTensor[DType.int32, type_of(LS), MutAnyOrigin],
    rfc: TileTensor[DType.int32, type_of(LS), MutAnyOrigin],
    ring: TileTensor[DType.int32, type_of(LRING), MutAnyOrigin],
    fill: TileTensor[DType.int32, type_of(LSLOT), MutAnyOrigin],
    t: Int32, slot: Int32,
):
    # LANES threads per spike stride over its out-edges; groups stride over the spikes. One thread
    # per spike serialised hub neurons' thousands of edges: deliver was 96% of a sniff.
    var k = block_idx.x * block_dim.x + thread_idx.x
    var lane = k % LANES
    var spikes = min(Int(fill[Int(slot)]), CAP)
    for s in range(k // LANES, spikes, DGRID * BLOCK // LANES):
        var i = Int(ring[Int(slot) * CAP + s])
        var base = (i // N) * N
        var pre = i - base
        for e in range(Int(rows[pre]) + lane, Int(rows[pre + 1]), LANES):
            var j = base + Int(cols[e])
            var lj = last[j]
            if t - lj >= rfc[j] and lj != t:
                _ = Atomic.fetch_add(g.ptr.unsafe_offset(j), weights[e])


def uniform(seed: UInt32, t: UInt32, entry: UInt32) -> Float32:
    """Murmur3 fmix32 of (seed, t, entry); twin of shiu.uniform."""
    var h = seed * 0x9E3779B9 + t * 0x85EBCA6B + entry * 0xC2B2AE35
    h ^= h >> 16
    h *= 0x85EBCA6B
    h ^= h >> 13
    h *= 0xC2B2AE35
    h ^= h >> 16
    return Float32(h >> 8) * Float32(5.9604644775390625e-08)


def kick(
    stim: TileTensor[DType.int32, type_of(LS), MutAnyOrigin],
    p: TileTensor[DType.float32, type_of(LS), MutAnyOrigin],
    u: TileTensor[DType.float32, type_of(LS), MutAnyOrigin],
    last: TileTensor[DType.int32, type_of(LS), MutAnyOrigin],
    fill: TileTensor[DType.int32, type_of(LSLOT), MutAnyOrigin],
    count: Int32, seed: UInt32, t: Int32, clear: Int32, amount: Float32,
):
    var k = block_idx.x * block_dim.x + thread_idx.x
    if k == 0:
        fill[Int(clear)] = 0  # delivered this step; integrate writes it next step
    if k >= Int(count):
        return
    var i = Int(stim[k])
    if last[i] != t and uniform(seed, UInt32(t), UInt32(i)) < p[k]:
        u[i] += amount


def gather(
    counts: TileTensor[DType.int32, type_of(LS), MutAnyOrigin],
    idx: TileTensor[DType.int32, type_of(LS), MutAnyOrigin],
    dst: TileTensor[DType.int32, type_of(LS), MutAnyOrigin],
    count: Int32, offset: Int32,
):
    var k = block_idx.x * block_dim.x + thread_idx.x
    if k < Int(count):
        dst[Int(offset) + k] = counts[Int(idx[k])]


def load(path: String, size: Int) raises -> List[UInt8]:
    """Raw little-endian file, length checked; device copies read it in place, no host copy."""
    var raw = open(path, "r").read_bytes()
    if size >= 0 and len(raw) != size * 4:
        raise Error(path + ": unexpected size")
    return raw^


def main() raises:
    var args = argv()
    if len(args) != 6 and not (len(args) == 7 and String(args[6]) == "resident"):
        raise Error("usage: shiu_worker GRAPH_DIR/ WORK_DIR/ SEED DELAY_STEPS REFRACTORY_STEPS [resident]")
    var resident = len(args) == 7
    var graph = String(args[1])
    var work = String(args[2])
    var seed = UInt32(Int(String(args[3])))
    var delay = Int(String(args[4]))
    var refractory = Int32(Int(String(args[5])))
    if delay + 1 != SLOTS:
        raise Error("ring is built for an 18-step delay")
    var rows = load(graph + "row_ptr.bin", N + 1)
    var cols = load(graph + "col_idx.bin", E)
    var weights = load(graph + "weights.bin", E)
    var stim_raw = load(work + "stim_index.bin", -1)
    var p_raw = load(work + "stim_p.bin", len(stim_raw) // 4)
    var windows_raw = load(work + "windows.bin", -1)
    var k_raw = load(work + "consts.bin", 5)
    var windows = windows_raw.unsafe_ptr().unsafe_bitcast[Int32]()
    var k = k_raw.unsafe_ptr().unsafe_bitcast[Float32]()
    var stim_count = len(stim_raw) // 4
    var window_count = len(windows_raw) // 4
    if stim_count > S or window_count == 0:
        raise Error("invalid run inputs")
    var stim = List[Int32](length=S, fill=0)
    var p = List[Float32](length=S, fill=0.0)
    var rfc = List[Int32](length=S, fill=refractory)
    for i in range(stim_count):
        stim[i] = stim_raw.unsafe_ptr().unsafe_bitcast[Int32]().unsafe_offset(i)[]
        p[i] = p_raw.unsafe_ptr().unsafe_bitcast[Float32]().unsafe_offset(i)[]
        rfc[Int(stim[i])] = 0

    var ctx = DeviceContext(0)
    var step_integrate = ctx.compile_function[integrate]()
    var step_deliver = ctx.compile_function[deliver]()
    var step_kick = ctx.compile_function[kick]()
    var row_b = ctx.enqueue_create_buffer[DType.int32](N + 1)
    var col_b = ctx.enqueue_create_buffer[DType.int32](E)
    var w_b = ctx.enqueue_create_buffer[DType.float32](E)
    var u_b = ctx.enqueue_create_buffer[DType.float32](S)
    var g_b = ctx.enqueue_create_buffer[DType.float32](S)
    var last_b = ctx.enqueue_create_buffer[DType.int32](S)
    var rfc_b = ctx.enqueue_create_buffer[DType.int32](S)
    var counts_b = ctx.enqueue_create_buffer[DType.int32](S)
    var stim_b = ctx.enqueue_create_buffer[DType.int32](S)
    var p_b = ctx.enqueue_create_buffer[DType.float32](S)
    var ring_b = ctx.enqueue_create_buffer[DType.int32](SLOTS * CAP)
    var fill_b = ctx.enqueue_create_buffer[DType.int32](SLOTS + 1)
    ctx.enqueue_copy(row_b, rows.unsafe_ptr().unsafe_bitcast[Int32]())
    ctx.enqueue_copy(col_b, cols.unsafe_ptr().unsafe_bitcast[Int32]())
    ctx.enqueue_copy(w_b, weights.unsafe_ptr().unsafe_bitcast[Float32]())
    ctx.enqueue_copy(rfc_b, rfc.unsafe_ptr())
    ctx.enqueue_copy(stim_b, stim.unsafe_ptr())
    ctx.enqueue_copy(p_b, p.unsafe_ptr())
    u_b.enqueue_fill(0.0)
    g_b.enqueue_fill(0.0)
    last_b.enqueue_fill(-1000000000)
    counts_b.enqueue_fill(0)
    fill_b.enqueue_fill(0)
    var rows_t = TileTensor(row_b, LR)
    var cols_t = TileTensor(col_b, LE)
    var w_t = TileTensor(w_b, LE)
    var u_t = TileTensor(u_b, LS)
    var g_t = TileTensor(g_b, LS)
    var last_t = TileTensor(last_b, LS)
    var rfc_t = TileTensor(rfc_b, LS)
    var counts_t = TileTensor(counts_b, LS)
    var stim_t = TileTensor(stim_b, LS)
    var p_t = TileTensor(p_b, LS)
    var ring_t = TileTensor(ring_b, LRING)
    var fill_t = TileTensor(fill_b, LSLOT)

    var step_gather = ctx.compile_function[gather]()
    var readout_raw = load(work + "readout.bin", -1) if resident else List[UInt8]()
    var readout_count = len(readout_raw) // 4
    if readout_count * window_count > S:
        raise Error("readout too large")
    var idx_b = ctx.enqueue_create_buffer[DType.int32](S)
    var got_b = ctx.enqueue_create_buffer[DType.int32](S)
    if resident and readout_count > 0:
        ctx.enqueue_copy(idx_b.create_sub_buffer[DType.int32](0, readout_count), readout_raw.unsafe_ptr().unsafe_bitcast[Int32]())
    var idx_t = TileTensor(idx_b, LS)
    var got_t = TileTensor(got_b, LS)
    var out = List[Int32](length=(window_count * readout_count) if resident else (window_count * S), fill=0)
    var kick_grid = max((Int(stim_count) + BLOCK - 1) // BLOCK, 1)
    var sniff = 0
    while True:
        if resident:
            var line = input()
            if line == "q":
                break
            var fresh = load(work + "stim_p.bin", stim_count)
            ctx.enqueue_copy(p_b.create_sub_buffer[DType.float32](0, stim_count), fresh.unsafe_ptr().unsafe_bitcast[Float32]())
            u_b.enqueue_fill(0.0)
            g_b.enqueue_fill(0.0)
            last_b.enqueue_fill(-1000000000)
            counts_b.enqueue_fill(0)
            fill_b.enqueue_fill(0)
        var window = 0
        var t = 0
        while window < window_count:
            var stop = Int(windows.unsafe_offset(window)[])
            while t < stop:
                var slot = Int32(t % SLOTS)
                var due = Int32((t + 1) % SLOTS)  # == (t - delay) mod SLOTS
                ctx.enqueue_function(step_integrate, u_t, g_t, last_t, rfc_t, counts_t, ring_t, fill_t,
                    k.unsafe_offset(0)[], k.unsafe_offset(1)[], k.unsafe_offset(2)[], k.unsafe_offset(3)[], Int32(t), slot, grid_dim=(S + BLOCK - 1) // BLOCK, block_dim=BLOCK)
                ctx.enqueue_function(step_deliver, rows_t, cols_t, w_t, g_t, last_t, rfc_t, ring_t, fill_t,
                    Int32(t), due, grid_dim=DGRID, block_dim=BLOCK)
                ctx.enqueue_function(step_kick, stim_t, p_t, u_t, last_t, fill_t, Int32(stim_count), seed + UInt32(sniff), Int32(t),
                    due, k.unsafe_offset(4)[], grid_dim=kick_grid, block_dim=BLOCK)
                t += 1
            if resident:
                ctx.enqueue_function(step_gather, counts_t, idx_t, got_t, Int32(readout_count), Int32(window * readout_count),
                    grid_dim=max((readout_count + BLOCK - 1) // BLOCK, 1), block_dim=BLOCK)
            else:
                ctx.enqueue_copy(out.unsafe_ptr().unsafe_offset(window * S), counts_b)
            window += 1
        if resident:
            ctx.enqueue_copy(out.unsafe_ptr(), got_b.create_sub_buffer[DType.int32](0, len(out)))
        var overflow = List[Int32](length=SLOTS + 1, fill=0)
        ctx.enqueue_copy(overflow.unsafe_ptr(), fill_b)
        ctx.synchronize()
        if overflow[SLOTS] != 0:
            raise Error("spike ring overflow: " + String(overflow[SLOTS]) + " spikes over capacity")
        with open(work + ("gather.bin" if resident else "counts.bin"), "w") as f:
            f.write_bytes(Span(unsafe_ptr=out.unsafe_ptr().unsafe_bitcast[UInt8](), length=len(out) * 4))
        if not resident:
            break
        print("ok", flush=True)
        sniff += 1
