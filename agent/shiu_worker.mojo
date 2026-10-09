"""Shiu et al. 2024 whole-brain LIF on the GPU; twin of ShiuBrain in agent/shiu.py.

usage: shiu_worker GRAPH_DIR/ WORK_DIR/ SEED DELAY_STEPS REFRACTORY_STEPS [resident]
GRAPH_DIR holds the out-edge CSR (row_ptr, col_idx int32; weights float32 mV). WORK_DIR holds
stim_index.bin (int32 entries, column * N + neuron), stim_p.bin (float32 Poisson probability
per step), windows.bin (int32 ascending step counts) and consts.bin (float32 em, eg, c, v_th,
poisson kick); counts.bin (int32, windows x C x N) is written there. Optional edge groups:
grp.bin (uint8 group per edge) and scale.bin (float32 C x G); an edge delivers weight x scale of its
group in the spiking cell's column, so weight variants of one topology run as columns of one graph.
Without them every edge is group 0 at scale 1.0, which leaves each weight bit-identical.
Optional plastic.bin (int32: plastic edge group, then KC neuron ids) gives every column one factor
per KC, initially 1, multiplying that KC's plastic-group edges; resident "reward" reads dopamine.bin
(float32 per column) and scales the factor of every KC that spiked in the last sniff by 1 - d.
Factors persist across sniffs (the learned state); without plastic.bin nothing changes.
Optional punish.bin (int32: a second plastic group, KC -> approach MBONs) adds a second factor
plane at offset C x KMAX; dopamine.bin then holds 2 x C floats (reward, then punishment per column).
Optional factor.bin (float32 P x C x KMAX, P planes) sets the starting factors (a saved fly);
resident "save" writes the current ones to factor.bin and prints "ok".
Optional stim_key.bin (int32 per stimulus entry) names the entry whose Poisson draws an entry uses
(default itself), so columns of different weight variants can see the same input events.
Optional cell groups: cgrp.bin (int32 group per neuron, -1 none) makes a one-shot run write sums.bin
(int32 windows x G x C, spikes per group and column) instead of counts.bin, plus gather.bin for the
cells in readout.bin when that file is there.

resident: the graph stays on the device; each stdin line "sniff" resets the state, rereads
stim_p.bin, runs to the last window and writes counts at every window for the entries in
readout.bin only (gather.bin, int32 windows x R), then prints "ok"; "advance" advances by the int32 in advance.bin without resetting voltage, conductance, spike times, delay slots, or modulators. Advance preserves one global tick and seed schedule. "q" ends. Sniff k uses
seed + k, so repeated sniffs draw fresh Poisson events.
Slow modulation (modp.bin + modk.bin, ShiuBrain.modulate): edges of a modulatory group add to the target's level
m[plane] instead of g; every step m decays by its plane's factor and the target integrates g x c x max(0, 1 + sum
k m) + c x sum b m (b: a slow depolarising (> 0) or hyperpolarising current). Levels reset with each sniff; off (no modp.bin) the arithmetic is the unmodulated model's, bit for bit. FLY_RECORD=path appends every neuron's count at every
window of every sniff to path (int32 sniffs x windows x C x N, counts since the sniff began): the full brain.
FLY_TAP=path (resident advance only; off by default, the run is unchanged): every FLY_TAP_EVERY-th advance (default 5)
writes one column's activity for a live brain view (kinestride brain_view.mojo): int32 seq, column, tick, N, then N
bytes, 255 exp(-age / 20 ms) of each neuron's last spike (0 when it never spiked or 200 ms ago); written to path.tmp
and renamed, so a reader never sees half a frame. The column is the int32 in path.sel when it exists (else 0).

Per step t, three launches in ShiuBrain's order: integrate + threshold + reset, push delivery
of the spikes from t - DELAY (atomic adds into g; integrate clears g of refractory and just-spiked
neurons, which equals skipping their arrivals and saves two scattered reads per edge;
last-spike step is int32 for continuous runs beyond 32000 ticks; refractory remains uint8,
Poisson kicks. Spikes wait in a ring of DELAY + 1 slots so step t never overwrites the slot it
delivers. A full slot is counted and fails the run; nothing is dropped silently.
"""
from std.atomic import Atomic
from std.math import exp
from std.ffi import c_ssize_t, external_call
from std.os import getenv, remove
from std.os.path import exists
from std.sys import argv, get_defined_int
from max.gpu import block_idx, block_dim, thread_idx
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
comptime GROUPS = 256  # uint8 edge groups
comptime KMAX = get_defined_int["SHIU_K", 8192]()  # plastic KCs per column (FlyWire: 5,177)
comptime DGRID = 1024
comptime LS = row_major[S]()
comptime LN = row_major[N]()
comptime LR = row_major[N + 1]()
comptime LE = row_major[E]()
comptime LSCALE = row_major[C * GROUPS]()
comptime LF = row_major[2 * C * KMAX]()
comptime LK = row_major[KMAX]()
comptime LC = row_major[2 * C]()
comptime LRING = row_major[SLOTS * CAP]()
comptime LSLOT = row_major[SLOTS + 1]()  # last entry counts overflowed spikes
comptime MP = 3  # slow modulator planes (dopamine, octopamine, serotonin)
comptime LM = row_major[MP * S]()
comptime LMP = row_major[GROUPS]()
comptime LMK = row_major[3 * MP]()


def integrate(
    u: TileTensor[DType.float32, type_of(LS), MutAnyOrigin],
    g: TileTensor[DType.float32, type_of(LS), MutAnyOrigin],
    last: TileTensor[DType.int32, type_of(LS), MutAnyOrigin],
    rfc: TileTensor[DType.uint8, type_of(LS), MutAnyOrigin],
    counts: TileTensor[DType.int32, type_of(LS), MutAnyOrigin],
    ring: TileTensor[DType.int32, type_of(LRING), MutAnyOrigin],
    fill: TileTensor[DType.int32, type_of(LSLOT), MutAnyOrigin],
    m: TileTensor[DType.float32, type_of(LM), MutAnyOrigin],
    mk: TileTensor[DType.float32, type_of(LMK), MutAnyOrigin],
    em: Float32, eg: Float32, c: Float32, v_th: Float32, t: Int32, slot: Int32, mplanes: Int32,
):
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i >= S:
        return
    var gain: Float32 = 1.0  # slow modulation: gain and drive from the levels before this step's decay
    var drive: Float32 = 0.0
    if mplanes > 0:
        var acc: Float32 = 0.0
        for q in range(Int(mplanes)):
            var level = m[q * S + i]
            acc += mk[MP + q] * level
            drive += mk[2 * MP + q] * level
            m[q * S + i] = level * mk[q]
        gain = max(1.0 + acc, 0.0)
    var uu = u[i]
    var gg = g[i]
    if uu == 0.0 and gg == 0.0 and drive == 0.0:  # at rest: stays exactly 0 and cannot spike (v_th > 0); skip 16 B of state
        return
    var since = t - last[i]
    var r = Int32(rfc[i])
    if since - 1 < r or since == 1:  # refractory, or spiked last step: the oracle drops these arrivals
        gg = 0.0                      # (deliver adds unconditionally; g is 0 here in the oracle)
    if since - 1 >= r:
        if mplanes > 0:
            uu = uu * em + gg * c * gain + drive * c
        else:
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
    grp: TileTensor[DType.uint8, type_of(LE), MutAnyOrigin],
    scale: TileTensor[DType.float32, type_of(LSCALE), MutAnyOrigin],
    rank: TileTensor[DType.int32, type_of(LR), MutAnyOrigin],
    factor: TileTensor[DType.float32, type_of(LF), MutAnyOrigin],
    g: TileTensor[DType.float32, type_of(LS), MutAnyOrigin],
    ring: TileTensor[DType.int32, type_of(LRING), MutAnyOrigin],
    fill: TileTensor[DType.int32, type_of(LSLOT), MutAnyOrigin],
    m: TileTensor[DType.float32, type_of(LM), MutAnyOrigin],
    modp: TileTensor[DType.int32, type_of(LMP), MutAnyOrigin],
    t: Int32, slot: Int32, groups: Int32, plastic: Int32, punish: Int32, mplanes: Int32,
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
        var own = (i // N) * Int(groups)  # this column's row of scales
        var learned = (i // N) * KMAX + Int(rank[pre])  # read only for plastic edges, whose pre is a KC
        for e in range(Int(rows[pre]) + lane, Int(rows[pre + 1]), LANES):
            var j = base + Int(cols[e])
            var group = Int32(grp[e])
            var wt = weights[e] * scale[own + Int(group)]
            if group == plastic:
                wt *= factor[learned]
            elif group == punish:
                wt *= factor[C * KMAX + learned]
            var q = Int(modp[Int(group)]) if mplanes > 0 else -1
            if q >= 0:  # modulatory edge: into the target's slow level, never its fast conductance
                _ = Atomic.fetch_add(m.ptr.unsafe_offset(q * S + j), wt)
            else:
                _ = Atomic.fetch_add(g.ptr.unsafe_offset(j), wt)


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
    key: TileTensor[DType.int32, type_of(LS), MutAnyOrigin],
    p: TileTensor[DType.float32, type_of(LS), MutAnyOrigin],
    u: TileTensor[DType.float32, type_of(LS), MutAnyOrigin],
    last: TileTensor[DType.int32, type_of(LS), MutAnyOrigin],
    fill: TileTensor[DType.int32, type_of(LSLOT), MutAnyOrigin],
    amt: TileTensor[DType.float32, type_of(LS), MutAnyOrigin],
    w0: TileTensor[DType.int32, type_of(LS), MutAnyOrigin],
    w1: TileTensor[DType.int32, type_of(LS), MutAnyOrigin],
    count: Int32, seed: UInt32, t: Int32, clear: Int32,
):
    var k = block_idx.x * block_dim.x + thread_idx.x
    if k == 0:
        fill[Int(clear)] = 0  # delivered this step; integrate writes it next step
    if k >= Int(count):
        return
    var t1 = w1[k]  # entry's window [t0, t1) in steps; t1 < 0: square-wave flicker of period -t1 from t0, on the first half
    if t < w0[k] or (t1 >= 0 and t >= t1) or (t1 < 0 and (t - w0[k]) % (-t1) >= (-t1) // 2):
        return
    var i = Int(stim[k])
    if last[i] != t and uniform(seed, UInt32(t), UInt32(key[k])) < p[k]:
        u[i] += amt[k]


def gather(
    counts: TileTensor[DType.int32, type_of(LS), MutAnyOrigin],
    idx: TileTensor[DType.int32, type_of(LS), MutAnyOrigin],
    dst: TileTensor[DType.int32, type_of(LS), MutAnyOrigin],
    count: Int32, offset: Int32,
):
    var k = block_idx.x * block_dim.x + thread_idx.x
    if k < Int(count):
        dst[Int(offset) + k] = counts[Int(idx[k])]


def group_sum(
    counts: TileTensor[DType.int32, type_of(LS), MutAnyOrigin],
    cgrp: TileTensor[DType.int32, type_of(LR), MutAnyOrigin],
    sums: TileTensor[DType.int32, type_of(LS), MutAnyOrigin],
    offset: Int32,
):
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i >= S:
        return
    var group = Int(cgrp[i % N])
    var spikes = counts[i]
    if group >= 0 and spikes != 0:
        _ = Atomic.fetch_add(sums.ptr.unsafe_offset(Int(offset) + group * C + i // N), spikes)


def reward(
    counts: TileTensor[DType.int32, type_of(LS), MutAnyOrigin],
    kc: TileTensor[DType.int32, type_of(LK), MutAnyOrigin],
    factor: TileTensor[DType.float32, type_of(LF), MutAnyOrigin],
    dopamine: TileTensor[DType.float32, type_of(LC), MutAnyOrigin],
    kcs: Int32, planes: Int32,
):
    var k = block_idx.x * block_dim.x + thread_idx.x
    if k >= Int(kcs) * C:
        return
    var c = k // Int(kcs)
    var n = k % Int(kcs)
    if counts[c * N + Int(kc[n])] > 0:
        for p in range(Int(planes)):
            var f = p * C * KMAX + c * KMAX + n
            factor[f] = factor[f] * (1.0 - dopamine[p * C + c])


def tap_glow(last: TileTensor[DType.int32, type_of(LS), MutAnyOrigin], tap: TileTensor[DType.uint8, type_of(LN), MutAnyOrigin], col: Int32, t: Int32):
    """FLY_TAP: one column's activity byte per neuron, 255 exp(-age / 200 ticks) of its last spike."""
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= N:
        return
    var age = Int(t) - Int(last[Int(col) * N + i])
    tap[i] = UInt8(Float32(255) * exp(-Float32(age) / 200)) if 0 <= age < 2000 else 0


def read_line() raises -> String:
    """One stdin line, newline stripped; empty at EOF. Mojo 1.0's input() leaks a duplicate of the stdin
    pipe per call: a resident worker hit 'Too many open files' after ~1,000 sniffs (one per loop step)."""
    var buf = List[UInt8]()
    var b = List[UInt8](unsafe_uninit_length=1)
    while external_call["read", c_ssize_t](0, b.unsafe_ptr(), 1) > 0 and b[0] != 10:
        buf.append(b[0])
    return String(from_utf8=Span[UInt8](buf))


def load(path: String, size: Int) raises -> List[UInt8]:
    """Raw little-endian file, length checked; device copies read it in place, no host copy."""
    var raw = open(path, "r").read_bytes()
    if size >= 0 and len(raw) != size * 4:
        raise Error(path + ": unexpected size")
    return raw^


def read_drive(work: String, count: Int, mut amt: List[Float32], mut w0: List[Int32], mut w1: List[Int32]) raises -> Bool:
    """Fill amounts and windows from stim_amount.bin / stim_window.bin when present; True when either was read."""
    var got = False
    if exists(work + "stim_amount.bin"):
        var a = load(work + "stim_amount.bin", count)
        for i in range(count):
            amt[i] = a.unsafe_ptr().unsafe_bitcast[Float32]().unsafe_offset(i)[]
        got = True
    if exists(work + "stim_window.bin"):
        var w = load(work + "stim_window.bin", 2 * count)
        for i in range(count):
            w0[i] = w.unsafe_ptr().unsafe_bitcast[Int32]().unsafe_offset(2 * i)[]
            w1[i] = w.unsafe_ptr().unsafe_bitcast[Int32]().unsafe_offset(2 * i + 1)[]
        got = True
    return got


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
    var grouped = exists(work + "scale.bin")
    var scale_raw = load(work + "scale.bin", -1) if grouped else List[UInt8]()
    var groups = len(scale_raw) // 4 // C if grouped else 1
    if grouped and (groups < 1 or groups > GROUPS or groups * C * 4 != len(scale_raw)):
        raise Error("scale.bin must hold C x G float32, G in 1..256")
    var grp_raw = open(work + "grp.bin", "r").read_bytes() if grouped else List[UInt8]()
    if grouped and len(grp_raw) != E:
        raise Error("grp.bin must hold one uint8 per edge")
    var plastic_raw = load(work + "plastic.bin", -1) if exists(work + "plastic.bin") else List[UInt8]()
    var plastic = Int32(-1)
    var kcs = 0
    var rank = List[Int32](length=N + 1, fill=0)
    if len(plastic_raw) > 0:
        var words = plastic_raw.unsafe_ptr().unsafe_bitcast[Int32]()
        plastic = words.unsafe_offset(0)[]
        kcs = len(plastic_raw) // 4 - 1
        if kcs > KMAX or plastic < 0 or Int(plastic) >= groups:
            raise Error("plastic.bin: group out of range or more than KMAX KCs")
        for n in range(kcs):
            rank[Int(words.unsafe_offset(n + 1)[])] = Int32(n)
    var punish = Int32(-1)
    if exists(work + "punish.bin"):
        punish = load(work + "punish.bin", 1).unsafe_ptr().unsafe_bitcast[Int32]()[]
        if plastic < 0 or punish < 0 or Int(punish) >= groups or punish == plastic:
            raise Error("punish.bin: needs plastic.bin and a different group in range")
    var planes = 2 if punish >= 0 else 1
    var modp = List[Int32](length=GROUPS, fill=-1)
    var mk = List[Float32](length=3 * MP, fill=0)
    var mplanes = 0
    if exists(work + "modp.bin"):  # modp.bin: int32 plane per edge group (-1 fast); modk.bin: decay, gain k, drive b (x MP each)
        var mp_raw = load(work + "modp.bin", groups)
        var mk_raw = load(work + "modk.bin", 3 * MP)
        for gi in range(groups):
            modp[gi] = mp_raw.unsafe_ptr().unsafe_bitcast[Int32]().unsafe_offset(gi)[]
            mplanes = max(mplanes, Int(modp[gi]) + 1)
        for q in range(3 * MP):
            mk[q] = mk_raw.unsafe_ptr().unsafe_bitcast[Float32]().unsafe_offset(q)[]
        if mplanes > MP or not grouped:
            raise Error("modp.bin: needs edge groups and planes 0..2")
    var stim_raw = load(work + "stim_index.bin", -1)
    var p_raw = load(work + "stim_p.bin", len(stim_raw) // 4)
    var windows_raw = load(work + "windows.bin", -1)
    var k_raw = load(work + "consts.bin", 5)
    var windows = windows_raw.unsafe_ptr().unsafe_bitcast[Int32]()
    var k = k_raw.unsafe_ptr().unsafe_bitcast[Float32]()
    var stim_count = len(stim_raw) // 4
    var key_raw = load(work + "stim_key.bin", stim_count) if exists(work + "stim_key.bin") else stim_raw.copy()
    var window_count = len(windows_raw) // 4
    if stim_count > S or window_count == 0:
        raise Error("invalid run inputs")
    var stim = List[Int32](length=S, fill=0)
    var p = List[Float32](length=S, fill=0.0)
    if refractory > 255:
        raise Error("state uses uint8 refractory steps: needs <= 255")
    var rfc = List[UInt8](length=S, fill=UInt8(refractory))
    # drive entries: optional stim_amount.bin (float32 per entry, default the Poisson kick) and stim_window.bin
    # (int32 t0, t1 per entry, steps, default always); only Poisson-kick entries drop the refractory period
    var amt = List[Float32](length=max(stim_count, 1), fill=k.unsafe_offset(4)[])  # sized to the entries, not to N x C
    var w0 = List[Int32](length=max(stim_count, 1), fill=0)
    var w1 = List[Int32](length=max(stim_count, 1), fill=Int32.MAX)
    _ = read_drive(work, stim_count, amt, w0, w1)
    for i in range(stim_count):
        stim[i] = stim_raw.unsafe_ptr().unsafe_bitcast[Int32]().unsafe_offset(i)[]
        p[i] = p_raw.unsafe_ptr().unsafe_bitcast[Float32]().unsafe_offset(i)[]
        if amt[i] == k.unsafe_offset(4)[]:
            rfc[Int(stim[i])] = 0

    var ctx = DeviceContext(0)
    var step_integrate = ctx.compile_function[integrate]()
    var step_deliver = ctx.compile_function[deliver]()
    var step_kick = ctx.compile_function[kick]()
    var row_b = ctx.enqueue_create_buffer[DType.int32](N + 1)
    var col_b = ctx.enqueue_create_buffer[DType.int32](E)
    var w_b = ctx.enqueue_create_buffer[DType.float32](E)
    var grp_b = ctx.enqueue_create_buffer[DType.uint8](E)
    var scale_b = ctx.enqueue_create_buffer[DType.float32](C * GROUPS)
    var rank_b = ctx.enqueue_create_buffer[DType.int32](N + 1)
    var factor_b = ctx.enqueue_create_buffer[DType.float32](2 * C * KMAX)
    var kc_b = ctx.enqueue_create_buffer[DType.int32](KMAX)
    var dopamine_b = ctx.enqueue_create_buffer[DType.float32](2 * C)
    ctx.enqueue_copy(rank_b, rank.unsafe_ptr())
    if exists(work + "factor.bin"):
        factor_b.enqueue_fill(1.0)
        var saved = load(work + "factor.bin", planes * C * KMAX)
        ctx.enqueue_copy(factor_b.create_sub_buffer[DType.float32](0, planes * C * KMAX), saved.unsafe_ptr().unsafe_bitcast[Float32]())
        ctx.synchronize()  # saved is freed at the end of this block
    else:
        factor_b.enqueue_fill(1.0)
    if kcs > 0:
        ctx.enqueue_copy(kc_b.create_sub_buffer[DType.int32](0, kcs), plastic_raw.unsafe_ptr().unsafe_bitcast[Int32]().unsafe_offset(1))
    var u_b = ctx.enqueue_create_buffer[DType.float32](S)
    var g_b = ctx.enqueue_create_buffer[DType.float32](S)
    var last_b = ctx.enqueue_create_buffer[DType.int32](S)
    var rfc_b = ctx.enqueue_create_buffer[DType.uint8](S)
    var counts_b = ctx.enqueue_create_buffer[DType.int32](S)
    var stim_b = ctx.enqueue_create_buffer[DType.int32](S)
    var key_b = ctx.enqueue_create_buffer[DType.int32](S)
    var p_b = ctx.enqueue_create_buffer[DType.float32](S)
    var amt_b = ctx.enqueue_create_buffer[DType.float32](max(stim_count, 1))  # kick reads index < count only
    var w0_b = ctx.enqueue_create_buffer[DType.int32](max(stim_count, 1))
    var w1_b = ctx.enqueue_create_buffer[DType.int32](max(stim_count, 1))
    var ring_b = ctx.enqueue_create_buffer[DType.int32](SLOTS * CAP)
    var fill_b = ctx.enqueue_create_buffer[DType.int32](SLOTS + 1)
    var m_b = ctx.enqueue_create_buffer[DType.float32](MP * S if mplanes > 0 else 1)  # never read when off
    var modp_b = ctx.enqueue_create_buffer[DType.int32](GROUPS)
    var mk_b = ctx.enqueue_create_buffer[DType.float32](3 * MP)
    ctx.enqueue_copy(modp_b, modp.unsafe_ptr())
    ctx.enqueue_copy(mk_b, mk.unsafe_ptr())
    m_b.enqueue_fill(0.0)
    ctx.enqueue_copy(row_b, rows.unsafe_ptr().unsafe_bitcast[Int32]())
    ctx.enqueue_copy(col_b, cols.unsafe_ptr().unsafe_bitcast[Int32]())
    ctx.enqueue_copy(w_b, weights.unsafe_ptr().unsafe_bitcast[Float32]())
    if grouped:
        ctx.enqueue_copy(grp_b, grp_raw.unsafe_ptr())
        ctx.enqueue_copy(scale_b.create_sub_buffer[DType.float32](0, C * groups), scale_raw.unsafe_ptr().unsafe_bitcast[Float32]())
    else:
        grp_b.enqueue_fill(0)
        scale_b.enqueue_fill(1.0)
    ctx.enqueue_copy(rfc_b, rfc.unsafe_ptr())
    ctx.enqueue_copy(stim_b, stim.unsafe_ptr())
    if stim_count > 0:
        ctx.enqueue_copy(key_b.create_sub_buffer[DType.int32](0, stim_count), key_raw.unsafe_ptr().unsafe_bitcast[Int32]())
    ctx.enqueue_copy(p_b, p.unsafe_ptr())
    ctx.enqueue_copy(amt_b, amt.unsafe_ptr())
    ctx.enqueue_copy(w0_b, w0.unsafe_ptr())
    ctx.enqueue_copy(w1_b, w1.unsafe_ptr())
    u_b.enqueue_fill(0.0)
    g_b.enqueue_fill(0.0)
    last_b.enqueue_fill(-1000000000)
    counts_b.enqueue_fill(0)
    fill_b.enqueue_fill(0)
    var rows_t = TileTensor(row_b, LR)
    var cols_t = TileTensor(col_b, LE)
    var w_t = TileTensor(w_b, LE)
    var grp_t = TileTensor(grp_b, LE)
    var scale_t = TileTensor(scale_b, LSCALE)
    var rank_t = TileTensor(rank_b, LR)
    var factor_t = TileTensor(factor_b, LF)
    var kc_t = TileTensor(kc_b, LK)
    var dopamine_t = TileTensor(dopamine_b, LC)
    var step_reward = ctx.compile_function[reward]()
    var u_t = TileTensor(u_b, LS)
    var g_t = TileTensor(g_b, LS)
    var last_t = TileTensor(last_b, LS)
    var rfc_t = TileTensor(rfc_b, LS)
    var counts_t = TileTensor(counts_b, LS)
    var stim_t = TileTensor(stim_b, LS)
    var key_t = TileTensor(key_b, LS)
    var p_t = TileTensor(p_b, LS)
    var amt_t = TileTensor(amt_b, LS)
    var w0_t = TileTensor(w0_b, LS)
    var w1_t = TileTensor(w1_b, LS)
    var m_t = TileTensor(m_b, LM)
    var modp_t = TileTensor(modp_b, LMP)
    var mk_t = TileTensor(mk_b, LMK)
    var ring_t = TileTensor(ring_b, LRING)
    var fill_t = TileTensor(fill_b, LSLOT)

    var step_gather = ctx.compile_function[gather]()
    var step_group_sum = ctx.compile_function[group_sum]()
    var summed = not resident and exists(work + "cgrp.bin")
    var gathered = resident or (summed and exists(work + "readout.bin"))
    var readout_raw = load(work + "readout.bin", -1) if gathered else List[UInt8]()
    var readout_count = len(readout_raw) // 4
    if readout_count * window_count > S:
        raise Error("readout too large")
    var cgrp_raw = load(work + "cgrp.bin", N) if summed else List[UInt8]()
    var group_count = 0
    for n in range(N if summed else 0):
        group_count = max(group_count, Int(cgrp_raw.unsafe_ptr().unsafe_bitcast[Int32]().unsafe_offset(n)[]) + 1)
    if group_count * C * window_count > S:
        raise Error("group sums too large")
    var cgrp_b = ctx.enqueue_create_buffer[DType.int32](N + 1)
    var sums_b = ctx.enqueue_create_buffer[DType.int32](S)
    if summed:
        ctx.enqueue_copy(cgrp_b.create_sub_buffer[DType.int32](0, N), cgrp_raw.unsafe_ptr().unsafe_bitcast[Int32]())
        sums_b.enqueue_fill(0)
    var cgrp_t = TileTensor(cgrp_b, LR)
    var sums_t = TileTensor(sums_b, LS)
    var idx_b = ctx.enqueue_create_buffer[DType.int32](S)
    var got_b = ctx.enqueue_create_buffer[DType.int32](S)
    if gathered and readout_count > 0:
        ctx.enqueue_copy(idx_b.create_sub_buffer[DType.int32](0, readout_count), readout_raw.unsafe_ptr().unsafe_bitcast[Int32]())
    var idx_t = TileTensor(idx_b, LS)
    var got_t = TileTensor(got_b, LS)
    var out = List[Int32](length=(window_count * readout_count) if gathered else (0 if summed else window_count * S), fill=0)
    var sums_out = List[Int32](length=window_count * group_count * C, fill=0)
    var kick_grid = max((Int(stim_count) + BLOCK - 1) // BLOCK, 1)
    var record = getenv("FLY_RECORD", "")
    var tap_path = getenv("FLY_TAP", "")
    var tap_every = Int(getenv("FLY_TAP_EVERY", "5"))
    var tap_b = ctx.enqueue_create_buffer[DType.uint8](N if tap_path else 1)
    var tap_host = List[UInt8](length=(N + 16) if tap_path else 0, fill=0)
    var tap_n = 0
    var rec_file = open(record if record else "/dev/null", "w")
    var rec = List[Int32](length=S if record else 0, fill=0)
    var sniff = 0
    var sim_t = 0
    while True:
        var is_advance = False
        var is_sniff = False
        if resident:
            var line = read_line()
            if line == "q" or line == "":  # empty = EOF (parent gone)
                break
            if line == "save":
                var host = List[Float32](length=planes * C * KMAX, fill=0)
                ctx.enqueue_copy(host.unsafe_ptr(), factor_b.create_sub_buffer[DType.float32](0, planes * C * KMAX))
                ctx.synchronize()
                with open(work + "factor.bin", "w") as f:
                    f.write_bytes(Span(unsafe_ptr=host.unsafe_ptr().unsafe_bitcast[UInt8](), length=len(host) * 4))
                print("ok", flush=True)
                continue
            if line == "dump":
                var hu = List[Float32](length=S, fill=0.0)
                var hg = List[Float32](length=S, fill=0.0)
                var hl = List[Int32](length=S, fill=0)
                var hc = List[Int32](length=S, fill=0)
                var hr = List[Int32](length=SLOTS * CAP, fill=0)
                var hf = List[Int32](length=SLOTS + 1, fill=0)
                var hm = List[Float32](length=max(Int(mplanes) * S, 1), fill=0.0)
                ctx.enqueue_copy(hu.unsafe_ptr(), u_b)
                ctx.enqueue_copy(hg.unsafe_ptr(), g_b)
                ctx.enqueue_copy(hl.unsafe_ptr(), last_b)
                ctx.enqueue_copy(hc.unsafe_ptr(), counts_b)
                ctx.enqueue_copy(hr.unsafe_ptr(), ring_b)
                ctx.enqueue_copy(hf.unsafe_ptr(), fill_b)
                if mplanes > 0:
                    ctx.enqueue_copy(hm.unsafe_ptr(), m_b.create_sub_buffer[DType.float32](0, Int(mplanes) * S))
                ctx.synchronize()
                with open(work + "state_u.bin", "w") as f:
                    f.write_bytes(Span(unsafe_ptr=hu.unsafe_ptr().unsafe_bitcast[UInt8](), length=len(hu) * 4))
                with open(work + "state_g.bin", "w") as f:
                    f.write_bytes(Span(unsafe_ptr=hg.unsafe_ptr().unsafe_bitcast[UInt8](), length=len(hg) * 4))
                with open(work + "state_last.bin", "w") as f:
                    f.write_bytes(Span(unsafe_ptr=hl.unsafe_ptr().unsafe_bitcast[UInt8](), length=len(hl) * 4))
                with open(work + "state_counts.bin", "w") as f:
                    f.write_bytes(Span(unsafe_ptr=hc.unsafe_ptr().unsafe_bitcast[UInt8](), length=len(hc) * 4))
                with open(work + "state_ring.bin", "w") as f:
                    f.write_bytes(Span(unsafe_ptr=hr.unsafe_ptr().unsafe_bitcast[UInt8](), length=len(hr) * 4))
                with open(work + "state_fill.bin", "w") as f:
                    f.write_bytes(Span(unsafe_ptr=hf.unsafe_ptr().unsafe_bitcast[UInt8](), length=len(hf) * 4))
                with open(work + "state_m.bin", "w") as f:
                    f.write_bytes(Span(unsafe_ptr=hm.unsafe_ptr().unsafe_bitcast[UInt8](), length=len(hm) * 4))
                print("ok", flush=True)
                continue
            if line == "reward":
                var d = load(work + "dopamine.bin", planes * C)
                ctx.enqueue_copy(dopamine_b.create_sub_buffer[DType.float32](0, planes * C), d.unsafe_ptr().unsafe_bitcast[Float32]())
                ctx.enqueue_function(step_reward, counts_t, kc_t, factor_t, dopamine_t, Int32(kcs), Int32(planes),
                    grid_dim=max((kcs * C + BLOCK - 1) // BLOCK, 1), block_dim=BLOCK)
                ctx.synchronize()
                print("ok", flush=True)
                continue
            if line == "advance":
                is_advance = True
            if line == "sniff":
                is_sniff = True
            if line != "sniff" and line != "advance":
                raise Error("resident command must be sniff, advance, dump, save, reward, or q")
            var fresh = load(work + "stim_p.bin", stim_count)
            ctx.enqueue_copy(p_b.create_sub_buffer[DType.float32](0, stim_count), fresh.unsafe_ptr().unsafe_bitcast[Float32]())
            if exists(work + "pushed"):  # ShiuLive.push changed settings: the scale row (gains, cuts, silencing)
                remove(work + "pushed")  # and drive amounts / windows; otherwise nothing is reread
                if grouped:
                    var sc = load(work + "scale.bin", C * groups)
                    ctx.enqueue_copy(scale_b.create_sub_buffer[DType.float32](0, C * groups), sc.unsafe_ptr().unsafe_bitcast[Float32]())
                if read_drive(work, stim_count, amt, w0, w1):
                    ctx.enqueue_copy(amt_b, amt.unsafe_ptr())
                    ctx.enqueue_copy(w0_b, w0.unsafe_ptr())
                    ctx.enqueue_copy(w1_b, w1.unsafe_ptr())
                ctx.synchronize()
            if is_sniff:
                u_b.enqueue_fill(0.0)
                g_b.enqueue_fill(0.0)
                last_b.enqueue_fill(-1000000000)
                counts_b.enqueue_fill(0)
                fill_b.enqueue_fill(0)
                m_b.enqueue_fill(0.0)
                sim_t = 0
        var window = 0
        var t = 0
        var stop = Int(windows.unsafe_offset(0)[])
        if resident and is_advance:
            var advance_raw = load(work + "advance.bin", 1)
            var advance_steps = Int(advance_raw.unsafe_ptr().unsafe_bitcast[Int32]().unsafe_offset(0)[])
            if advance_steps <= 0:
                raise Error("advance.bin must contain a positive tick count")
            t = sim_t
            stop = t + advance_steps
        while window < (1 if resident and is_advance else window_count):
            if not (resident and is_advance):
                stop = Int(windows.unsafe_offset(window)[])
            while t < stop:
                var slot = Int32(t % SLOTS)
                var due = Int32((t + 1) % SLOTS)  # == (t - delay) mod SLOTS
                ctx.enqueue_function(step_integrate, u_t, g_t, last_t, rfc_t, counts_t, ring_t, fill_t, m_t, mk_t,
                    k.unsafe_offset(0)[], k.unsafe_offset(1)[], k.unsafe_offset(2)[], k.unsafe_offset(3)[], Int32(t), slot, Int32(mplanes), grid_dim=(S + BLOCK - 1) // BLOCK, block_dim=BLOCK)
                ctx.enqueue_function(step_deliver, rows_t, cols_t, w_t, grp_t, scale_t, rank_t, factor_t, g_t, ring_t, fill_t, m_t, modp_t,
                    Int32(t), due, Int32(groups), plastic, punish, Int32(mplanes), grid_dim=DGRID, block_dim=BLOCK)
                ctx.enqueue_function(step_kick, stim_t, key_t, p_t, u_t, last_t, fill_t, amt_t, w0_t, w1_t, Int32(stim_count), seed + UInt32(sniff if is_sniff else 0), Int32(t),
                    due, grid_dim=kick_grid, block_dim=BLOCK)
                t += 1
            if gathered:
                ctx.enqueue_function(step_gather, counts_t, idx_t, got_t, Int32(readout_count), Int32(window * readout_count),
                    grid_dim=max((readout_count + BLOCK - 1) // BLOCK, 1), block_dim=BLOCK)
            if summed:
                ctx.enqueue_function(step_group_sum, counts_t, cgrp_t, sums_t, Int32(window * group_count * C),
                    grid_dim=(S + BLOCK - 1) // BLOCK, block_dim=BLOCK)
            elif not resident:
                ctx.enqueue_copy(out.unsafe_ptr().unsafe_offset(window * S), counts_b)
            if record:
                ctx.enqueue_copy(rec.unsafe_ptr(), counts_b)
                ctx.synchronize()
                rec_file.write_bytes(Span(unsafe_ptr=rec.unsafe_ptr().unsafe_bitcast[UInt8](), length=S * 4))
            window += 1
        if resident and is_advance:
            sim_t = t
            tap_n += 1
            if tap_path and tap_n % tap_every == 0:
                var col = 0
                if exists(tap_path + ".sel"):
                    var sel = load(tap_path + ".sel", 1)
                    col = min(max(Int(sel.unsafe_ptr().unsafe_bitcast[Int32]().unsafe_offset(0)[]), 0), C - 1)
                ctx.enqueue_function(ctx.compile_function[tap_glow](), last_t, TileTensor(tap_b, LN), Int32(col), Int32(t),
                    grid_dim=(N + BLOCK - 1) // BLOCK, block_dim=BLOCK)
                ctx.enqueue_copy(tap_host.unsafe_ptr().unsafe_offset(16), tap_b)
                ctx.synchronize()
                var hd = tap_host.unsafe_ptr().unsafe_bitcast[Int32]()
                hd[0], hd[1], hd[2], hd[3] = Int32(tap_n // tap_every), Int32(col), Int32(t), Int32(N)
                with open(tap_path + ".tmp", "w") as f:
                    f.write_bytes(Span(tap_host))
                var tmp_c = tap_path + ".tmp"
                var dst_c = tap_path.copy()
                _ = external_call["rename", Int32](tmp_c.as_c_string_slice().unsafe_ptr(), dst_c.as_c_string_slice().unsafe_ptr())
        if gathered and len(out) > 0:
            ctx.enqueue_copy(out.unsafe_ptr(), got_b.create_sub_buffer[DType.int32](0, len(out)))
        if summed:
            ctx.enqueue_copy(sums_out.unsafe_ptr(), sums_b.create_sub_buffer[DType.int32](0, len(sums_out)))
        var overflow = List[Int32](length=SLOTS + 1, fill=0)
        ctx.enqueue_copy(overflow.unsafe_ptr(), fill_b)
        ctx.synchronize()
        if overflow[SLOTS] != 0:
            raise Error("spike ring overflow: " + String(overflow[SLOTS]) + " spikes over capacity")
        if summed:
            with open(work + "sums.bin", "w") as f:
                f.write_bytes(Span(unsafe_ptr=sums_out.unsafe_ptr().unsafe_bitcast[UInt8](), length=len(sums_out) * 4))
        if gathered or not summed:
            with open(work + ("gather.bin" if gathered else "counts.bin"), "w") as f:
                f.write_bytes(Span(unsafe_ptr=out.unsafe_ptr().unsafe_bitcast[UInt8](), length=len(out) * 4))
        if not resident:
            break
        print("ok", flush=True)
        if is_sniff:
            sniff += 1
