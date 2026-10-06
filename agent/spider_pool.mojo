"""Self-wiring spider pathway on the GPU: GPU twin of agent/spider_pool_gpu.py's numpy oracle (HashPathway), the
scaffold slice of tools/spider_scaffold_learn_test.py (retina -> AM1 -> AM2 -> eye-muscle motor neurons, sparse fan-in
slots, empty at start) with its teaching rule. Every trial is one column b; all state stays on the card.

Neurons are numbered globally: retinas 0..NR-1, then the groups (agent/spider_scaffold.py plan(): one group per
region-side, launched in tract topological order; the 3-layer slice is the same with 3 groups). Per neuron and column: u (float32), refr
(uint8 substeps left), spike (uint8, this substep), act (uint16 spikes this 10 ms step), tr (float32 pre trace),
ps (float32 sum of tr over the step, the eligibility). Synapse slots (int32 global pre id, float32 weight) are grouped
per group, post-major: slot base + (j - post0) * row fan + k; a row is its incoming tracts' segments in turn (segment
k // segment fan), each with its own contiguous pre pool for regrowth.
Per substep, launches in order: retina (Poisson at the motion rate of the receptor's patch and direction), then one
layer launch per dependency level (groups whose pre pools include no group of the same level; gather drive over the
fan-in from this substep's spikes, Poisson noise kick, LIF, spike, trace). TEACH, prune and regrow: one launch each. Random numbers:
uniform(seed + stream, substep counter, column * NT + neuron), shiu_worker's counter hash. State arrays are
neuron-major (address neuron * B + column) so the B columns of one neuron are contiguous for the spike gather.
stdin commands (int32 code, then payload; replies on stdout):
  1 STEP  m (B x 12 float32 motion)  -> act of the motor neurons (B x NM uint16)
  2 TEACH err (B x 2 float32)          weights += ETA / B x sum_b (err_b . fb_j) [act_bj > 0, hidden layers only] ps_b[pre],
                                       clamped to +-WMAX, then prune |w| < THETA and regrow (prune, regrow kernels)
  3 RESET                              zero u, refr, tr (a new trial)
  4 GETW                               -> w (NS float32), idx (NS int32)
  5 SETW  w, idx                       (host rewiring between trials)
  0 quit
usage: spider_pool DIR/ SEED   (DIR: idx.bin, w.bin, fb.bin (NT x 2 float32), ch.bin (NR int32: patch * 2 + dir, -1 silent),
       plan.bin (int32: NG x 7 group rows, then NSEG x 2 segment rows, see spider_scaffold.plan))
build: mojo build -DB=.. -DNR=.. -DNT=.. -DNS=.. -DNG=.. -DNSEG=.. -DOUT0=.. -DNM=.. agent/spider_pool.mojo
"""
from std.atomic import Atomic
from std.ffi import external_call
from std.math import exp
from std.sys import argv, get_defined_int
from max.gpu import block_idx, block_dim, thread_idx
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major

comptime B = get_defined_int["B", 32]()
comptime NR = get_defined_int["NR", 1200]()
comptime NT = get_defined_int["NT", 5212]()
comptime NS = get_defined_int["NS", 82400]()
comptime NG = get_defined_int["NG", 3]()
comptime NSEG = get_defined_int["NSEG", 3]()
comptime OUT0 = get_defined_int["OUT0", 5200]()  # first motor neuron read out by STEP
comptime NM = get_defined_int["NM", 12]()
comptime FREF = 20  # fan-in the teaching rate is set for; a row of fan f learns at ETA * FREF / f
comptime S = 20  # substeps of 0.5 ms per 10 ms step
comptime REFR = UInt8(5)  # 2.2 ms refractory at 0.5 ms substeps: blocks 5 substeps, as the CPU slice's float countdown
comptime SUB = Float32(5e-4)
comptime V_TH = Float32(7.0)
comptime KICK = Float32(0.275 * 250)  # W_SYN x F_POI
comptime GAIN = Float32(400.0)
comptime P_NOISE = Float32(30.0 * 5e-4)
comptime ETA = Float32(0.01)
comptime WMAX = Float32(0.3)
comptime THETA = Float32(0.02)
comptime FRAC = Float32(0.3)
comptime LK = row_major[NSEG]()
comptime LG = row_major[2 * NSEG]()
comptime TG = TileTensor[DType.int32, type_of(LG), MutAnyOrigin]
comptime TK = TileTensor[DType.int32, type_of(LK), MutAnyOrigin]
comptime LN = row_major[B * NT]()
comptime LS = row_major[NS]()
comptime LF = row_major[NT * 2]()
comptime LM = row_major[B * 12]()
comptime LE = row_major[B * 2]()
comptime LC = row_major[NR]()
comptime TNf = TileTensor[DType.float32, type_of(LN), MutAnyOrigin]
comptime TNb = TileTensor[DType.uint8, type_of(LN), MutAnyOrigin]
comptime TNa = TileTensor[DType.uint16, type_of(LN), MutAnyOrigin]
comptime TSi = TileTensor[DType.int32, type_of(LS), MutAnyOrigin]
comptime TSf = TileTensor[DType.float32, type_of(LS), MutAnyOrigin]
comptime TF = TileTensor[DType.float32, type_of(LF), MutAnyOrigin]
comptime TM = TileTensor[DType.float32, type_of(LM), MutAnyOrigin]
comptime TE = TileTensor[DType.float32, type_of(LE), MutAnyOrigin]
comptime TC = TileTensor[DType.int32, type_of(LC), MutAnyOrigin]
comptime LP = row_major[NG * 8]()  # group table on the card: 7 plan columns per group, then NG group ids ordered by level
comptime TP = TileTensor[DType.int32, type_of(LP), MutAnyOrigin]


def uniform(seed: UInt32, t: UInt32, entry: UInt32) -> Float32:
    """Murmur3 fmix32 of (seed, t, entry); twin of shiu.uniform."""
    var h = seed * 0x9E3779B9 + t * 0x85EBCA6B + entry * 0xC2B2AE35
    h ^= h >> 16
    h *= 0x85EBCA6B
    h ^= h >> 13
    h *= 0xC2B2AE35
    h ^= h >> 16
    return Float32(h >> 8) * Float32(5.9604644775390625e-08)


def trace(tr: TNf, ps: TNf, e: Int, s: UInt8, dk: Float32):
    var x = tr[e] * dk + Float32(s)
    tr[e] = x
    ps[e] += x


def retina(m: TM, ch: TC, spk: TNb, tr: TNf, ps: TNf, seed: UInt32, t: UInt32, dk: Float32):
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= B * NR:
        return
    var b = i % B  # neuron-major state [neuron][column]: a wave holds the B columns of one neuron
    var k = i // B
    var e = k * B + b
    var c = Int(ch[k])
    if c < 0:
        spk[e] = 0
        trace(tr, ps, e, 0, dk)
        return
    var v = m[b * 12 + c // 2]
    var rate = GAIN * SUB * (max(v, Float32(0)) if c % 2 == 0 else max(-v, Float32(0)))
    var s = UInt8(1) if uniform(seed, t, UInt32(b * NT + k)) < rate else UInt8(0)  # hash entry: column * NT + neuron
    spk[e] = s
    trace(tr, ps, e, s, dk)


def pick(gt: TP, l0: Int32, nl: Int32, mul: Int, mut i: Int) -> Int:
    """Group of fused thread i among the level's groups gt[7 NG + l0 ..][:nl] (work per group = npost x mul, mul B or
    the row fan); i becomes the index inside that group. -1 past the end."""
    for q in range(Int(nl)):
        var g = Int(gt[7 * NG + Int(l0) + q])
        var n = Int(gt[7 * g + 1]) * (mul if mul > 0 else Int(gt[7 * g + 2]))
        if i < n:
            return g
        i -= n
    return -1


def layer(idx: TSi, w: TSf, u: TNf, refr: TNb, spk: TNb, act: TNa, tr: TNf, ps: TNf,
          gt: TP, l0: Int32, nl: Int32, seed: UInt32, t: UInt32, decay: Float32, dk: Float32):
    """One launch per dependency level (groups of a level read no group of the same level this substep)."""
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    var g = pick(gt, l0, nl, B, i)
    if g < 0:
        return
    var post0, npost, fan, sbase = Int(gt[7 * g]), Int(gt[7 * g + 1]), Int(gt[7 * g + 2]), Int(gt[7 * g + 3])
    var b = i % B
    var j = i // B
    var e = (post0 + j) * B + b
    var drive = Float32(0)
    var s0 = sbase + j * fan
    for k in range(fan):
        drive += w[s0 + k] * Float32(spk[Int(idx[s0 + k]) * B + b])  # w, idx uniform across the wave; spk contiguous
    var noise = KICK if uniform(seed + 1, t, UInt32(b * NT + post0 + j)) < P_NOISE else Float32(0)
    var s = UInt8(0)
    if refr[e] > 0:
        refr[e] = refr[e] - 1
        u[e] = 0
    else:
        var x = u[e] * decay + drive * KICK + noise
        if x > V_TH:
            s = 1
            x = 0
            refr[e] = REFR
        u[e] = x
    spk[e] = s
    act[e] = act[e] + UInt16(s)
    trace(tr, ps, e, s, dk)


def teach(idx: TSi, w: TSf, act: TNa, ps: TNf, fb: TF, err: TE, gt: TP):
    """All groups in one launch (no group reads another's weights)."""
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    var g0 = pick(gt, 0, Int32(NG), 0, i)
    if g0 < 0:
        return
    var post0, fan, sbase, gate = Int(gt[7 * g0]), Int(gt[7 * g0 + 2]), Int(gt[7 * g0 + 3]), Int(gt[7 * g0 + 6])
    var j = i // fan
    var g = post0 + j
    var pre = Int(idx[sbase + i])
    var acc = Float32(0)
    for b in range(B):
        if gate == 0 or act[g * B + b] > 0:  # motor layer ungated (as the pool test's w_out): a silent motor can recover
            acc += (err[2 * b] * fb[2 * g] + err[2 * b + 1] * fb[2 * g + 1]) * ps[pre * B + b]
    w[sbase + i] = min(max(w[sbase + i] + ETA * Float32(FREF) / Float32(fan) * acc / Float32(B), -WMAX), WMAX)  # bounded every step, as the CPU slice


def prune(w: TSf, alive: TK, gt: TP):
    """Structural pass 1 (every teach step, all slots of all groups): |w| < THETA -> 0; count survivors per segment."""
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    var g = pick(gt, 0, Int32(NG), 0, i)
    if g < 0:
        return
    var s = Int(gt[7 * g + 3]) + i
    if abs(w[s]) < THETA:
        w[s] = 0
    else:
        _ = Atomic.fetch_add(alive.ptr.unsafe_offset(Int(gt[7 * g + 5]) + i % Int(gt[7 * g + 2]) // Int(gt[7 * g + 4])), Int32(1))


def regrow(idx: TSi, w: TSf, alive: TK, gt: TP, segs: TG, seed: UInt32, step: UInt32):
    """Structural pass 2: each empty slot regrows with probability (FRAC n - alive) / (n - alive), at a new random pre
    of the same tract, weight +-1.5 THETA; random numbers from the counter hash (stream seed + 2)."""
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    var gr = pick(gt, 0, Int32(NG), 0, i)
    if gr < 0:
        return
    var rowfan, segfan = Int(gt[7 * gr + 2]), Int(gt[7 * gr + 4])
    var s = Int(gt[7 * gr + 3]) + i
    if w[s] != 0:
        return
    var g = Int(gt[7 * gr + 5]) + i % rowfan // segfan
    var n = Int(gt[7 * gr + 1]) * segfan  # slots of this segment
    var pre_lo, pre_n = segs[2 * g], segs[2 * g + 1]
    var a = Int(alive[g])
    var q = (FRAC * Float32(n) - Float32(a)) / Float32(max(n - a, 1))
    if uniform(seed + 2, step, UInt32(s)) < q:
        var pick = uniform(seed + 3, step, UInt32(s))
        idx[s] = pre_lo + min(Int32(pick * Float32(pre_n)), pre_n - 1)
        w[s] = Float32(1.5) * THETA if uniform(seed + 4, step, UInt32(s)) < 0.5 else Float32(-1.5) * THETA


def io(fd: Int, p: Pointer[UInt8, MutUntrackedOrigin], n: Int, write: Bool) -> Bool:
    var left = n
    var q = p
    while left > 0:
        var r = external_call["write", Int](fd, q.unsafe_bitcast[NoneType](), left) if write else external_call["read", Int](fd, q.unsafe_bitcast[NoneType](), left)
        if r <= 0:
            return False
        q = q.unsafe_offset(r)
        left -= r
    return True


def load(path: String, size: Int) raises -> List[UInt8]:
    var raw = open(path, "r").read_bytes()
    if len(raw) != size * 4:
        raise Error(path + ": expected " + String(size) + " words, got " + String(len(raw) // 4))
    return raw^


def main() raises:
    var a = argv()
    var dir = String(a[1])
    var seed = UInt32(Int(String(a[2])))
    _ = external_call["setenv", Int32]("MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT".unsafe_ptr(), "0".unsafe_ptr(), Int32(0))
    var ctx = DeviceContext(0)
    var idx_b = ctx.enqueue_create_buffer[DType.int32](NS)
    var w_b = ctx.enqueue_create_buffer[DType.float32](NS)
    var fb_b = ctx.enqueue_create_buffer[DType.float32](NT * 2)
    var ch_b = ctx.enqueue_create_buffer[DType.int32](NR)
    var u_b = ctx.enqueue_create_buffer[DType.float32](B * NT)
    var tr_b = ctx.enqueue_create_buffer[DType.float32](B * NT)
    var ps_b = ctx.enqueue_create_buffer[DType.float32](B * NT)
    var refr_b = ctx.enqueue_create_buffer[DType.uint8](B * NT)
    var spk_b = ctx.enqueue_create_buffer[DType.uint8](B * NT)
    var act_b = ctx.enqueue_create_buffer[DType.uint16](B * NT)
    var m_b = ctx.enqueue_create_buffer[DType.float32](B * 12)
    var err_b = ctx.enqueue_create_buffer[DType.float32](B * 2)
    var idx_h = ctx.enqueue_create_host_buffer[DType.int32](NS)
    var w_h = ctx.enqueue_create_host_buffer[DType.float32](NS)
    var m_h = ctx.enqueue_create_host_buffer[DType.float32](B * 12)
    var err_h = ctx.enqueue_create_host_buffer[DType.float32](B * 2)
    var act_h = ctx.enqueue_create_host_buffer[DType.uint16](B * NT)
    var out_h = ctx.enqueue_create_host_buffer[DType.uint16](B * NM)
    var raw = load(dir + "idx.bin", NS)
    ctx.enqueue_copy(idx_b, raw.unsafe_ptr().unsafe_bitcast[Int32]())
    raw = load(dir + "w.bin", NS)
    ctx.enqueue_copy(w_b, raw.unsafe_ptr().unsafe_bitcast[Float32]())
    raw = load(dir + "fb.bin", NT * 2)
    ctx.enqueue_copy(fb_b, raw.unsafe_ptr().unsafe_bitcast[Float32]())
    raw = load(dir + "ch.bin", NR)
    ctx.enqueue_copy(ch_b, raw.unsafe_ptr().unsafe_bitcast[Int32]())
    u_b.enqueue_fill(0)
    tr_b.enqueue_fill(0)
    ps_b.enqueue_fill(0)
    refr_b.enqueue_fill(0)
    spk_b.enqueue_fill(0)
    act_b.enqueue_fill(0)
    var kr = ctx.compile_function[retina]()
    var kl = ctx.compile_function[layer]()
    var kt = ctx.compile_function[teach]()
    var idx = TileTensor(idx_b, LS)
    var w = TileTensor(w_b, LS)
    var u = TileTensor(u_b, LN)
    var tr = TileTensor(tr_b, LN)
    var ps = TileTensor(ps_b, LN)
    var refr = TileTensor(refr_b, LN)
    var spk = TileTensor(spk_b, LN)
    var act = TileTensor(act_b, LN)
    var decay = exp(-SUB / Float32(20e-3))
    var dk = exp(-SUB / Float32(5e-3))
    var t = UInt32(Int(String(a[3]))) if len(a) > 3 else UInt32(0)  # a restart continues the substep and teach counters,
    var code = ctx.enqueue_create_host_buffer[DType.int32](1)
    var alive_b = ctx.enqueue_create_buffer[DType.int32](NSEG)
    var kp = ctx.compile_function[prune]()
    var kg = ctx.compile_function[regrow]()
    var tstep = UInt32(Int(String(a[4]))) if len(a) > 4 else UInt32(0)  # so its noise and regrow streams do not replay from 0
    raw = load(dir + "plan.bin", NG * 7 + NSEG * 2)
    var pl = raw.unsafe_ptr().unsafe_bitcast[Int32]()
    var P0 = List[Int]()
    var NP = List[Int]()
    var FN = List[Int]()
    var SB = List[Int]()
    var SF = List[Int]()
    var S0 = List[Int]()
    for g in range(NG):  # group rows: post0, npost, row fan, slot base, segment fan, first segment, teach gate
        P0.append(Int(pl[7 * g])); NP.append(Int(pl[7 * g + 1])); FN.append(Int(pl[7 * g + 2])); SB.append(Int(pl[7 * g + 3]))
        SF.append(Int(pl[7 * g + 4])); S0.append(Int(pl[7 * g + 5]))
    var segs_b = ctx.enqueue_create_buffer[DType.int32](2 * NSEG)
    ctx.enqueue_copy(segs_b, pl.unsafe_offset(7 * NG))
    # dependency levels: a group reads the groups its segments' pre pools overlap; level = 1 + its deepest such group.
    # A read of a LATER group would be last substep's spikes in the per-group order: refused, fusion would change it.
    var LV = List[Int]()
    var nlev = 0
    for g in range(NG):
        var lv = 1
        for k in range(S0[g], S0[g] + FN[g] // SF[g]):
            var lo, n = Int(pl[7 * NG + 2 * k]), Int(pl[7 * NG + 2 * k + 1])
            for h in range(NG):
                if lo < P0[h] + NP[h] and P0[h] < lo + n:
                    if h >= g:
                        raise Error("group " + String(g) + " reads group " + String(h) + ": not acyclic in plan order")
                    lv = max(lv, LV[h] + 1)
        LV.append(lv)
        nlev = max(nlev, lv)
    var gt_h = ctx.enqueue_create_host_buffer[DType.int32](NG * 8)
    var gp = gt_h.unsafe_ptr()
    for q in range(7 * NG):
        gp[q] = pl[q]
    var L0 = List[Int]()
    var NL = List[Int]()
    var LW = List[Int]()  # threads per level launch (B x npost summed)
    var c = 0
    for l in range(1, nlev + 1):
        L0.append(c)
        var wk = 0
        for g in range(NG):
            if LV[g] == l:
                gp[7 * NG + c] = Int32(g); c += 1; wk += B * NP[g]
        NL.append(c - L0[len(L0) - 1]); LW.append(wk)
    var nslot = 0
    for g in range(NG):
        nslot += NP[g] * FN[g]
    var gt_b = ctx.enqueue_create_buffer[DType.int32](NG * 8)
    ctx.enqueue_copy(gt_b, gt_h)
    var gt = TileTensor(gt_b, LP)
    var gall_h = ctx.enqueue_create_host_buffer[DType.int32](NG * 8)  # same table, ids in plan order (teach, prune, regrow)
    for q in range(7 * NG):
        gall_h.unsafe_ptr()[q] = pl[q]
    for g in range(NG):
        gall_h.unsafe_ptr()[7 * NG + g] = Int32(g)
    var gall_b = ctx.enqueue_create_buffer[DType.int32](NG * 8)
    ctx.enqueue_copy(gall_b, gall_h)
    var gall = TileTensor(gall_b, LP)
    ctx.synchronize()
    while True:
        if not io(0, code.unsafe_ptr().unsafe_bitcast[UInt8](), 4, False) or code.unsafe_ptr()[] == 0:
            return
        if code.unsafe_ptr()[] == 1:
            if not io(0, m_h.unsafe_ptr().unsafe_bitcast[UInt8](), B * 12 * 4, False):
                return
            ctx.enqueue_copy(m_b, m_h)
            ps_b.enqueue_fill(0)
            act_b.enqueue_fill(0)
            for _ in range(S):
                ctx.enqueue_function(kr, TileTensor(m_b, LM), TileTensor(ch_b, LC), spk, tr, ps, seed, t, dk,
                                     grid_dim=(B * NR + 255) // 256, block_dim=256)
                for l in range(nlev):
                    ctx.enqueue_function(kl, idx, w, u, refr, spk, act, tr, ps, gt, Int32(L0[l]), Int32(NL[l]), seed, t, decay, dk,
                                         grid_dim=(LW[l] + 255) // 256, block_dim=256)
                t += 1
            ctx.enqueue_copy(act_h, act_b)
            ctx.synchronize()
            var ap = act_h.unsafe_ptr()
            var op = out_h.unsafe_ptr()
            for b in range(B):
                for j in range(NM):
                    op[b * NM + j] = ap[(OUT0 + j) * B + b]
            if not io(1, out_h.unsafe_ptr().unsafe_bitcast[UInt8](), B * NM * 2, True):
                return
        elif code.unsafe_ptr()[] == 2:
            if not io(0, err_h.unsafe_ptr().unsafe_bitcast[UInt8](), B * 2 * 4, False):
                return
            ctx.enqueue_copy(err_b, err_h)
            ctx.enqueue_function(kt, idx, w, act, ps, TileTensor(fb_b, LF), TileTensor(err_b, LE), gall,
                                 grid_dim=(nslot + 255) // 256, block_dim=256)
            alive_b.enqueue_fill(0)
            ctx.enqueue_function(kp, w, TileTensor(alive_b, LK), gall, grid_dim=(nslot + 255) // 256, block_dim=256)
            ctx.enqueue_function(kg, idx, w, TileTensor(alive_b, LK), gall, TileTensor(segs_b, LG), seed, tstep,
                                 grid_dim=(nslot + 255) // 256, block_dim=256)
            tstep += 1
            ctx.synchronize()
        elif code.unsafe_ptr()[] == 3:
            u_b.enqueue_fill(0)
            tr_b.enqueue_fill(0)
            refr_b.enqueue_fill(0)
            ctx.synchronize()
        elif code.unsafe_ptr()[] == 4:
            ctx.enqueue_copy(w_h, w_b)
            ctx.enqueue_copy(idx_h, idx_b)
            ctx.synchronize()
            if not io(1, w_h.unsafe_ptr().unsafe_bitcast[UInt8](), NS * 4, True) or not io(1, idx_h.unsafe_ptr().unsafe_bitcast[UInt8](), NS * 4, True):
                return
        elif code.unsafe_ptr()[] == 5:
            if not io(0, w_h.unsafe_ptr().unsafe_bitcast[UInt8](), NS * 4, False) or not io(0, idx_h.unsafe_ptr().unsafe_bitcast[UInt8](), NS * 4, False):
                return
            ctx.enqueue_copy(w_b, w_h)
            ctx.enqueue_copy(idx_b, idx_h)
            ctx.synchronize()
