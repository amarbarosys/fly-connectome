"""Self-wiring spider pathway on the GPU: GPU twin of agent/spider_pool_gpu.py's numpy oracle (HashPathway), the
scaffold slice of tools/spider_scaffold_learn_test.py (retina -> AM1 -> AM2 -> eye-muscle motor neurons, sparse fan-in
slots, empty at start) with its teaching rule. Every trial is one column b; all state stays on the card.

Neurons are numbered globally: retina 0..NR-1, then layer 1, 2, 3 (motor). Per neuron and column: u (float32), refr
(uint8 substeps left), spike (uint8, this substep), act (uint16 spikes this 10 ms step), tr (float32 pre trace),
ps (float32 sum of tr over the step, the eligibility). Synapse slots (int32 global pre id, float32 weight) are grouped
per layer, post-major: slot base + (j - post0) * fan + k.
Per substep, launches in order: retina (Poisson at the motion rate of the receptor's patch and direction), then each
layer (gather drive over its fan-in from this substep's spikes, Poisson noise kick, LIF, spike, trace). Random numbers:
uniform(seed + stream, substep counter, column * NT + neuron), shiu_worker's counter hash.
stdin commands (int32 code, then payload; replies on stdout):
  1 STEP  m (B x 12 float32 motion)  -> act of the motor neurons (B x NM uint16)
  2 TEACH err (B x 2 float32)          weights += ETA / B x sum_b (err_b . fb_j) [act_bj > 0, hidden layers only] ps_b[pre],
                                       clamped to +-WMAX, then prune |w| < THETA and regrow (prune, regrow kernels)
  3 RESET                              zero u, refr, tr (a new trial)
  4 GETW                               -> w (NS float32), idx (NS int32)
  5 SETW  w, idx                       (host rewiring between trials)
  0 quit
usage: spider_pool DIR/ SEED   (DIR: idx.bin, w.bin, fb.bin (NT x 2 float32), ch.bin (NR int32: patch * 2 + dir))
build: mojo build -DB=.. -DNR=.. -DN1=.. -DN2=.. -DNM=.. -DF12=.. -DFM=.. agent/spider_pool.mojo
"""
from std.atomic import Atomic
from std.ffi import external_call
from std.math import exp
from std.sys import argv, get_defined_int
from std.gpu import block_idx, block_dim, thread_idx
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major

comptime B = get_defined_int["B", 32]()
comptime NR = get_defined_int["NR", 1200]()
comptime N1 = get_defined_int["N1", 2000]()
comptime N2 = get_defined_int["N2", 2000]()
comptime NM = get_defined_int["NM", 12]()
comptime F12 = get_defined_int["F12", 20]()
comptime FM = get_defined_int["FM", 200]()
comptime NT = NR + N1 + N2 + NM
comptime NS = N1 * F12 + N2 * F12 + NM * FM
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
comptime LK = row_major[8]()
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
    var b = i // NR
    var k = i % NR
    var c = Int(ch[k])
    var v = m[b * 12 + c // 2]
    var rate = GAIN * SUB * (max(v, Float32(0)) if c % 2 == 0 else max(-v, Float32(0)))
    var e = b * NT + k
    var s = UInt8(1) if uniform(seed, t, UInt32(e)) < rate else UInt8(0)
    spk[e] = s
    trace(tr, ps, e, s, dk)


def layer(idx: TSi, w: TSf, u: TNf, refr: TNb, spk: TNb, act: TNa, tr: TNf, ps: TNf,
          post0_: Int32, npost_: Int32, fan_: Int32, sbase_: Int32, seed: UInt32, t: UInt32, decay: Float32, dk: Float32):
    var post0, npost, fan, sbase = Int(post0_), Int(npost_), Int(fan_), Int(sbase_)
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= B * npost:
        return
    var b = i // npost
    var j = i % npost
    var e = b * NT + post0 + j
    var drive = Float32(0)
    var s0 = sbase + j * fan
    for k in range(fan):
        drive += w[s0 + k] * Float32(spk[b * NT + Int(idx[s0 + k])])
    var noise = KICK if uniform(seed + 1, t, UInt32(e)) < P_NOISE else Float32(0)
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


def teach(idx: TSi, w: TSf, act: TNa, ps: TNf, fb: TF, err: TE, post0_: Int32, npost_: Int32, fan_: Int32, sbase_: Int32, gate: Int32):
    var post0, npost, fan, sbase = Int(post0_), Int(npost_), Int(fan_), Int(sbase_)
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= npost * fan:
        return
    var j = i // fan
    var g = post0 + j
    var pre = Int(idx[sbase + i])
    var acc = Float32(0)
    for b in range(B):
        if gate == 0 or act[b * NT + g] > 0:  # motor layer ungated (as the pool test's w_out): a silent motor can recover
            acc += (err[2 * b] * fb[2 * g] + err[2 * b + 1] * fb[2 * g + 1]) * ps[b * NT + pre]
    w[sbase + i] = min(max(w[sbase + i] + ETA * Float32(F12) / Float32(fan) * acc / Float32(B), -WMAX), WMAX)  # bounded every step, as the CPU slice


def prune(w: TSf, alive: TK, l: Int32, n_: Int32, sbase_: Int32):
    """Structural pass 1 (every teach step, all slots of layer l): |w| < THETA -> 0; count the survivors."""
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= Int(n_):
        return
    var s = Int(sbase_) + i
    if abs(w[s]) < THETA:
        w[s] = 0
    else:
        _ = Atomic.fetch_add(alive.ptr.unsafe_offset(Int(l)), Int32(1))


def regrow(idx: TSi, w: TSf, alive: TK, l: Int32, n_: Int32, sbase_: Int32, pre_lo: Int32, pre_n: Int32, seed: UInt32, step: UInt32):
    """Structural pass 2: each empty slot regrows with probability (FRAC n - alive) / (n - alive), at a new random pre
    of the same tract, weight +-1.5 THETA; random numbers from the counter hash (stream seed + 2)."""
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    var n = Int(n_)
    if i >= n:
        return
    var s = Int(sbase_) + i
    if w[s] != 0:
        return
    var a = Int(alive[Int(l)])
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
    var t = UInt32(0)
    var code = ctx.enqueue_create_host_buffer[DType.int32](1)
    var alive_b = ctx.enqueue_create_buffer[DType.int32](8)
    var kp = ctx.compile_function[prune]()
    var kg = ctx.compile_function[regrow]()
    var tstep = UInt32(0)
    var PL = [0, NR, NR + N1]
    var PN = [NR, N1, N2]
    # per layer: post0, npost, fan, slot base
    var P0 = [NR, NR + N1, NR + N1 + N2]
    var NP = [N1, N2, NM]
    var FN = [F12, F12, FM]
    var SB = [0, N1 * F12, N1 * F12 + N2 * F12]
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
                for l in range(3):
                    ctx.enqueue_function(kl, idx, w, u, refr, spk, act, tr, ps, Int32(P0[l]), Int32(NP[l]), Int32(FN[l]), Int32(SB[l]), seed, t, decay, dk,
                                         grid_dim=(B * NP[l] + 255) // 256, block_dim=256)
                t += 1
            ctx.enqueue_copy(act_h, act_b)
            ctx.synchronize()
            var ap = act_h.unsafe_ptr()
            var op = out_h.unsafe_ptr()
            for b in range(B):
                for j in range(NM):
                    op[b * NM + j] = ap[b * NT + NR + N1 + N2 + j]
            if not io(1, out_h.unsafe_ptr().unsafe_bitcast[UInt8](), B * NM * 2, True):
                return
        elif code.unsafe_ptr()[] == 2:
            if not io(0, err_h.unsafe_ptr().unsafe_bitcast[UInt8](), B * 2 * 4, False):
                return
            ctx.enqueue_copy(err_b, err_h)
            for l in range(3):
                ctx.enqueue_function(kt, idx, w, act, ps, TileTensor(fb_b, LF), TileTensor(err_b, LE), Int32(P0[l]), Int32(NP[l]), Int32(FN[l]), Int32(SB[l]), Int32(1 if l < 2 else 0),
                                     grid_dim=(NP[l] * FN[l] + 255) // 256, block_dim=256)
            alive_b.enqueue_fill(0)
            for l in range(3):
                ctx.enqueue_function(kp, w, TileTensor(alive_b, LK), Int32(l), Int32(NP[l] * FN[l]), Int32(SB[l]),
                                     grid_dim=(NP[l] * FN[l] + 255) // 256, block_dim=256)
            for l in range(3):
                ctx.enqueue_function(kg, idx, w, TileTensor(alive_b, LK), Int32(l), Int32(NP[l] * FN[l]), Int32(SB[l]),
                                     Int32(PL[l]), Int32(PN[l]), seed, tstep, grid_dim=(NP[l] * FN[l] + 255) // 256, block_dim=256)
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
