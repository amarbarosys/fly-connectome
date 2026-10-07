"""Fly gait on the card (docs/plans/BRAIN_BODY_LOOP.md step 3, `live.mojo -DBRAIN=3`): flygym_demo's HybridTurningController for zero observations
(no retraction, no stumbling: flyloop's mapping, tools/fly_gait_oracle.py), one thread, float64, no host copy.

Per control step (timestep dt = K x 0.1 ms) with descending drives (left, right):
  hybrid_controller.py / turning_controller.py:  amps = |drive| per side (x3 legs), freqs = 12 Hz x sign(drive) per side
  cpg_controller.py calculate_ddt, CPGNetwork.step:  dtheta_i = 2 pi nu_i + sum_j r_j w_ij sin(theta_j - theta_i - phi_ij), dr_i = alpha (R_i - r_i),
      Euler with dt; tripod: phi_ij = pi on the odd-parity pairs, w = 10 there, alpha = 20
  preprogrammed.py get_joint_angles:  angles = neutral + r x (spline(theta mod 2 pi) - neutral), the periodic cubic spline given as its 44 polynomial
      segments per dof (scipy CubicSpline .c), 7 dofs per leg;  adhesion = not (0 < theta mod 2 pi < swing_end + pi / 4)  (hybrid_controller _get_adhesion_onoff)
Writes the 42 leg servo targets and the 6 adhesion values into the engine's ctrl buffer at the indices of gait.idx (every other servo keeps its
value, the two head servos hold). tab = per leg: 4 x NSEG x 7 coefficients (c3 dx^3 + c2 dx^2 + c1 dx + c0 order as scipy: c[0] highest), 7 neutral values,
swing_end; state = 6 phases, 6 magnitudes, 2 slewed drives (fly_link.Gait: the drive moves at most 1 per SLEW_S, then x (1 - TRIM, 1 + TRIM)
left/right; the Python side carries the same two values in Gait.sig).
"""
from max.gpu import thread_idx
from std.sys import get_defined_int
from layout import TileTensor, row_major
from smooth import F, NU, TU

comptime NSEG = get_defined_int["GSEG", 44]()
comptime NLEG = 6
comptime NDOF = 7
comptime LEGSZ = 4 * NSEG * NDOF + NDOF + 1
comptime TWO_PI = 6.283185307179586
comptime LTAB = row_major[NLEG * LEGSZ]()
comptime NST = 2 * NLEG + 2
comptime LST = row_major[NST]()
comptime LIDX = row_major[NLEG * NDOF + NLEG]()
comptime LDR = row_major[2]()
comptime TTAB = TileTensor[DType.float64, type_of(LTAB), MutAnyOrigin]
comptime TST = TileTensor[DType.float64, type_of(LST), MutAnyOrigin]
comptime TIDX = TileTensor[DType.int32, type_of(LIDX), MutAnyOrigin]
comptime TDR = TileTensor[DType.float64, type_of(LDR), MutAnyOrigin]
comptime FREQ = 12.0
comptime COUPLING = 10.0
comptime CONVERGE = 20.0
comptime SLEW_S = 0.2  # fly_link.Gait slew_s
comptime TRIM = 0.02  # fly_link.Gait trim
comptime SWING_EXT = 0.7853981633974483  # pi / 4


@always_inline
def sin64(x: Float64) -> Float64:
    """sin in float64: the card has no f64 sine. Reduce to [-pi, pi], Taylor series to x^23 (error < 1e-11 on that range)."""
    var r = x - TWO_PI * Float64(Int(x / TWO_PI + (0.5 if x >= 0 else -0.5)))
    var x2 = r * r
    var term = r
    var sum = r
    for k in range(1, 12):
        term = term * (-x2) / Float64((2 * k) * (2 * k + 1))
        sum += term
    return sum


@always_inline
def odd(i: Int, j: Int) -> Bool:
    """Tripod bias pattern: phi_ij = pi where i and j are legs of opposite tripod (index parity differs)."""
    return (i + j) % 2 == 1


def gait(ctrl: TU, state: TST, drive: TDR, tab: TTAB, idx: TIDX, dt: Float64):
    if thread_idx.x != 0:
        return
    var rate = dt / SLEW_S
    var sig = SIMD[DType.float64, 2](0)
    for side in range(2):
        var step = drive[side] - state[2 * NLEG + side]
        step = rate if step > rate else (-rate if step < -rate else step)
        sig[side] = state[2 * NLEG + side] + step
        state[2 * NLEG + side] = sig[side]
    var amp = SIMD[DType.float64, 8](0)
    var freq = SIMD[DType.float64, 8](0)
    for leg in range(NLEG):
        var d = sig[0] * (1.0 - TRIM) if leg < 3 else sig[1] * (1.0 + TRIM)
        amp[leg] = d if d >= 0 else -d
        freq[leg] = FREQ if d >= 0 else -FREQ
    var theta = SIMD[DType.float64, 8](0)
    var mag = SIMD[DType.float64, 8](0)
    for leg in range(NLEG):
        theta[leg] = state[leg]
        mag[leg] = state[NLEG + leg]
    var dth = SIMD[DType.float64, 8](0)
    var dr = SIMD[DType.float64, 8](0)
    for i in range(NLEG):
        var coupling = Float64(0)
        for j in range(NLEG):
            if odd(i, j):
                # phase_diff = theta_j - theta_i; sin(phase_diff - phi) with phi = pi
                coupling += mag[j] * COUPLING * sin64(theta[j] - theta[i] - 3.141592653589793)
        dth[i] = TWO_PI * freq[i] + coupling
        dr[i] = CONVERGE * (amp[i] - mag[i])
    for i in range(NLEG):
        theta[i] += dth[i] * dt
        mag[i] += dr[i] * dt
        state[i] = theta[i]
        state[NLEG + i] = mag[i]
    for leg in range(NLEG):
        var ph = theta[leg] - TWO_PI * Float64(Int(theta[leg] / TWO_PI))  # python's % 2 pi for a negative phase: add 2 pi below
        if ph < 0:
            ph += TWO_PI
        var seg = min(Int(ph / TWO_PI * Float64(NSEG)), NSEG - 1)
        var dx = ph - Float64(seg) * TWO_PI / Float64(NSEG)
        var base = leg * LEGSZ
        for dof in range(NDOF):
            var o = base + dof
            var c0 = tab[o + (0 * NSEG + seg) * NDOF]
            var c1 = tab[o + (1 * NSEG + seg) * NDOF]
            var c2 = tab[o + (2 * NSEG + seg) * NDOF]
            var c3 = tab[o + (3 * NSEG + seg) * NDOF]
            var y = ((c0 * dx + c1) * dx + c2) * dx + c3
            var neutral = tab[base + 4 * NSEG * NDOF + dof]
            ctrl[Int(idx[leg * NDOF + dof]) * F] = Float32(neutral + mag[leg] * (y - neutral))
        var swing_end = tab[base + 4 * NSEG * NDOF + NDOF] + SWING_EXT
        ctrl[Int(idx[NLEG * NDOF + leg]) * F] = Float32(0.0) if (ph > 0 and ph < swing_end) else Float32(1.0)
