"""Brain v0 on the card (docs/plans/SPIDER_BRAIN.md unit 1, transport b): the alternating-tetrapod gait of
spider_gait.py (the oracle: same constants, same layout), one thread per (animal, servo), writing the engine's ctrl
buffer [servo][animal] in place. The brain stage is then in the same process as the K-step launch: no host copy per
control step. Drives per animal: speed, turn (rows 0 and 1 of the drive buffer, [row][animal]).

ctrl layout: servo 0,1 = pedicel (0), servo 2 + 7 leg + j = leg (l1..l4, r1..r4), j 0 coxa_yaw, 1 coxa_pitch, rest 0.
"""
from std.math import cos, sin
from max.gpu import block_idx, block_dim, thread_idx
from layout import TileTensor, row_major
from smooth import F, NU, TU

comptime FREQ = Float32(5.0)
comptime YAW = Float32(0.4)
comptime LIFT = Float32(0.3)
comptime TWO_PI = Float32(6.2831853)
comptime LD = row_major[2 * F]()
comptime TD = TileTensor[DType.float32, type_of(LD), MutAnyOrigin]


def gait(ctrl: TU, drive: TD, t: Float32):
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= NU * F:
        return
    var u = i // F
    var f = i % F
    var out = Float32(0)
    if u >= 2:
        var leg = (u - 2) // 7
        var j = (u - 2) % 7
        var side = Float32(1) if leg < 4 else Float32(-1)
        var tet = Float32(0.5) if (leg % 2 == 1) == (leg < 4) else Float32(0)  # l2 l4 r1 r3 half a cycle behind
        var x = FREQ * t + tet
        var ph = x - Float32(Int(x))
        var speed = drive[f]
        var sweep = (1 + drive[F + f] * side) * speed
        if j == 0:
            out = side * YAW * sweep * cos(TWO_PI * ph)
        elif j == 1:
            out = LIFT * speed * max(Float32(0), -sin(TWO_PI * ph))
    ctrl[u * F + f] = out
