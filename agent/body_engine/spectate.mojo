"""Spectator kernels for live.mojo (-DFOLLOW=1, -DHUD=1): view framing and overlay, no MuJoCo function ported.

follow  one thread: the root bodies' xpos rows of the engine's own output buffer (the rows render.mojo reads, animal 0)
        -> cam eye and target, by follow_camera.fit_distance (the host fit_camera's formula): the camera follows with no
        pose copied to the host. rootspec = [body, radius] per root (NROOT, from roots.bin).
hud     one thread per pixel of the bar panel, drawn over the shaded image: energy, health, stamina bars of animal 0 from
        four float32 rows [row][animal] (energy, health, stamina, dead; smooth.mojo OGAME layout). Fill = hud.bar_rect
        rounded to a pixel. HUD pixels carry alpha 254: alpha 255 stays "animal" for the stand and framing gates.
"""
from std.gpu import block_idx, block_dim, thread_idx
from std.math import sqrt
from std.sys import get_defined_int
from layout import TileTensor, row_major
import render as ren
from follow_camera import DIR_X, DIR_Z, fit_distance
from hud import bar_rect

comptime NROOT = get_defined_int["NROOT", 1]()
comptime LROOT = row_major[2 * NROOT]()
comptime TROOT = TileTensor[DType.float32, type_of(LROOT), MutAnyOrigin]
comptime LGS = row_major[4 * ren.F]()
comptime TGS = TileTensor[DType.float32, type_of(LGS), MutAnyOrigin]

comptime BAR_X = 24  # panel geometry, pixels from the top left
comptime BAR_Y = 12
comptime BAR_W = 120
comptime BAR_H = 8
comptime BAR_PITCH = 14
comptime PANEL_W = BAR_X + BAR_W + 8
comptime PANEL_H = BAR_Y + 3 * BAR_PITCH
comptime HUD_ALPHA = UInt32(254)


def follow(cam: ren.TCAM, pose: ren.TPOSE, root: TROOT):
    if thread_idx.x != 0:
        return
    var tx = Float32(0)
    var ty = Float32(0)
    var tz = Float32(0)
    for i in range(NROOT):
        var b = Int(root[2 * i]) - 1
        tx += pose[3 * b * ren.F]
        ty += pose[(3 * b + 1) * ren.F]
        tz += pose[(3 * b + 2) * ren.F]
    tx /= Float32(NROOT)
    ty /= Float32(NROOT)
    tz /= Float32(NROOT)
    var radius = Float32(0)
    for i in range(NROOT):
        var b = Int(root[2 * i]) - 1
        var dx = pose[3 * b * ren.F] - tx
        var dy = pose[(3 * b + 1) * ren.F] - ty
        var dz = pose[(3 * b + 2) * ren.F] - tz
        radius = max(radius, sqrt(dx * dx + dy * dy + dz * dz) + root[2 * i + 1])
    var d = fit_distance(radius, ren.W, ren.H, cam[6] * 57.29578)
    cam[0] = tx + d * DIR_X
    cam[1] = ty - d * DIR_X
    cam[2] = tz + d * DIR_Z
    cam[3] = tx
    cam[4] = ty
    cam[5] = tz


@always_inline
def glyph_bit(which: Int, gx: Int, gy: Int) -> Bool:
    """Pixel (gx, gy) of the 3 x 5 label glyph E (0), H (1) or S (2): 15 bits, top row in the high bits."""
    var rows = SIMD[DType.uint16, 4](0b111100110100111, 0b101101111101101, 0b111100111001111, 0)
    return (rows[which] >> UInt16(14 - (3 * gy + gx))) & 1 == 1


def hud(img: ren.TIMG, state: TGS):
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= PANEL_W * PANEL_H:
        return
    var x = i % PANEL_W
    var y = i // PANEL_W
    var bar = (y - BAR_Y + 1) // BAR_PITCH  # a bar's cell spans one row above it to 6 rows below its 8 (glyph rows -1..8)
    var row = (y - BAR_Y + 1) % BAR_PITCH - 1
    if y < BAR_Y - 1 or bar > 2:
        return
    var rgb = UInt32(0)
    if x >= BAR_X - 12 and x < BAR_X - 6:  # label glyph at 2x: 6 x 10 px
        if row < -1 or row > 8 or not glyph_bit(bar, (x - (BAR_X - 12)) // 2, (row + 1) // 2):
            return
        rgb = UInt32(0xFFFFFF)
    elif x >= BAR_X and x < BAR_X + BAR_W and row >= 0 and row < BAR_H:
        var fill = bar_rect(Float32(BAR_X), Float32(BAR_Y), Float32(BAR_W), Float32(BAR_H), state[bar * ren.F])
        if Float32(x - BAR_X) + 0.5 < fill.width:
            rgb = UInt32(0x808080) if state[3 * ren.F] != 0 else (UInt32(0x3CC8F0) if bar == 0 else (UInt32(0x3C3CDC) if bar == 1 else UInt32(0xE68C46)))
        else:
            rgb = UInt32(0x282828)
    else:
        return
    img[y * ren.W + x] = rgb | HUD_ALPHA << 24
