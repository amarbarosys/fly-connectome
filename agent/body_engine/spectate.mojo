"""Spectator kernels for live.mojo (-DFOLLOW=1, -DHUD=1): view framing and overlay, no MuJoCo function ported.

follow  one thread: the root bodies' xpos rows of the engine's own output buffer (the rows render.mojo reads, animal 0)
        -> cam eye and target, by follow_camera.fit_distance (the host fit_camera's formula): the camera follows with no
        pose copied to the host. rootspec = [body, radius] per root (NROOT, from roots.bin).
hud     one thread per pixel of the bar panel, drawn over the shaded image: energy, health, stamina bars of animal 0 from
        four float32 rows [row][animal] (energy, health, stamina, dead; smooth.mojo OGAME layout). Fill = hud.bar_rect
        rounded to a pixel. HUD pixels carry alpha 254: alpha 255 stays "animal" for the stand and framing gates.
sense   (WORLD) one thread per odor site: world xyz from the engine's pose rows (body xpos + xquat x the site's local offset), then
        world_content.sample_two on the descriptor src.bin: food and decoy concentration per site, on the card, every step.
hud_text, minimap   (WORLD + HUD) the sampled concentrations as 'd.ddd' text under the bars, and the same descriptor's sources drawn at
        hud.project_source in a mini-map (world XY orthographic around the animal, MAP_SCALE px per mm).
"""
from max.gpu import block_idx, block_dim, thread_idx
from std.math import sqrt
from std.sys import get_defined_int
from layout import TileTensor, row_major
import render as ren
from follow_camera import DIR_X, DIR_Z, fit_distance
from hud import bar_rect, project_source
import world_content as wc

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
comptime NSITE = get_defined_int["NSITE", 2]()  # WORLD: odor sites (fly/l_odor, fly/r_odor; spider: leg I tarsus tips), sites.bin = body, local xyz
comptime NSRC = 2  # WORLD: plume sources, src.bin = x_mm, y_mm, kind, present, remaining per source (world_content.sample_two takes two)
comptime LSITE = row_major[4 * NSITE]()
comptime TSITE = TileTensor[DType.float32, type_of(LSITE), MutAnyOrigin]
comptime LSRC = row_major[5 * NSRC]()
comptime TSRC = TileTensor[DType.float32, type_of(LSRC), MutAnyOrigin]
comptime LPLUME = row_major[2 * NSITE]()
comptime TPLUME = TileTensor[DType.float32, type_of(LPLUME), MutAnyOrigin]
comptime TXT_Y = PANEL_H + 4  # concentration text: one line each for food and decoy, sites left to right
comptime TXT_H = 2 * BAR_PITCH
comptime MAP = 128  # mini-map: MAP x MAP px at the top right, world XY orthographic around the animal, MAP_SCALE px per mm
comptime MAP_X = ren.W - MAP - 8
comptime MAP_Y = 8
comptime MAP_SCALE = Float32(get_defined_int["MAP_SCALE_E3", 400]()) * 0.001


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


def sense(plume: TPLUME, pose: ren.TPOSE, sites: TSITE, src: TSRC):
    """One thread per odor site: world xyz = body xpos + body quat x local offset (the pose rows render.mojo reads), then the plume
    concentration at its XY by world_content.sample_two (maps.py:104-120): plume[2 s] food, plume[2 s + 1] decoy. Sources are
    the descriptor src.bin, the same one the mini-map draws."""
    var s = Int(block_idx.x * block_dim.x + thread_idx.x)
    if s >= NSITE:
        return
    var b = Int(sites[4 * s]) - 1
    var q = ren.Q4(pose[(ren.OXQ + 4 * b) * ren.F], pose[(ren.OXQ + 4 * b + 1) * ren.F], pose[(ren.OXQ + 4 * b + 2) * ren.F], pose[(ren.OXQ + 4 * b + 3) * ren.F])
    var w = ren.rot(q, ren.V3(sites[4 * s + 1], sites[4 * s + 2], sites[4 * s + 3], 0)) + ren.V3(pose[3 * b * ren.F], pose[(3 * b + 1) * ren.F], pose[(3 * b + 2) * ren.F], 0)
    var c = wc.sample_two(w[0], w[1],
        src[0], src[1], Int32(src[2]), Int32(src[3]), src[4],
        src[5], src[6], Int32(src[7]), Int32(src[8]), src[9])
    plume[2 * s] = c[0]
    plume[2 * s + 1] = c[1]


@always_inline
def glyph16(which: Int, gx: Int, gy: Int) -> Bool:
    """Pixel of a 3 x 5 glyph: 0-9, '.' (10), F D L R (11-14), 15 bits, top row in the high bits."""
    var rows = SIMD[DType.uint16, 16](0x7B6F, 0x2C97, 0x73E7, 0x73CF, 0x5BC9, 0x79CF, 0x79EF, 0x7249, 0x7BEF, 0x7BCF, 0x0002, 0x79A4, 0x6B6E, 0x4927, 0x6B75, 0)
    return (rows[which] >> UInt16(14 - (3 * gy + gx))) & 1 == 1


@always_inline
def text_glyph(plume: TPLUME, line: Int, site: Int, ch: Int) -> Int:
    """Character `ch` (0-4) of the 'd.ddd' text of plume[line, site]: a digit 0-9, or 10 for the point."""
    var v = min(max(plume[2 * site + line], 0), Float32(9.999))
    var n = Int(v * 1000 + 0.5)
    if ch == 1:
        return 10
    return (n // 1000) % 10 if ch == 0 else ((n // 100) % 10 if ch == 2 else ((n // 10) % 10 if ch == 3 else n % 10))


def hud_text(img: ren.TIMG, plume: TPLUME):
    """Concentrations of every site as text under the bars: line 0 'F' food, line 1 'D' decoy, 'd.ddd' per site, 2x glyphs, 8 px pitch."""
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= PANEL_W * TXT_H:
        return
    var x = i % PANEL_W
    var y = TXT_Y + i // PANEL_W
    var line = (y - TXT_Y) // BAR_PITCH
    var row = (y - TXT_Y) % BAR_PITCH
    if row >= 10:
        return
    var gy = row // 2
    var which = -1
    if x >= BAR_X - 12 and x < BAR_X - 6:
        which = 11 if line == 0 else 12
        if not glyph16(which, (x - (BAR_X - 12)) // 2, gy):
            return
    else:
        var cx = x - BAR_X
        if cx < 0 or cx >= NSITE * 48:
            return
        var site = cx // 48
        var ch = (cx % 48) // 8
        if ch > 4 or cx % 8 >= 6:
            return
        which = text_glyph(plume, line, site, ch)
        if not glyph16(which, (cx % 8) // 2, gy):
            return
    img[y * ren.W + x] = UInt32(0xFFFFFF) | HUD_ALPHA << 24


def minimap(img: ren.TIMG, pose: ren.TPOSE, root: TROOT, src: TSRC):
    """World XY orthographic around animal 0's root: the sources of the descriptor as 7 x 7 squares at hud.project_source, coloured by
    kind (world_content colours: food (255, 190, 64), decoy (96, 160, 255); a depleted or absent source grey), the animal a white 3 x 3 at the centre."""
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= MAP * MAP:
        return
    var px = Float32(i % MAP) + 0.5
    var py = Float32(i // MAP) + 0.5
    var b = Int(root[0]) - 1
    var tx = pose[3 * b * ren.F]
    var ty = pose[(3 * b + 1) * ren.F]
    var rgb = UInt32(0x1E1E1E)
    if abs(px - Float32(MAP) * 0.5) < 2 and abs(py - Float32(MAP) * 0.5) < 2:
        rgb = UInt32(0xFFFFFF)
    for k in range(NSRC):
        var m = project_source(src[5 * k], src[5 * k + 1], tx, ty, MAP_SCALE, Float32(MAP) * 0.5, Float32(MAP) * 0.5, Int(src[5 * k + 2]))
        if abs(px - m.x) < 3.5 and abs(py - m.y) < 3.5:
            rgb = UInt32(0x808080) if src[5 * k + 3] == 0 or src[5 * k + 4] <= 0 else (UInt32(0x40BEFF) if m.kind == 0 else UInt32(0xFFA060))
    img[(MAP_Y + i // MAP) * ren.W + MAP_X + i % MAP] = rgb | HUD_ALPHA << 24
