"""The play world's assets (docs/plans/WORLD.md, PLAY.md): seeded grass tufts and flowers (drawn only, no contact)
and the food and decoy domes at the plume sources, on Kinestride's terrain function, every part in world mm on
body 0. Grouped by colour, five parts in all: the renderer holds 255 parts, the fly uses 69. Port of assets.py
(kept as the reference of the checks below); the generator is this file's own PCG32, so a seed lays out different
grass than numpy did.

usage: scene build OUTDIR SEED P0..P15 [X Y R]...   -> OUTDIR/parts.bin, OUTDIR/src.bin
       scene check P0..P15 SEED...                  -> the five world checks per seed, exit 1 on a failure
parts.bin: uint32 nparts; per part float64 rgba 4, uint32 nv, uint32 nf, float64 vertices, int32 faces
src.bin: float32 per source x, y, kind (0 food, 1 decoy), amount, presence (live.mojo -DWORLD=1)
Build: mojo build -I ~/Projects/mojo/kinestride/kinestride agent/body_engine/scene.mojo
"""
from std.math import cos, sin, sqrt, pi, log
from std.memory import bitcast
from std.sys import argv, exit, stderr
from assets import Mesh, sphere, primitive
from world import NP, PW, height_grad

comptime ARENA = 19.0  # mm: inside body-v2's walls (inner faces at +-20)
comptime CLEAR = 5.0   # mm around the start (origin) left empty: the fly is dropped there


struct Rng(Movable):
    """PCG32 (O'Neill), seeded by (seed, stream 54)."""
    var s: UInt64
    var inc: UInt64

    def __init__(out self, seed: UInt64):
        self.s = 0
        self.inc = (UInt64(54) << 1) | 1
        _ = self.next()
        self.s += seed
        _ = self.next()

    def next(mut self) -> UInt32:
        var old = self.s
        self.s = old * 6364136223846793005 + self.inc
        var xs = UInt32(((old >> 18) ^ old) >> 27)
        var rot = UInt32(old >> 59)
        return (xs >> rot) | (xs << ((UInt32(0) - rot) & 31))

    def uniform(mut self, lo: Float64, hi: Float64) -> Float64:
        var a = Float64(self.next() >> 5)
        var b = Float64(self.next() >> 6)
        return lo + (hi - lo) * (a * 67108864.0 + b) / 9007199254740992.0

    def normal(mut self, mu: Float64, sd: Float64) -> Float64:
        var u = self.uniform(1e-300, 1.0)
        return mu + sd * sqrt(-2.0 * log(u)) * cos(2.0 * pi * self.uniform(0.0, 1.0))

    def integers(mut self, lo: Int, hi: Int) -> Int:
        return lo + Int(self.next() % UInt32(hi - lo))


struct Part(Movable):
    var m: Mesh
    var rgba: List[Float64]

    def __init__(out self, var m: Mesh, rgba: List[Float64]):
        self.m = m^
        self.rgba = rgba.copy()


def ground(p: PW, x: Float64, y: Float64) -> Float64:
    return Float64(height_grad(Float32(x), Float32(y), p)[0])


def low(p: PW, x: Float64, y: Float64) -> Bool:
    """Off the landmark cones: ground within 0.3 mm on a 1.5 mm ring (9 points, as assets.py) and at the centre."""
    for i in range(9):
        var a = 2.0 * pi * Float64(i) / 8.0
        if ground(p, x + 1.5 * cos(a), y + 1.5 * sin(a)) >= 0.3:
            return False
    return ground(p, x, y) < 0.3


def far(x: Float64, y: Float64, cx: Float64, cy: Float64, r: Float64) -> Bool:
    return sqrt((x - cx) ** 2 + (y - cy) ** 2) > r


def spot(mut rng: Rng, p: PW, stones: List[Float64], food: Tuple[Float64, Float64], decoy: Tuple[Float64, Float64]) -> Tuple[Float64, Float64]:
    """A random free place: low ground, clear of the start, the stones and both sources."""
    while True:
        var x = rng.uniform(-ARENA, ARENA)
        var y = rng.uniform(-ARENA, ARENA)
        var ok = low(p, x, y) and far(x, y, 0, 0, CLEAR) and far(x, y, food[0], food[1], 2) and far(x, y, decoy[0], decoy[1], 2)
        for s in range(0, len(stones), 3):
            ok = ok and far(x, y, stones[s], stones[s + 1], stones[s + 2] + 0.5)
        if ok:
            return (x, y)


def append(mut into: Mesh, m: Mesh, dx: Float64, dy: Float64, dz: Float64, sc: Float64):
    """Merge m (scaled by sc, moved by d) into `into`, faces re-indexed."""
    var base = len(into.v) // 3
    for i in range(len(m.v) // 3):
        into.v.append(m.v[3 * i] * sc + dx)
        into.v.append(m.v[3 * i + 1] * sc + dy)
        into.v.append(m.v[3 * i + 2] * sc + dz)
    for f in m.f:
        into.f.append(f + base)


def blade(mut into: Mesh, mut rng: Rng, bx: Float64, by: Float64, bz: Float64, h: Float64, lean: Float64, w: Float64):
    """One grass blade: a tapered quad bent once, both windings (the rasteriser keeps either face)."""
    var a = rng.uniform(0, 2 * pi)
    var sx = cos(a) * w / 2
    var sy = sin(a) * w / 2
    var tx = bx + lean * cos(a + 1.3)
    var ty = by + lean * sin(a + 1.3)
    var tz = bz + h
    var mx = bx + (tx - bx) * 0.55
    var my = by + (ty - by) * 0.55
    var mz = bz + (tz - bz) * 0.55 + 0.05 * h
    var base = len(into.v) // 3
    var v: List[Float64] = [bx - sx, by - sy, bz, bx + sx, by + sy, bz, mx + 0.6 * sx, my + 0.6 * sy, mz, mx - 0.6 * sx, my - 0.6 * sy, mz, tx, ty, tz]
    for c in v:
        into.v.append(c)
    for f in [0, 1, 2, 0, 2, 3, 3, 2, 4, 2, 1, 0, 3, 2, 0, 4, 2, 3]:
        into.f.append(base + f)


def scene(p: PW, seed: Int, stones: List[Float64], mut src: List[Float32]) -> List[Part]:
    """(world parts for model.meshes, source rows): five parts (grass, stalks, blooms, food dome, decoy dome)."""
    var rng = Rng(UInt64(seed))
    var food: Tuple[Float64, Float64] = (12.0, 12.0)
    var decoy: Tuple[Float64, Float64] = (-12.0, -10.0)
    if not low(p, food[0], food[1]):
        food = spot(rng, p, stones, (99.0, 99.0), decoy)
    if not low(p, decoy[0], decoy[1]):
        decoy = spot(rng, p, stones, food, (99.0, 99.0))
    var grass = Mesh()
    for _ in range(70):  # tufts of 4-7 blades, 1.5-5 mm: taller than the fly (2.5 mm long) in places
        var c = spot(rng, p, stones, food, decoy)
        var h0 = rng.uniform(1.5, 5.0)
        for _ in range(rng.integers(4, 8)):
            var bx = c[0] + rng.normal(0, 0.25)
            var by = c[1] + rng.normal(0, 0.25)
            var h = h0 * rng.uniform(0.6, 1.1)
            var lean = h0 * rng.uniform(0.1, 0.35)
            blade(grass, rng, bx, by, ground(p, bx, by) - 0.02, h, lean, rng.uniform(0.12, 0.25))
    var stalks = Mesh()
    var blooms = Mesh()
    var ball = sphere(1)
    for _ in range(12):  # flowers: a thin stalk and a head, 4-8 mm
        var c = spot(rng, p, stones, food, decoy)
        var z = ground(p, c[0], c[1])
        var h = rng.uniform(4, 8)
        append(stalks, primitive(True, 0.06, 0.06, h / 2), c[0], c[1], z + h / 2, 1.0)
        append(blooms, ball, c[0], c[1], z + h, rng.uniform(0.5, 0.9))
    var parts = List[Part]()
    parts.append(Part(grass^, [0.33, 0.55, 0.22, 1.0]))
    parts.append(Part(stalks^, [0.30, 0.48, 0.20, 1.0]))
    parts.append(Part(blooms^, [0.92, 0.42, 0.62, 1.0]))
    var dome = sphere(2)
    for k in range(2):  # a low dome per source (world_content kinds: 0 food, 1 decoy); spectate.minimap's colours
        var c = food if k == 0 else decoy
        var mk = Mesh()
        var zc = ground(p, c[0], c[1])
        for i in range(len(dome.v) // 3):
            mk.v.append(dome.v[3 * i] * 1.2 + c[0])
            mk.v.append(dome.v[3 * i + 1] * 1.2 + c[1])
            mk.v.append(dome.v[3 * i + 2] * 0.5 + zc)
        mk.f = dome.f.copy()
        var col: List[Float64] = [1.0, 0.75, 0.25, 1.0]
        if k == 1:
            col = [0.38, 0.63, 1.0, 1.0]
        parts.append(Part(mk^, col))
        for x in [Float32(c[0]), Float32(c[1]), Float32(k), Float32(1), Float32(1)]:
            src.append(x)
    return parts^


def le(mut b: List[UInt8], v: UInt64, n: Int):
    for s in range(n):
        b.append(UInt8(Int((v >> UInt64(8 * s)) & 255)))


def save(parts: List[Part], src: List[Float32], dir: String) raises:
    var b = List[UInt8]()
    le(b, UInt64(len(parts)), 4)
    for qi in range(len(parts)):
        ref q = parts[qi]
        for c in q.rgba:
            le(b, bitcast[DType.uint64](c), 8)
        le(b, UInt64(len(q.m.v) // 3), 4)
        le(b, UInt64(len(q.m.f) // 3), 4)
        for c in q.m.v:
            le(b, bitcast[DType.uint64](c), 8)
        for f in q.m.f:
            le(b, UInt64(f), 4)
    with open(dir + "/parts.bin", "w") as f:
        f.write_bytes(Span(b))
    var s = List[UInt8]()
    for x in src:
        le(s, UInt64(bitcast[DType.uint32](x)), 4)
    with open(dir + "/src.bin", "w") as f:
        f.write_bytes(Span(s))


def params(a: List[String], at: Int) raises -> PW:
    var p = PW()
    for k in range(NP):
        p[k] = Float32(Float64(a[at + k]))
    return p


def check(p: PW, seed: Int) -> Int:
    """assets.py's five checks: part budget, triangles, nothing floats or sinks, the start stays clear, two sources."""
    var stones: List[Float64] = [-9, 7, 1.8, 8, -5, 2.2, 8, 9, 1.4]
    var src = List[Float32]()
    var parts = scene(p, seed, stones, src)
    var tri = 0
    var lowest = 1e30
    var near = 1e30
    for qi in range(len(parts)):
        ref q = parts[qi]
        tri += len(q.m.f) // 3
        for i in range(len(q.m.v) // 3):
            var x = q.m.v[3 * i]
            var y = q.m.v[3 * i + 1]
            lowest = min(lowest, q.m.v[3 * i + 2] - ground(p, x, y))
            near = min(near, sqrt(x * x + y * y))
    var bad = 0
    var tag = "seed " + String(seed) + ": "
    var oks: List[Bool] = [len(parts) + 69 <= 255, tri < 20000, lowest >= -1.3, near > CLEAR - 1.5, len(src) == 10]
    var names: List[String] = [String(len(parts)) + " world parts (+69 fly <= 255)", String(tri) + " triangles (< 20000)",
                               "lowest point vs ground " + String(lowest) + " mm (>= -1.3: domes sink half)",
                               "nearest asset to the start " + String(near) + " mm (> 3.5)", String(len(src) // 5) + " plume sources"]
    for i in range(5):
        print(("PASS " if oks[i] else "FAIL ") + tag + names[i])
        bad += 0 if oks[i] else 1
    return bad


def main() raises:
    var a = List[String]()
    for x in argv():
        a.append(String(x))
    try:
        if len(a) >= 4 + NP and a[1] == "build" and (len(a) - 4 - NP) % 3 == 0:
            var stones = List[Float64]()
            for i in range(4 + NP, len(a)):
                stones.append(Float64(a[i]))
            var src = List[Float32]()
            var parts = scene(params(a, 4), Int(a[3]), stones, src)
            save(parts, src, a[2])
        elif len(a) > 2 + NP and a[1] == "check":
            var bad = 0
            for i in range(2 + NP, len(a)):
                var p = params(a, 2)
                p[5] = Float32(Int(a[i]))  # the terrain seed follows the scene seed, as play.setup does
                bad += check(p, Int(a[i]))
            if bad:
                print("FAIL scene check:", bad, "checks", file=stderr)
                exit(1)
            print("PASS scene check", len(a) - 2 - NP, "seeds")
        else:
            print("usage: scene build OUTDIR SEED P0..P15 [X Y R]... | scene check P0..P15 SEED...", file=stderr)
            exit(2)
    except e:
        print("FAIL scene:", e, file=stderr)
        exit(1)
