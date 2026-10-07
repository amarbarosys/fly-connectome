"""World assets for the renderer (docs/plans/WORLD.md): triangle meshes for the world body's box and ellipsoid geoms
(walls, stones: they collide through the engine's convex pairs), seeded grass and flowers (drawn only, no contact), and
the food and decoy markers at the plume sources. Every part is in world mm on body 0 (render.vertex draws it fixed).
Grouped by colour, a few parts in all: the renderer holds 255 parts, the fly uses 69.

  .venv/bin/python -m agent.body_engine.assets check     (CPU: part count, triangles, everything on the ground, nothing at the start)
"""
import sys

import numpy as np

from agent.body_engine.world import height_grad

GRASS, BLOOM, STALK = (0.33, 0.55, 0.22, 1), (0.92, 0.42, 0.62, 1), (0.30, 0.48, 0.20, 1)
FOOD, DECOY = (1.0, 0.75, 0.25, 1), (0.38, 0.63, 1.0, 1)  # spectate.minimap's colours
ARENA = 19.0  # mm: inside body-v2's walls (inner faces at +-20)
CLEAR = 5.0  # mm around the start (origin) left empty: the fly is dropped there


def sphere(n=2):
    """unit icosphere, n subdivisions: (vertices, faces), outward winding."""
    t = (1 + 5 ** 0.5) / 2
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
         (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    v = [np.array(p, float) / np.linalg.norm(p) for p in v]
    for _ in range(n):
        mid, g = {}, []
        def m(a, b):
            k = (min(a, b), max(a, b))
            if k not in mid:
                p = v[a] + v[b]
                v.append(p / np.linalg.norm(p))
                mid[k] = len(v) - 1
            return mid[k]
        for a, b, c in f:
            ab, bc, ca = m(a, b), m(b, c), m(c, a)
            g += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        f = g
    return np.array(v), np.array(f)


def primitive(box, size):
    """(vertices, faces) of a MuJoCo box (half sizes) or ellipsoid (radii) in its geom frame."""
    if box:
        v = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float) * size[:3]
        f = np.array([(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)])
        return v, f
    v, f = sphere(2)
    return v * size[:3], f


def merge(parts):
    vs, fs, n = [], [], 0
    for v, f in parts:
        vs.append(v); fs.append(np.asarray(f) + n); n += len(v)
    return np.concatenate(vs), np.concatenate(fs)


def blade(base, h, lean, w, rng):
    """one grass blade: a tapered quad bent once, both windings (the rasteriser keeps either face)."""
    a = rng.uniform(0, 2 * np.pi)
    side = np.array([np.cos(a), np.sin(a), 0]) * w / 2
    tip = base + np.array([lean * np.cos(a + 1.3), lean * np.sin(a + 1.3), h])
    mid = base + (tip - base) * 0.55 + np.array([0, 0, 0.05 * h])
    v = np.array([base - side, base + side, mid + side * 0.6, mid - side * 0.6, tip])
    f = [(0, 1, 2), (0, 2, 3), (3, 2, 4)]
    return v, f + [(c, b, a) for a, b, c in f]


def scene(params, seed=1, stones=(), food=(12.0, 12.0), decoy=(-12.0, -10.0)):
    """(world parts for model.meshes, src rows for live.mojo -DWORLD=1). params: the terrain (live.WORLD with its seed);
    stones: (x, y, radius) kept clear of grass."""
    rng = np.random.default_rng(seed)
    ground = lambda x, y: float(height_grad(np.array([x]), np.array([y]), params)[0][0])

    def low(x, y):  # off the landmark cones (live.WORLD: up to tens of mm): ground within 0.3 mm over a 1.5 mm ring
        return all(ground(x + 1.5 * np.cos(a), y + 1.5 * np.sin(a)) < 0.3 for a in np.linspace(0, 2 * np.pi, 9)) and ground(x, y) < 0.3

    def spot(clear=CLEAR):
        while True:
            x, y = rng.uniform(-ARENA, ARENA, 2)
            if low(x, y) and np.hypot(x, y) > clear and all(np.hypot(x - sx, y - sy) > r + 0.5 for sx, sy, r in stones) \
                    and np.hypot(x - food[0], y - food[1]) > 2 and np.hypot(x - decoy[0], y - decoy[1]) > 2:
                return x, y

    if not low(*food):
        food = (99.0, 99.0); food = spot()
    if not low(*decoy):
        decoy = (99.0, 99.0); decoy = spot()
    grass = []
    for _ in range(70):  # tufts of 4-7 blades, 1.5-5 mm: taller than the fly (2.5 mm long) in places
        x, y = spot()
        h0 = rng.uniform(1.5, 5.0)
        for _ in range(rng.integers(4, 8)):
            b = np.array([x + rng.normal(0, 0.25), y + rng.normal(0, 0.25), 0])
            b[2] = ground(b[0], b[1]) - 0.02
            grass.append(blade(b, h0 * rng.uniform(0.6, 1.1), h0 * rng.uniform(0.1, 0.35), rng.uniform(0.12, 0.25), rng))
    stalks, blooms = [], []
    ball = sphere(1)
    for _ in range(12):  # flowers: a thin stalk and a head, 4-8 mm
        x, y = spot()
        z, h = ground(x, y), rng.uniform(4, 8)
        stalks.append(primitive(1, np.array([0.06, 0.06, h / 2])))
        stalks[-1] = (stalks[-1][0] + [x, y, z + h / 2], stalks[-1][1])
        blooms.append((ball[0] * rng.uniform(0.5, 0.9) + [x, y, z + h], ball[1]))
    marks, src = [], []
    for (x, y), rgba, kind in ((food, FOOD, 0), (decoy, DECOY, 1)):  # a low dome per source (world_content kinds: 0 food, 1 decoy)
        v, f = sphere(2)
        marks.append((v * [1.2, 1.2, 0.5] + [x, y, ground(x, y)], f, rgba))
        src.append([x, y, kind, 1.0, 1.0])
    parts = [(*merge(grass), GRASS), (*merge(stalks), STALK), (*merge(blooms), BLOOM), *marks]
    return parts, np.array(src, np.float32)


def check():
    from agent.body_engine.live import WORLD
    params = WORLD[:5] + (1,) + WORLD[6:]
    parts, src = scene(params, stones=((-9, 7, 1.8), (8, -5, 2.2), (8, 9, 1.4)))
    tri = sum(len(f) for _, f, _ in parts)
    lows = [float(np.min(v[:, 2] - height_grad(v[:, 0], v[:, 1], params)[0])) for v, _, _ in parts]
    near = min(float(np.min(np.hypot(v[:, 0], v[:, 1]))) for v, _, _ in parts)
    ok = [(f'{len(parts)} world parts (+69 fly <= 255)', len(parts) + 69 <= 255), (f'{tri} triangles (< 20000)', tri < 20000),
          (f'lowest point vs ground {min(lows):.2f} mm (>= -1.3: domes sink half)', min(lows) >= -1.3),
          (f'nearest asset to the start {near:.1f} mm (> {CLEAR - 1.5})', near > CLEAR - 1.5), (f'{len(src)} plume sources', len(src) == 2)]
    for k, v in ok:
        print(('PASS ' if v else 'FAIL ') + k)
    if not all(v for _, v in ok):
        sys.exit('FAIL assets')


if __name__ == '__main__':
    {'check': check}[sys.argv[1]]()
