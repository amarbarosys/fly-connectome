"""Maps for the fast 2D closed loop: walls, odour sources and food, vectorised over flies.

A map is data: wall segments (x0, y0, x1, y1), sources (x, y, kind 0 food / 1 decoy, amount),
the starting area and the odour model. Each fly carries a mirror sign; the map is reflected
across `axis` for flies with sign -1, so food and decoy swap sides between paired columns and a
side bias cannot score. Odour is diffusion (exp(-d / scale)) or a
plume carried along +x by wind (Gaussian in y, widening downwind). Diffusion spreads along
the open floor: the distance is geodesic (shortest path on a 0.05 grid that crosses no wall),
so odour comes round corners and through gaps, not through walls. Food depletes as it is eaten.
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

EAT, BITE = .15, 1.


@dataclass(frozen=True)
class Map:
    name: str
    walls: np.ndarray        # (W, 4)
    sources: np.ndarray      # (K, 4) x, y, kind, amount
    start: tuple             # (x0, x1, y0, y1) box, heading (radians or None = random)
    axis: int = 0            # mirror x (0) or y (1)
    plume: bool = False
    scale: float = .6


def _box(x0, x1, y0, y1):
    return [(x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)]


def build(name, seed=0):
    if name == 'tmaze':  # stem up from y=-1.5, arms along y=0 to x=+-1.5, corridors 0.4 wide
        walls = [(-.2, -1.5, .2, -1.5), (-.2, -1.5, -.2, -.2), (.2, -1.5, .2, -.2), (-1.5, -.2, -.2, -.2),
                 (.2, -.2, 1.5, -.2), (-1.5, .2, 1.5, .2), (-1.5, -.2, -1.5, .2), (1.5, -.2, 1.5, .2)]
        return Map(name, np.array(walls), np.array([(1.3, 0, 0, 50), (-1.3, 0, 1, 50)]),
                   ((-.1, .1, -1.4, -1.2), np.pi / 2))
    if name == 'open':  # three food and three decoy patches that run out
        rng = np.random.default_rng(seed)
        xy = rng.uniform(-1.7, 1.7, (6, 2))
        return Map(name, np.array(_box(-2, 2, -2, 2)),
                   np.array([(x, y, k // 3, 5) for k, (x, y) in enumerate(xy)]), ((-1.8, 1.8, -1.8, 1.8), None))
    if name == 'plume':  # wind along +x, sources upwind, flies start downwind
        return Map(name, np.array(_box(0, 4, -.75, .75)), np.array([(.3, .35, 0, 50), (.3, -.35, 1, 50)]),
                   ((3.3, 3.7, -.5, .5), np.pi), axis=1, plume=True)
    if name == 'maze':  # snaking corridors (gap right at y=-0.4, left at y=0.4) to the food at the top;
        # the decoy sits beside the start, so the food:decoy ratio rises along the true route
        walls = _box(-1.5, 1.5, -1.5, 1.5) + [(-1.5, -.4, .8, -.4), (-.8, .4, 1.5, .4)]
        return Map(name, np.array(walls), np.array([(1.2, 1.1, 0, 50), (-1.2, -1.1, 1, 50)]),
                   ((-.2, .2, -1.3, -1.1), np.pi / 2))
    raise ValueError(f'unknown map {name}')


NAMES = ('tmaze', 'open', 'plume', 'maze')


def mirrored(m, sign):
    """Per-fly copies of walls and sources, reflected where sign is -1: (F, W, 4), (F, K, 4)."""
    walls = np.repeat(m.walls[None], len(sign), 0).astype(float)
    sources = np.repeat(m.sources[None], len(sign), 0).astype(float)
    walls[:, :, [m.axis, m.axis + 2]] *= sign[:, None, None]
    sources[:, :, m.axis] *= sign[:, None]
    return walls, sources


def starts(m, sign, rng):
    (x0, x1, y0, y1), heading = m.start
    pos = np.stack([rng.uniform(x0, x1, len(sign)), rng.uniform(y0, y1, len(sign))], 1)
    head = rng.uniform(0, 2 * np.pi, len(sign)) if heading is None else np.full(len(sign), heading)
    pos[:, m.axis] *= sign
    if heading is not None and m.axis == 0:
        head = np.where(sign < 0, np.pi - head, head)
    elif heading is not None:
        head = np.where(sign < 0, -head, head)
    return pos, head


@lru_cache(maxsize=None)
def geodesic(name, h=.05):
    """Per source, the walking distance from every grid cell of the unmirrored map (inf if walled off)."""
    m = build(name)
    lo, hi = m.walls[:, [0, 1]].min(0) - h, m.walls[:, [0, 1]].max(0) + h
    lo, hi = np.minimum(lo, m.walls[:, [2, 3]].min(0)), np.maximum(hi, m.walls[:, [2, 3]].max(0))
    nx, ny = (np.ceil((hi - lo) / h).astype(int) + 1)
    cell = np.stack(np.meshgrid(np.arange(nx), np.arange(ny)), -1).reshape(-1, 2)
    src, dst, w = [], [], []
    walls = np.repeat(m.walls[None], len(cell), 0)
    for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
        to = cell + (dx, dy)
        ok = (to[:, 0] < nx) & (to[:, 1] >= 0) & (to[:, 1] < ny)
        a = lo + cell[ok] * h
        _, blocked = move(a, np.full(ok.sum(), np.arctan2(dy, dx)), h * np.hypot(dx, dy), walls[ok])
        i, j = np.flatnonzero(ok)[~blocked], (to[ok][~blocked] * (1, nx)).sum(1)
        src += [i, j]; dst += [j, i]; w += [np.full(len(i), h * np.hypot(dx, dy))] * 2
    graph = coo_matrix((np.concatenate(w), (np.concatenate(src), np.concatenate(dst))), (len(cell),) * 2).tocsr()
    at = np.round((m.sources[:, :2] - lo) / h).astype(int)
    return lo, h, (nx, ny), dijkstra(graph, indices=at[:, 1] * nx + at[:, 0])


def concentration(m, pos, sources, sign):
    """(F, 2): food and decoy odour at each fly, summed over sources, scaled by what is left."""
    d = pos[:, None, :] - sources[:, :, :2]
    if m.plume:
        down = d[..., 0]
        width = .1 + .15 * np.maximum(down, 0)
        c = np.where(down > 0, np.exp(-d[..., 1] ** 2 / (2 * width ** 2)) * .1 / width, np.exp(np.minimum(down, 0) / .1))
    else:
        lo, h, (nx, ny), field = geodesic(m.name)
        local = pos.copy()
        local[:, m.axis] *= sign  # the field is for the unmirrored map
        ij = np.clip(np.round((local - lo) / h).astype(int), 0, [nx - 1, ny - 1])
        c = np.exp(-field[:, ij[:, 1] * nx + ij[:, 0]].T / m.scale)
    c = c * (sources[..., 3] > 0)
    kind = sources[..., 2]
    return np.stack([(c * (kind == 0)).sum(1), (c * (kind == 1)).sum(1)], 1)


def move(pos, head, step, walls):
    """Advance unless the path crosses a wall; returns new positions and the blocked mask."""
    new = pos + step * np.stack([np.cos(head), np.sin(head)], 1)
    a, b = walls[..., :2], walls[..., 2:]

    def side(p, q, r):  # sign of the turn p -> q -> r, broadcast over (F, W)
        return np.sign((q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0]))
    p, q = pos[:, None, :], new[:, None, :]
    cross = (side(a, b, p) != side(a, b, q)) & (side(p, q, a) != side(p, q, b))
    blocked = cross.any(1)
    return np.where(blocked[:, None], pos, new), blocked


def eat(pos, sources):
    """Each fly within EAT of a source with something left takes one BITE; returns (F, 2) bites."""
    near = (np.linalg.norm(pos[:, None, :] - sources[:, :, :2], axis=2) < EAT) & (sources[..., 3] > 0)
    sources[..., 3] -= near * BITE
    kind = sources[..., 2]
    return np.stack([(near & (kind == 0)).sum(1), (near & (kind == 1)).sum(1)], 1)
