"""Schema and Python oracle for the bounded plume scenario.

The plume equation is ported from agent/maps.py:104-120. Engine coordinates
are millimetres while maps.py uses metres; conversion is explicit here.
"""
from dataclasses import dataclass

import numpy as np
from agent import maps

MM_PER_M = 1000.0
FOOD, DECOY = 0, 1


@dataclass(frozen=True)
class Source:
    """Immutable render/sense descriptor in world millimetres."""
    source_id: int
    x_mm: float
    y_mm: float
    kind: int
    color_rgb: tuple[int, int, int]


@dataclass(frozen=True)
class ResourceState:
    """Resource-state snapshot; absent or empty sources emit no odor."""
    remaining: tuple[float, ...]
    present: tuple[bool, ...]


@dataclass(frozen=True)
class PlumeWorld:
    seed: int
    sources: tuple[Source, ...]
    state: ResourceState


def build(seed: int = 0) -> PlumeWorld:
    """Create the two source positions from maps.build('plume'), in mm."""
    rng = np.random.default_rng(seed)
    base = maps.build("plume", seed=seed).sources
    colors = ((255, 190, 64), (96, 160, 255))
    sources = tuple(
        Source(i, (x + rng.uniform(-0.02, 0.02)) * MM_PER_M,
               (y + rng.uniform(-0.02, 0.02)) * MM_PER_M,
               int(kind), colors[int(kind)])
        for i, (x, y, kind, _amount) in enumerate(base)
    )
    return PlumeWorld(seed, sources, ResourceState((50.0, 50.0), (True, True)))


def concentration_xy_mm(xy_mm, world: PlumeWorld):
    """Food/decoy concentration at body-derived XY samples; no source positions returned."""
    xy = np.asarray(xy_mm, dtype=np.float32).reshape(-1, 2) / np.float32(MM_PER_M)
    out = np.zeros((len(xy), 2), dtype=np.float32)
    for i, src in enumerate(world.sources):
        if i >= len(world.state.present) or not world.state.present[i] or world.state.remaining[i] <= 0:
            continue
        dx = xy[:, 0] - np.float32(src.x_mm / MM_PER_M)
        dy = xy[:, 1] - np.float32(src.y_mm / MM_PER_M)
        width = np.float32(0.1) + np.float32(0.15) * np.maximum(dx, np.float32(0))
        c = np.where(dx > 0,
                     np.exp(-(dy * dy) / (np.float32(2) * width * width)) * np.float32(0.1) / width,
                     np.exp(np.minimum(dx, np.float32(0)) / np.float32(0.1)))
        out[:, src.kind] += c
    return out


def antenna_xy_mm(root_xy_mm, heading_rad, forward_mm=0.0, lateral_mm=1.0):
    """Fixture helper: derive left/right antenna XY from a planar body pose."""
    root, h = np.asarray(root_xy_mm, np.float32), np.asarray(heading_rad, np.float32)
    fwd = np.stack((np.cos(h), np.sin(h)), axis=-1) * np.float32(forward_mm)
    left = np.stack((-np.sin(h), np.cos(h)), axis=-1) * np.float32(lateral_mm)
    return np.stack((root + fwd + left, root + fwd - left), axis=-2)


def render_rows(world: PlumeWorld):
    """Rows shared with the spectator: id, xy, kind, RGB, presence and remaining."""
    return tuple((s.source_id, s.x_mm, s.y_mm, s.kind, *s.color_rgb,
                  world.state.present[i], world.state.remaining[i])
                 for i, s in enumerate(world.sources))
