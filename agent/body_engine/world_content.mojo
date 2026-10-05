"""Plume equation ported from agent/maps.py:104-120; coordinates are world mm."""
from std.math import exp, min

comptime MM_PER_M = Float32(1000.0)


@always_inline
def plume_concentration(x_mm: Float32, y_mm: Float32, sx_mm: Float32, sy_mm: Float32,
                        present: Int32, remaining: Float32) -> Float32:
    """Return maps.py plume concentration at one sample; absent/depleted is zero."""
    if present == 0 or remaining <= 0:
        return 0
    var dx = (x_mm - sx_mm) / MM_PER_M
    var dy = (y_mm - sy_mm) / MM_PER_M
    if dx > 0:
        var width = 0.1 + 0.15 * dx
        return exp(-(dy * dy) / (2 * width * width)) * 0.1 / width
    return exp(min(dx, Float32(0)) / 0.1)


@always_inline
def sample_two(x_mm: Float32, y_mm: Float32,
               sx0_mm: Float32, sy0_mm: Float32, kind0: Int32, present0: Int32, rem0: Float32,
               sx1_mm: Float32, sy1_mm: Float32, kind1: Int32, present1: Int32, rem1: Float32
               ) -> SIMD[DType.float32, 2]:
    var food: Float32 = 0
    var decoy: Float32 = 0
    var c0 = plume_concentration(x_mm, y_mm, sx0_mm, sy0_mm, present0, rem0)
    var c1 = plume_concentration(x_mm, y_mm, sx1_mm, sy1_mm, present1, rem1)
    if kind0 == 0: food += c0
    else: decoy += c0
    if kind1 == 0: food += c1
    else: decoy += c1
    return SIMD[DType.float32, 2](food, decoy)
