"""Renderer-independent HUD layout values for the body engine spectator.

These helpers expose pixel extents and descriptor positions to the live owner;
glyph drawing remains in the existing renderer.
"""

@fieldwise_init
struct BarRect:
    var x: Float32
    var y: Float32
    var width: Float32
    var height: Float32


@fieldwise_init
struct Marker:
    var x: Float32
    var y: Float32
    var kind: Int


def normalized_extent(value: Float32, width: Float32) -> Float32:
    """Pixel fill width for a normalized energy, health, or stamina value."""
    var v = value
    if v < 0.0:
        v = 0.0
    if v > 1.0:
        v = 1.0
    return v * width


def bar_rect(x: Float32, y: Float32, width: Float32, height: Float32,
             value: Float32) -> BarRect:
    return BarRect(x, y, normalized_extent(value, width), height)


def project_source(source_x: Float32, source_y: Float32, target_x: Float32,
                   target_y: Float32, scale: Float32, center_x: Float32,
                   center_y: Float32, kind: Int) -> Marker:
    """Map a world XY source descriptor into the simple orthographic HUD plane."""
    return Marker(center_x + (source_x - target_x) * scale,
                  center_y - (source_y - target_y) * scale, kind)


def source_matches(marker: Marker, expected_x: Float32, expected_y: Float32,
                   expected_kind: Int) -> Bool:
    """Fixture comparison for source marker agreement, pixel tolerance 0.01."""
    return abs(marker.x - expected_x) <= 0.01 and abs(marker.y - expected_y) <= 0.01 and marker.kind == expected_kind


def main() raises:
    var energy = bar_rect(12.0, 20.0, 120.0, 8.0, 0.75)
    var health = bar_rect(12.0, 32.0, 120.0, 8.0, 0.25)
    var stamina = bar_rect(12.0, 44.0, 120.0, 8.0, 0.5)
    assert abs(energy.width - 90.0) < 1e-5
    assert abs(health.width - 30.0) < 1e-5
    assert abs(stamina.width - 60.0) < 1e-5
    var marker = project_source(100.0, -50.0, 0.0, 0.0, 0.5, 320.0, 180.0, 1)
    assert source_matches(marker, 370.0, 205.0, 1)
    assert marker.kind == 1 and marker.x == 370.0 and marker.y == 205.0
    print("PASS HUD bars: energy 90 px, health 30 px, stamina 60 px")
    print("PASS source marker: descriptor position and kind preserved")
