"""Deterministic contract check for the plume descriptor and maps.py oracle."""
from dataclasses import replace

import numpy as np

from agent import maps
from agent.body_engine import world_content as wc


def _maps_oracle(xy_mm, world):
    # Use the existing maps.py equation with the descriptor converted to metres.
    m = maps.build("plume", seed=world.seed)
    sources = np.asarray([[s.x_mm / wc.MM_PER_M, s.y_mm / wc.MM_PER_M,
                           s.kind, world.state.remaining[i]]
                          for i, s in enumerate(world.sources)], dtype=np.float64)[None]
    sources[0, :, 3] *= np.asarray(world.state.present, dtype=np.float64)
    return np.asarray([maps.concentration(
        m, np.asarray(point, dtype=np.float64)[None, :] / wc.MM_PER_M,
        sources, np.ones(1))[0] for point in xy_mm])


def check():
    w = wc.build(17)
    xy = np.asarray([[1000, 350], [1200, -350], [300, 350], [800, -100]], np.float32)
    got, ref = wc.concentration_xy_mm(xy, w), _maps_oracle(xy, w)
    err = float(np.max(np.abs(got - ref)))
    assert err <= 2e-7, f"maps.py plume mismatch {err:.3g}"
    print(f"PASS fixed-pose plume vs maps.py max abs err {err:.3e}; conversion 1000 mm/m")

    # A heading reversal exchanges physical left/right antenna locations.
    root, heading = np.asarray([950, 0], np.float32), np.float32(0.4)
    a = wc.antenna_xy_mm(root, heading, forward_mm=0.0, lateral_mm=1.2)
    b = wc.antenna_xy_mm(root, heading + np.pi, forward_mm=0.0, lateral_mm=1.2)
    left_vec = np.asarray([-np.sin(heading), np.cos(heading)], np.float32) * np.float32(1.2)
    assert np.allclose(a[0], root + left_vec, atol=2e-4)
    assert np.allclose(a[1], root - left_vec, atol=2e-4)
    assert np.allclose(a[0], b[1], atol=2e-4) and np.allclose(a[1], b[0], atol=2e-4)
    ca, cb = wc.concentration_xy_mm(a, w), wc.concentration_xy_mm(b, w)
    assert np.allclose(ca, _maps_oracle(a, w), atol=2e-7)
    assert np.allclose(cb, _maps_oracle(b, w), atol=2e-7)
    assert np.allclose(ca[0], cb[1], atol=2e-7) and np.allclose(ca[1], cb[0], atol=2e-7)
    print("PASS symmetric heading reversal swaps left/right antenna samples")

    empty = replace(w, state=replace(w.state, present=(False, True)))
    zero = replace(w, state=replace(w.state, remaining=(0.0, 50.0)))
    assert np.all(wc.concentration_xy_mm(xy, empty)[:, 0] == 0)
    assert np.all(wc.concentration_xy_mm(xy, zero)[:, 0] == 0)
    print("PASS absent and depleted food sources emit zero")

    w2 = wc.build(18)
    assert wc.render_rows(w) != wc.render_rows(w2)
    assert not np.allclose(wc.concentration_xy_mm(xy, w), wc.concentration_xy_mm(xy, w2))
    print("PASS changed seed changes shared render descriptor and concentration samples")

    # Mutant controls: swapping antenna indexing and dropping presence must fail.
    swapped_mutant_fails = not np.allclose(ca, ca[::-1])
    ignored_presence_fails = not np.allclose(wc.concentration_xy_mm(xy, empty),
                                              wc.concentration_xy_mm(xy, w))
    assert swapped_mutant_fails and ignored_presence_fails
    print("PASS mutants detected: antenna index swap; ignored presence flag")


if __name__ == "__main__":
    check()
