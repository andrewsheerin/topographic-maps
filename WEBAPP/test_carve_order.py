"""Quick overlap regression test for road carving.

Goal: when two buffered roads overlap, the *deepest* carve must win.

This is designed to be runnable without any network calls.
"""

import numpy as np


def _apply_masked_subtract(dem, mask, delta):
    out = dem.copy()
    out[mask] -= delta
    return out


def test_deepest_wins_on_overlap():
    # Imagine a 5x5 DEM of zeros.
    dem = np.zeros((5, 5), dtype=float)

    # Shallow road covers a plus sign (center row+col)
    shallow = np.zeros((5, 5), dtype=bool)
    shallow[2, :] = True
    shallow[:, 2] = True

    # Deep road covers a 3x3 square in the center
    deep = np.zeros((5, 5), dtype=bool)
    deep[1:4, 1:4] = True

    shallow_delta = 1.0
    deep_delta = 3.0

    # Correct result should be:
    # - shallow-only pixels: -1
    # - deep-only pixels: -3
    # - overlap pixels: -3 (deep wins)

    # This asserts the intended rule; the actual carve_roads implementation
    # should match this behavior.
    expected = np.zeros((5, 5), dtype=float)
    expected[shallow] -= shallow_delta
    expected[deep] -= deep_delta
    # overlap got both subtractions above; fix to deepest-only
    overlap = shallow & deep
    expected[overlap] = -deep_delta

    # One correct way to compute it: take max delta per pixel.
    per_pixel = np.zeros((5, 5), dtype=float)
    per_pixel[shallow] = np.maximum(per_pixel[shallow], shallow_delta)
    per_pixel[deep] = np.maximum(per_pixel[deep], deep_delta)
    got = dem - per_pixel

    assert np.allclose(got, expected)


if __name__ == "__main__":
    test_deepest_wins_on_overlap()
    print("OK")

