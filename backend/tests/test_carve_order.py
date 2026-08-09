"""Regression test documenting the intended road-overlap rule: where two
buffered roads overlap, the *deepest* carve should win (not accumulate).

NOTE: this asserts the intended contract with a reference implementation. The
current `core.mesh.carve_roads` subtracts depths cumulatively at overlaps, which
does NOT satisfy this rule — tracked as an open decision (see
project-management/DECISIONS.md, D-2 / FEATURE_REGISTER F-2). Runs without any
network calls."""

import numpy as np


def test_deepest_wins_on_overlap():
    dem = np.zeros((5, 5), dtype=float)

    # Shallow road covers a plus sign (center row + col).
    shallow = np.zeros((5, 5), dtype=bool)
    shallow[2, :] = True
    shallow[:, 2] = True

    # Deep road covers a 3x3 square in the center.
    deep = np.zeros((5, 5), dtype=bool)
    deep[1:4, 1:4] = True

    shallow_delta = 1.0
    deep_delta = 3.0

    # Intended result: shallow-only -1, deep-only -3, overlap -3 (deepest wins).
    expected = np.zeros((5, 5), dtype=float)
    expected[shallow] -= shallow_delta
    expected[deep] -= deep_delta
    overlap = shallow & deep
    expected[overlap] = -deep_delta

    # Reference "deepest wins" computation: max delta per pixel.
    per_pixel = np.zeros((5, 5), dtype=float)
    per_pixel[shallow] = np.maximum(per_pixel[shallow], shallow_delta)
    per_pixel[deep] = np.maximum(per_pixel[deep], deep_delta)
    got = dem - per_pixel

    assert np.allclose(got, expected)


if __name__ == "__main__":
    test_deepest_wins_on_overlap()
    print("OK")
