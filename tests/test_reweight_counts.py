"""reweightNeighbourCounts against a brute-force count, at widths around the
192-position chunks the C++ core counts in and at thresholds that settle early
either way."""
import numpy as np
import pytest

from massivora import cpp_bindings


@pytest.mark.parametrize("N", [1, 191, 192, 193, 385, 1000])
def test_neighbour_counts_match_brute_force(N):
    rng = np.random.default_rng(N)
    parents = rng.integers(0, 21, (6, N), dtype=np.int8)
    A = parents[rng.integers(0, 6, 120)]
    flip = rng.random(A.shape) < rng.uniform(0.0, 0.5, (120, 1))
    A[flip] = rng.integers(0, 21, int(flip.sum()), dtype=np.int8)
    matches = (A[:, None, :] == A[None, :, :]).sum(-1)
    for x in (0.0, 0.3, 0.8, 0.95, 1.0):
        t = x * N
        want = (matches >= t).sum(1)
        got = np.asarray(cpp_bindings.reweightNeighbourCounts(A, t, 2))
        np.testing.assert_array_equal(got, want, err_msg=f"N={N} x={x}")
