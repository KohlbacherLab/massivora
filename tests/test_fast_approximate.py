"""Checks for the --fast-approximate coupling path.

These deliberately avoid the alignment fixtures, so they run on a bare checkout.
"""
import numpy as np
import pytest

from massivora.alignment import BinaryAlignment
from massivora.main import _build_parser


def _alignment(matrix):
    a = BinaryAlignment()
    a.matrix = np.asarray(matrix, dtype=np.int8)
    a.weights = [1.0] * len(matrix)
    a.Beff = float(len(matrix))
    return a


def test_drop_gap_columns_keeps_only_informative_columns():
    # col 0: no gaps | col 1: 3/4 gaps | col 2: 1/4 gaps | col 3: all gaps
    a = _alignment([[1, 0, 0, 0],
                    [2, 0, 3, 0],
                    [3, 0, 4, 0],
                    [4, 5, 5, 0]])
    kept = a.Drop_Gap_Columns(gap_ratio=0.5)

    assert kept.tolist() == [0, 2]
    assert a.matrix.shape == (4, 2)
    np.testing.assert_array_equal(a.matrix[:, 0], [1, 2, 3, 4])


def test_drop_gap_columns_invalidates_weights():
    """Identity must be recomputed on the reduced matrix: agreement between two
    gaps in a near-empty column would otherwise dominate it."""
    a = _alignment([[1, 0], [2, 0], [3, 0], [4, 5]])
    a.Drop_Gap_Columns(gap_ratio=0.5)

    assert a.weights is None
    assert a.Beff is None


def test_drop_gap_columns_threshold_is_exclusive():
    # exactly half gaps -> not below the threshold -> dropped
    a = _alignment([[1, 0], [2, 0], [3, 7], [4, 8]])
    assert a.Drop_Gap_Columns(gap_ratio=0.5).tolist() == [0]


def test_cli_exposes_fast_approximate_on_couple_only():
    p = _build_parser()

    assert p.parse_args(['run', 'couple', 'c.yml', '--fast-approximate']).fast_approximate
    assert not p.parse_args(['run', 'couple', 'c.yml']).fast_approximate
    assert p.parse_args(['batch', 'couple', 'c.yml', '--fast-approximate']).fast_approximate

    # the flag only affects the coupling stage, so align must reject it
    with pytest.raises(SystemExit):
        p.parse_args(['run', 'align', 'c.yml', '--fast-approximate'])
