"""Checks for the fast_approximation coupling path.

In this mode the optimizer only sees the columns that `saved_columns` keeps,
while the coupling score it produces is scattered back to the width of the
complete alignment so the analyzers keep indexing it by residue number. Both
halves read the same mask -- the alignment object on the way in, the Zarr
archive's own attributes on the way out -- so the tests here mostly guard that
the two agree.

These build their own fixtures, so they run on a bare checkout.
"""
import numpy as np
import pandas as pd
import pytest
import zarr

from massivora.alignment import BinaryAlignment
from massivora.analyzer import ComplexAnalyzer
from massivora.executors.couple import BaseCouplingExecutor


def _alignment(matrix, saved_columns=None, lengthes=None):
    a = BinaryAlignment()
    a.matrix = np.asarray(matrix, dtype=np.int8)
    a.weights = [1.0] * len(matrix)
    a.Beff = float(len(matrix))
    if saved_columns is not None:
        a.saved_columns = saved_columns
    if lengthes is not None:
        a.lengthes = lengthes
    return a


# ---------------------------------------------------------------------------
# Drop_Masked_Gap_Columns
# ---------------------------------------------------------------------------

def test_drop_keeps_only_the_saved_columns():
    a = _alignment([[1, 2, 3],
                    [4, 5, 6]], saved_columns=[[0, 2]], lengthes=[3])

    assert a.Drop_Masked_Gap_Columns() is None
    np.testing.assert_array_equal(a.matrix, [[1, 3], [4, 6]])


def test_drop_offsets_local_indices_by_monomer():
    # Monomer A is 4 wide and monomer B 3, so B's local column 0 is global 4.
    a = _alignment([[0, 1, 2, 3, 4, 5, 6]],
                   saved_columns=[[0, 3], [0, 2]], lengthes=[4, 3])

    a.Drop_Masked_Gap_Columns()

    np.testing.assert_array_equal(a.matrix, [[0, 3, 4, 6]])


def test_drop_never_recomputes_the_mask():
    """The analyzers read the stored mask back out of the archive, so deriving
    a different one here would misattribute every coupling."""
    # Column 1 is gap-free and any gap-ratio pass would keep it, but the stored
    # mask leaves it out.
    a = _alignment([[1, 7, 3],
                    [4, 8, 6]], saved_columns=[[0, 2]], lengthes=[3])

    a.Drop_Masked_Gap_Columns()

    np.testing.assert_array_equal(a.matrix, [[1, 3], [4, 6]])
    assert a.saved_columns == [[0, 2]] and a.lengthes == [3]


def test_drop_invalidates_weights():
    """Identity must be recounted on the reduced matrix: agreement between two
    gaps in a near-empty column would otherwise dominate it."""
    a = _alignment([[1, 0], [2, 0], [3, 0], [4, 5]],
                   saved_columns=[[0]], lengthes=[2])

    a.Drop_Masked_Gap_Columns()

    assert a.weights is None
    assert a.Beff is None


def test_drop_is_idempotent():
    a = _alignment([[1, 2, 3],
                    [4, 5, 6]], saved_columns=[[0, 2]], lengthes=[3])
    a.Drop_Masked_Gap_Columns()
    reduced = a.matrix.copy()

    a.Drop_Masked_Gap_Columns()

    np.testing.assert_array_equal(a.matrix, reduced)


@pytest.mark.parametrize('saved_columns, lengthes', [
    (None, None),                       # nothing stored
    ([[0, 1]], [4, 3]),                 # fewer mask entries than monomers
    ([[0], [0, 9]], [4, 3]),            # a column outside its monomer
    ([[0, 1], [0]], [4, 4]),            # lengthes do not add up to the width
])
def test_drop_leaves_the_alignment_alone_on_unusable_metadata(saved_columns, lengthes):
    a = _alignment(np.arange(7).reshape(1, 7),
                   saved_columns=saved_columns, lengthes=lengthes)

    a.Drop_Masked_Gap_Columns()

    assert a.matrix.shape == (1, 7)
    assert a.weights is not None


def test_drop_does_nothing_when_the_mask_keeps_everything():
    a = _alignment([[1, 2], [3, 4]], saved_columns=[[0, 1]], lengthes=[2])

    a.Drop_Masked_Gap_Columns()

    assert a.matrix.shape == (2, 2)
    assert a.weights is not None


# ---------------------------------------------------------------------------
# expand_score
# ---------------------------------------------------------------------------

def test_expand_score_scatters_to_the_kept_columns():
    # kept globals: monomer A -> 0, 2 ; monomer B -> 3+0 = 3
    saved_columns, lengthes = [[0, 2], [0]], [3, 3]
    reduced = np.arange(1, 10, dtype=np.float16).reshape(3, 3)

    full = BaseCouplingExecutor.expand_score(reduced, saved_columns, lengthes)

    columns = [0, 2, 3]
    assert full.shape == (6, 6)
    assert full.dtype == reduced.dtype
    np.testing.assert_array_equal(full[np.ix_(columns, columns)], reduced)
    dropped = np.setdiff1d(np.arange(6), columns)
    assert not full[dropped, :].any()
    assert not full[:, dropped].any()


def test_expand_score_passes_through_when_nothing_was_dropped():
    reduced = np.arange(4, dtype=np.float16).reshape(2, 2)

    full = BaseCouplingExecutor.expand_score(reduced, [[0, 1]], [2])

    assert full is reduced


@pytest.mark.parametrize('saved_columns, lengthes', [
    (None, None),
    ([], []),
    ([[0, 1]], [4, 3]),
])
def test_expand_score_passes_through_on_unusable_metadata(saved_columns, lengthes):
    """load_pair could not have dropped anything either, so the score is
    already at full width."""
    reduced = np.arange(4, dtype=np.float16).reshape(2, 2)

    assert BaseCouplingExecutor.expand_score(reduced, saved_columns, lengthes) is reduced


def test_expand_score_refuses_a_mask_of_the_wrong_size():
    """A mask that does not match what was optimized would put every value in
    the wrong cell, so this must fail loudly rather than write the archive."""
    reduced = np.zeros((3, 3), dtype=np.float16)

    with pytest.raises(RuntimeError, match='misaligned'):
        BaseCouplingExecutor.expand_score(reduced, [[0, 2]], [4])


# ---------------------------------------------------------------------------
# The mask the optimizer drops by and the one the analyzer reads must agree
# ---------------------------------------------------------------------------

def _write_pair_zarr(path, n_a, n_b, saved_columns):
    """A minimal two-monomer alignment archive the analyzer can read."""
    grp = zarr.open_group(store=str(path))
    z = grp.create_array(name='align', shape=(4, n_a + n_b), dtype='int8')
    z.attrs.update({
        'saved_columns': saved_columns,
        'lengthes': [n_a, n_b],
        'species_list': ['Query'],
        'species_index_map': [0],
    })
    z[:] = np.ones((4, n_a + n_b), dtype=np.int8)
    return grp


def test_reduced_score_lands_where_the_analyzer_looks_for_it(tmp_path):
    """The end-to-end property: an alignment reduced by the stored mask, scored,
    then expanded from the archive's own attributes, must be picked up cell for
    cell by ComplexAnalyzer."""
    n_a, n_b = 6, 4
    saved_columns = [[0, 2, 5], [1, 3]]        # 3 kept in A, 2 in B
    store = tmp_path / 'pair'
    grp = _write_pair_zarr(store, n_a, n_b, saved_columns)

    align = BinaryAlignment(str(store))
    align.Drop_Masked_Gap_Columns()
    n_red = align.matrix.shape[1]
    assert n_red == 5

    # A score over the reduced columns, distinct in every cell.
    reduced = np.arange(1, n_red * n_red + 1, dtype=np.float16).reshape(n_red, n_red)
    # Exactly what collect_results does: re-read the mask from the archive.
    attrs = grp['align'].attrs
    full = BaseCouplingExecutor.expand_score(reduced, attrs.get('saved_columns'),
                                             attrs.get('lengthes'))
    assert full.shape == (n_a + n_b, n_a + n_b)

    z = grp.require_array(name='couplings', shape=full.shape, dtype='float16',
                          overwrite=True)
    z[:] = full

    inter = ComplexAnalyzer(str(store), chainA_length=n_a).interprotein_couplings

    # Monomer A owns the leading columns of the reduced frame and B the rest,
    # so this is the inter-protein block the analyzer should have found.
    n_kept_a = len(saved_columns[0])
    expected = pd.DataFrame(
        reduced[:n_kept_a, n_kept_a:],
        index=[c + 1 for c in saved_columns[0]],
        columns=[c + 1 for c in saved_columns[1]],
    )

    assert list(inter.index) == list(expected.index)
    assert list(inter.columns) == list(expected.columns)
    np.testing.assert_array_equal(inter.to_numpy().astype(np.float16),
                                  expected.to_numpy())
    # nothing selected may come from a column that was never optimized
    assert (inter.to_numpy() != 0).all()
