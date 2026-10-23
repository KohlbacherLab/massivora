"""The shared-source check from per-protein run tables (issue #15).

A pair concatenated with `+` is tested for shared-source rows from its two
monomers' run tables (the sorted unique residue runs of each species row),
which are built once per protein and may be saved for every process to map.
These tests check that the tables give exactly the booleans of the row-wise
check (`_shared_source_rows`) through the C++ merge and the numpy fallback,
that saved tables round-trip and are only reused for the matrix they were
built from, that nothing is written unless asked, and that a pair modified
after concatenation falls back to its rows.
"""
import logging
import os

import numpy as np
import pytest

from massivora import alignment, cpp_bindings
from massivora.alignment import (BinaryAlignment, RunTable, SharedSourceError,
                                 _shared_source_rows, alignment_fingerprint,
                                 resolve_run_table_cache, resolve_shared_source,
                                 shared_source_rows_cached)
from test_shared_source import LEN_A, LEN_B, _monomer, _pair_monomers

FIXTURES = os.path.join(os.path.dirname(__file__), 'fixtures', 'shared_source')
# A real monomer tree (`$B/data/helpy_msa` on data) for the pairs at the threshold
MONOMERS = os.environ.get('MASSIVORA_MONOMERS')
HAS_CPP = getattr(cpp_bindings, 'sharedSourceRows', None) is not None


def _flagged(pair, shared):
    """The species of the pair rows flagged by `shared` (the query excluded)."""
    return {pair.species_list[i + 1] for i in np.flatnonzero(shared)}


def _row_check(pair, k=10):
    """The booleans of the row-wise check for a concatenated pair."""
    part_a, part_b = pair._pair_blocks()
    return _shared_source_rows(part_a[1:], part_b[1:], k)


@pytest.fixture(params=['cpp', 'numpy'])
def merge(request, monkeypatch):
    """Both implementations of the merge; the C++ one only where it is built."""
    if request.param == 'cpp':
        if not HAS_CPP:
            pytest.skip('cpp_bindings.sharedSourceRows is not built')
    else:
        monkeypatch.setattr(cpp_bindings, 'sharedSourceRows', None, raising=False)
    return request.param


# -- the table and its file format ----------------------------------------------

def test_run_table_round_trip(tmp_path):
    rng = np.random.default_rng(3)
    matrix = rng.integers(-20, 21, size=(50, 40)).astype(np.int8)
    rows = np.array([0, 3, 7, 20, 49])
    fp = alignment_fingerprint(matrix, rows, [f's{i}' for i in rows])
    table = RunTable.build(matrix, rows, k=10, fingerprint=fp)
    assert table.fingerprint == fp and table.shape == (50, 40)
    assert len(table.offsets) == len(rows) + 1
    assert table.save(str(tmp_path / 'm'))
    assert os.path.exists(str(tmp_path / 'm' / 'shared_source_k10' / 'meta.npy'))

    loaded = RunTable.load(str(tmp_path / 'm'), 10, matrix.shape, fp)
    assert loaded is not None and loaded.fingerprint == fp and loaded.k == 10
    np.testing.assert_array_equal(loaded.codes, table.codes)
    np.testing.assert_array_equal(loaded.offsets, table.offsets)
    np.testing.assert_array_equal(loaded.rows, rows)
    assert isinstance(loaded.codes, np.memmap)         # mapped, not read
    np.testing.assert_array_equal(loaded.positions([7, 49, 0]), [2, 4, 0])
    with pytest.raises(KeyError):
        loaded.positions([7, 8])
    with pytest.raises(KeyError):
        loaded.positions([50])

    # built for another k, shape or matrix: not reused
    assert RunTable.load(str(tmp_path / 'm'), 9, matrix.shape, fp) is None
    assert RunTable.load(str(tmp_path / 'm'), 10, (50, 41), fp) is None
    other = alignment_fingerprint(matrix[::-1], rows, [])
    assert other != fp
    assert RunTable.load(str(tmp_path / 'm'), 10, matrix.shape, other) is None
    # without a fingerprint to check, the shape decides
    assert RunTable.load(str(tmp_path / 'm'), 10, matrix.shape) is not None
    assert RunTable.load(str(tmp_path / 'nowhere'), 10, matrix.shape, fp) is None

    # a second save keeps the table there; replace=True writes it again
    assert table.save(str(tmp_path / 'm'))
    assert RunTable.build(matrix[::-1], rows, 10, other).save(str(tmp_path / 'm'), replace=True)
    assert RunTable.load(str(tmp_path / 'm'), 10, matrix.shape, fp) is None
    assert RunTable.load(str(tmp_path / 'm'), 10, matrix.shape, other) is not None

    # an older format is ignored
    np.save(str(tmp_path / 'm' / 'shared_source_k10' / 'meta.npy'),
            np.array([1, 10, 50, 40], dtype=np.int64))
    assert RunTable.load(str(tmp_path / 'm'), 10, matrix.shape) is None
    RunTable.remove_all(str(tmp_path / 'm'))
    assert not os.path.exists(str(tmp_path / 'm' / 'shared_source_k10'))


def test_fingerprint_sees_content_shape_and_species():
    rng = np.random.default_rng(0)
    m = rng.integers(1, 21, size=(600, 30)).astype(np.int8)
    fp = alignment_fingerprint(m, range(600), [])
    assert len(fp) == RunTable.FINGERPRINT_BYTES
    assert alignment_fingerprint(m.copy(), range(600), []) == fp
    changed = m.copy()
    changed[-1, 0] = 21 - changed[-1, 0]                    # the last row is always hashed
    assert alignment_fingerprint(changed, range(600), []) != fp
    assert alignment_fingerprint(m, range(599), []) != fp
    assert alignment_fingerprint(m, range(600), ['x']) != fp
    assert alignment_fingerprint(m[:599], range(599), []) != fp
    assert alignment_fingerprint(np.zeros((0, 30), np.int8)) != fp


def test_resolve_run_table_cache(tmp_path):
    assert resolve_run_table_cache(None) is None
    assert resolve_run_table_cache('off', '/p') is None
    assert resolve_run_table_cache(' OFF ', '/p') is None
    assert resolve_run_table_cache('auto') is None               # no project: never writes
    assert resolve_run_table_cache('auto', '/p') == os.path.join('/p', 'shared_source_cache')
    assert resolve_run_table_cache('monomer', '/p') == 'monomer'
    assert resolve_run_table_cache('cache/ss', '/p') == os.path.join('/p', 'cache', 'ss')
    assert resolve_run_table_cache('/dev/shm/ss', '/p') == '/dev/shm/ss'
    assert resolve_run_table_cache('/dev/shm/ss') == '/dev/shm/ss'


# -- the cached check equals the row check --------------------------------------

CASES = {
    'clean': {},
    'shared': {'shared': (3, 7, 21, 40)},
    'one_below_threshold': {'shared': (5,)},
    'gap_interrupted': {'shared': (1, 2, 3), 'gap_in_copy': True},
    'lower_case': {'shared': (1, 2, 3), 'lower_case_copy': True},
    'all_shared': {'n_species': 5, 'shared': range(1, 6)},
}


@pytest.mark.parametrize('case', sorted(CASES))
def test_cached_check_equals_row_check(merge, case):
    m1, m2 = _pair_monomers(**CASES[case])
    pair = m1 + m2
    assert pair._pair_source is not None
    expected = _row_check(pair)
    cached = pair._shared_rows()
    np.testing.assert_array_equal(cached, expected)
    assert _flagged(pair, cached) == {f'sp{i}' for i in CASES[case].get('shared', ())}
    # the public calls agree with a pair that has no tables (loaded, say)
    plain = m1 + m2
    plain._pair_source = None
    assert pair.Is_Shared_Source() == plain.Is_Shared_Source()
    assert pair.shared_source_share == pytest.approx(plain.shared_source_share)
    # the monomers kept their tables, built once, in memory
    assert m1._run_table is not None and m2._run_table is not None
    assert m1._run_table is m1.run_table()


def test_cached_check_on_species_with_several_rows(merge):
    """Monomers with several rows per species: the pair takes each species'
    first row, and the tables hold just those rows."""
    rng = np.random.default_rng(7)
    species = [f'sp{i}' for i in range(1, 31)]
    monomers = []
    for L, seed in ((LEN_A, 1), (LEN_B, 2)):
        rows_per_species = rng.integers(1, 4, size=31)
        rows_per_species[0] = 1
        starts = np.concatenate(([0], np.cumsum(rows_per_species)[:-1]))
        matrix = np.random.default_rng(seed).integers(1, 21, size=(int(rows_per_species.sum()), L))
        m = _monomer(matrix, species)
        m.species_index_map = starts.tolist()
        monomers.append(m)
    a, b = monomers
    for s in (4, 9, 17):                     # copies of A's species row into B's
        b.matrix[b.species_index_map[s], 5:20] = a.matrix[a.species_index_map[s], 10:25]
    pair = a + b
    expected = _row_check(pair)
    np.testing.assert_array_equal(pair._shared_rows(), expected)
    assert _flagged(pair, expected) == {'sp4', 'sp9', 'sp17'}
    assert len(a.run_table().rows) == 31 and len(b.run_table().rows) == 31
    np.testing.assert_array_equal(a.run_table().rows, a.species_index_map)
    extra = sorted(set(range(a.matrix.shape[0])) - set(a.species_index_map))
    assert extra                             # some species have more rows
    with pytest.raises(KeyError):            # which the table does not hold
        a.run_table().positions([extra[0]])


def test_pair_modified_after_concatenation_uses_its_rows(merge):
    pair = _pair_from(shared=(2, 4, 6))
    assert pair.Remove_Shared_Source() == 3
    assert pair._pair_source is None
    assert not pair.Is_Shared_Source() and pair.shared_source_share == 0
    np.testing.assert_array_equal(pair._shared_rows(), _row_check(pair))

    pair = _pair_from(shared=(2, 4, 6, 8, 10, 12, 14, 16))
    np.random.seed(0)
    pair.Downsample_Randomly(to=25)
    assert pair._pair_source is None
    np.testing.assert_array_equal(pair._shared_rows(), _row_check(pair))
    assert pair.Is_Shared_Source(threshold=1e-9) == bool(_row_check(pair).any())

    pair = _pair_from(shared=(2, 4, 6))
    pair.matrix[1:30, 0] = 0
    pair.matrix[1:30, LEN_A + 1] = 0
    pair.Mask_Gap_Columns(0.5)
    assert pair._pair_source is not None     # masking does not touch the matrix
    pair.Drop_Masked_Gap_Columns()
    assert pair._pair_source is None
    np.testing.assert_array_equal(pair._shared_rows(), _row_check(pair))
    assert pair.Is_Shared_Source() and pair.Remove_Shared_Source() == 3


def _pair_from(**kwargs):
    m1, m2 = _pair_monomers(**kwargs)
    return m1 + m2


def test_pair_loaded_from_a_store_uses_its_rows(tmp_path, merge):
    pair = _pair_from(shared=(2, 4, 6))
    pair.To_Zarr(str(tmp_path / 'pair'))
    loaded = BinaryAlignment(str(tmp_path / 'pair'))
    assert loaded._pair_source is None
    np.testing.assert_array_equal(loaded._shared_rows(), pair._shared_rows())
    assert loaded.Is_Shared_Source() and loaded.shared_source_share == pytest.approx(3 / 40)


# -- tables on disk --------------------------------------------------------------

def _store(tmp_path, **kwargs):
    m1, m2 = _pair_monomers(**kwargs)
    os.makedirs(str(tmp_path / 'msa'), exist_ok=True)
    m1.To_Zarr(str(tmp_path / 'msa' / 'A'))
    m2.To_Zarr(str(tmp_path / 'msa' / 'B'))
    return str(tmp_path / 'msa' / 'A'), str(tmp_path / 'msa' / 'B')


def _tables_under(root):
    return sorted(os.path.relpath(d, root) for d, _, files in os.walk(root)
                  if 'meta.npy' in files)


def test_nothing_is_written_unless_asked(tmp_path, merge):
    path_a, path_b = _store(tmp_path, shared=(1, 2, 3))
    pair = BinaryAlignment(path_a) + BinaryAlignment(path_b)
    assert pair.Is_Shared_Source()
    assert resolve_shared_source(BinaryAlignment(path_a) + BinaryAlignment(path_b), 'fix',
                                 cache=None).matrix.shape[0] == 38
    for setting in (None, 'off', 'auto'):      # 'auto' is resolved by the executors
        BinaryAlignment(path_a).run_table(cache=setting)
    assert _tables_under(str(tmp_path)) == []


def test_tables_are_saved_once_and_mapped_by_the_next_load(tmp_path, merge):
    path_a, path_b = _store(tmp_path, shared=(1, 2, 3))
    cache = str(tmp_path / 'cache')
    a, b = BinaryAlignment(path_a), BinaryAlignment(path_b)
    pair = a + b
    assert pair.Is_Shared_Source(cache=cache) and pair.shared_source_share == pytest.approx(3 / 40)
    assert _tables_under(str(tmp_path)) == [os.path.join('cache', 'A', 'shared_source_k10'),
                                            os.path.join('cache', 'B', 'shared_source_k10')]
    assert not isinstance(a._run_table.codes, np.memmap)      # built here

    a2, b2 = BinaryAlignment(path_a), BinaryAlignment(path_b)
    pair2 = a2 + b2
    np.testing.assert_array_equal(pair2._shared_rows(cache=cache), pair._shared_rows(cache=cache))
    assert isinstance(a2._run_table.codes, np.memmap)         # mapped from the cache
    assert a2._run_table.fingerprint == a2.fingerprint() == a.fingerprint()
    mtime = os.path.getmtime(os.path.join(cache, 'A', 'shared_source_k10', 'meta.npy'))
    pair3 = BinaryAlignment(path_a) + BinaryAlignment(path_b)
    assert pair3.Is_Shared_Source(cache=cache)
    assert os.path.getmtime(os.path.join(cache, 'A', 'shared_source_k10', 'meta.npy')) == mtime

    # 'monomer' writes next to the monomers
    pair4 = BinaryAlignment(path_a) + BinaryAlignment(path_b)
    assert pair4.Is_Shared_Source(cache='monomer')
    assert os.path.exists(os.path.join(path_a, 'shared_source_k10', 'meta.npy'))
    assert os.path.exists(os.path.join(path_b, 'shared_source_k10', 'meta.npy'))
    # ... and the old loader ignores the extra directory
    assert BinaryAlignment(path_a).matrix.shape == a.matrix.shape


def test_stale_tables_are_not_reused(tmp_path, merge):
    """A monomer regenerated with the same shape gets new tables."""
    path_a, path_b = _store(tmp_path, shared=(1, 2, 3), seed=0)
    cache = str(tmp_path / 'cache')
    pair = BinaryAlignment(path_a) + BinaryAlignment(path_b)
    assert pair.Is_Shared_Source(cache=cache)
    old_fp = BinaryAlignment(path_a).fingerprint()

    # regenerate both monomers: same shape, other residues, the shared
    # species are now 10, 11, 12 and 13
    m1, m2 = _pair_monomers(shared=(10, 11, 12, 13), seed=1)
    m1.To_Zarr(path_a, overwrite=True)
    m2.To_Zarr(path_b, overwrite=True)
    a, b = BinaryAlignment(path_a), BinaryAlignment(path_b)
    assert a.matrix.shape == pair._pair_source[0].matrix.shape and a.fingerprint() != old_fp
    fresh = a + b
    np.testing.assert_array_equal(fresh._shared_rows(cache=cache), _row_check(fresh))
    assert _flagged(fresh, fresh._shared_rows(cache=cache)) == {'sp10', 'sp11', 'sp12', 'sp13'}
    # the stale table was replaced by the new one
    assert RunTable.load(os.path.join(cache, 'A'), 10, a.matrix.shape, a.fingerprint()) is not None
    assert RunTable.load(os.path.join(cache, 'A'), 10, a.matrix.shape, old_fp) is None


def test_overwriting_a_store_removes_its_tables(tmp_path):
    path_a, path_b = _store(tmp_path, shared=(1, 2, 3))
    assert (BinaryAlignment(path_a) + BinaryAlignment(path_b)).Is_Shared_Source(cache='monomer')
    assert os.path.exists(os.path.join(path_a, 'shared_source_k10'))
    m1, _ = _pair_monomers(seed=5)
    m1.To_Zarr(path_a, overwrite=True)
    assert not os.path.exists(os.path.join(path_a, 'shared_source_k10'))
    assert os.path.exists(os.path.join(path_b, 'shared_source_k10'))


def test_unwritable_cache_warns_once_and_falls_back(tmp_path, merge, caplog, monkeypatch):
    path_a, path_b = _store(tmp_path, shared=(1, 2, 3))
    blocker = tmp_path / 'blocked'
    blocker.write_text('not a directory')
    cache = str(blocker / 'tables')
    monkeypatch.setattr(alignment, '_run_table_save_warned', set())
    with caplog.at_level(logging.WARNING, logger='massivora.alignment'):
        pair = BinaryAlignment(path_a) + BinaryAlignment(path_b)
        assert pair.Is_Shared_Source(cache=cache)
        assert pair.shared_source_share == pytest.approx(3 / 40)
        np.testing.assert_array_equal(pair._shared_rows(cache=cache), _row_check(pair))
    warnings = [r for r in caplog.records if 'Cannot save the shared-source run table' in r.message]
    assert len(warnings) == 1                 # two monomers, one warning per cache root
    assert _tables_under(str(tmp_path)) == []


# -- real pairs at the threshold ---------------------------------------------------

@pytest.mark.skipif(not os.path.isdir(os.path.join(FIXTURES, 'HCPB_HELPY')),
                    reason='tests/fixtures/shared_source is not checked out')
def test_real_paralogue_pair_above_the_threshold(merge, tmp_path):
    """HCPB-HCPE (H. pylori): 121 of 1,273 species (9.51 %) pair one sequence
    with itself. The fixture holds the monomers' query and common-species rows."""
    a = BinaryAlignment(os.path.join(FIXTURES, 'HCPB_HELPY'))
    b = BinaryAlignment(os.path.join(FIXTURES, 'HCPE_HELPY'))
    pair = a + b
    assert pair.matrix.shape[0] == 1274
    expected = _row_check(pair)
    np.testing.assert_array_equal(pair._shared_rows(), expected)
    assert expected.sum() == 121
    assert pair.Is_Shared_Source() and pair.shared_source_share == pytest.approx(121 / 1273)
    with pytest.raises(SharedSourceError, match='9.5%'):
        resolve_shared_source(a + b, 'skip', 'HCPB_HELPY-HCPE_HELPY')
    # fresh monomers: `a` and `b` already hold their tables in memory
    fixed = resolve_shared_source(
        BinaryAlignment(os.path.join(FIXTURES, 'HCPB_HELPY')) +
        BinaryAlignment(os.path.join(FIXTURES, 'HCPE_HELPY')),
        'fix', 'HCPB_HELPY-HCPE_HELPY', cache=str(tmp_path))
    assert fixed.matrix.shape[0] == 1274 - 121
    # through the tables saved by the fix call as well
    pair2 = BinaryAlignment(os.path.join(FIXTURES, 'HCPB_HELPY')) + \
        BinaryAlignment(os.path.join(FIXTURES, 'HCPE_HELPY'))
    np.testing.assert_array_equal(pair2._shared_rows(cache=str(tmp_path)), expected)
    assert isinstance(pair2._pair_source[0]._run_table.codes, np.memmap)


@pytest.mark.skipif(not MONOMERS or not os.path.isdir(os.path.join(MONOMERS or '', 'GLUP_HELPY')),
                    reason='set MASSIVORA_MONOMERS to the H. pylori monomer tree')
@pytest.mark.parametrize('pair_names, n_shared, n_common', [
    (('GLUP_HELPY', 'O25780_HELPY'), 554, 12487),       # 4.44 %: below the 5 % cut
    (('HCPB_HELPY', 'HCPE_HELPY'), 121, 1273),          # 9.51 %: skipped
])
def test_real_pairs_at_the_threshold_from_the_monomer_tree(merge, tmp_path, pair_names, n_shared, n_common):
    a = BinaryAlignment(os.path.join(MONOMERS, pair_names[0]))
    b = BinaryAlignment(os.path.join(MONOMERS, pair_names[1]))
    pair = a + b
    assert pair.matrix.shape[0] == n_common + 1
    expected = _row_check(pair)
    assert expected.sum() == n_shared
    np.testing.assert_array_equal(pair._shared_rows(cache=str(tmp_path)), expected)
    assert pair.Is_Shared_Source(cache=str(tmp_path)) == (n_shared / n_common >= 0.05)
    pair2 = BinaryAlignment(os.path.join(MONOMERS, pair_names[0])) + \
        BinaryAlignment(os.path.join(MONOMERS, pair_names[1]))
    np.testing.assert_array_equal(pair2._shared_rows(cache=str(tmp_path)), expected)
    assert isinstance(pair2._pair_source[0]._run_table.codes, np.memmap)
    assert _tables_under(str(tmp_path)) == [
        os.path.join(pair_names[0], 'shared_source_k10'), os.path.join(pair_names[1], 'shared_source_k10')]
