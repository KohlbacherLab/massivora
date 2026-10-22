"""Checks for shared-source pairs: concatenations that glue a sequence to itself.

When the homologue searches of both proteins recruit the same sequence in a
species, its two halves share long exact runs of residues. These tests build
small synthetic pairs where chosen species do that, and check detection,
removal, the guard against non-pair alignments and how both concatenation
stages act on the 'skip' and 'fix' strategies.
"""
import os
import sqlite3

import numpy as np
import pytest

from massivora.alignment import (BinaryAlignment, SharedSourceError, TextAlignment,
                                 check_shared_source_strategy, resolve_shared_source)
from massivora.db import STATUS, create_job_db

LEN_A, LEN_B = 60, 50
TEST1_ZARR = os.path.join(os.path.dirname(__file__), "test1.zarr")
TEST2_ZARR = os.path.join(os.path.dirname(__file__), "test2.zarr")


def _monomer(matrix, species):
    """A one-protein BinaryAlignment with one row per species (the query first)."""
    align = BinaryAlignment()
    align.matrix = np.asarray(matrix, dtype=np.int8)
    align.species_list = ['Query'] + list(species)
    align.species_index_map = list(range(len(align.species_list)))
    align.species_list_map = {s: i for i, s in enumerate(align.species_list)}
    align.saved_columns = [list(range(align.matrix.shape[1]))]
    align.lengthes = [align.matrix.shape[1]]
    return align


def _pair_monomers(n_species=40, shared=(), seed=0, gap_in_copy=False, lower_case_copy=False):
    """
    Two monomers over the same species. For each species in `shared`, the row
    of B carries a 15-residue run copied from the row of A.
    """
    rng = np.random.default_rng(seed)
    a = rng.integers(1, 21, size=(n_species + 1, LEN_A))
    b = rng.integers(1, 21, size=(n_species + 1, LEN_B))
    for row in shared:
        run = a[row, 10:25].copy()
        if lower_case_copy:
            run = -run
        if gap_in_copy:
            # the copied run is interrupted by two gap columns in B
            b[row, 5:22] = np.concatenate([run[:6], [0, 0], run[6:15]])
        else:
            b[row, 5:20] = run
    species = [f'sp{i}' for i in range(1, n_species + 1)]
    return _monomer(a, species), _monomer(b, species)


def _pair(**kwargs):
    m1, m2 = _pair_monomers(**kwargs)
    return m1 + m2


def _text_monomer(prefix, matrix):
    """A TextAlignment of one protein whose records carry OX= taxids 1..n."""
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
    letters = np.array(list('-ACDEFGHIKLMNPQRSTVWY'))
    records = [SeqRecord(Seq(''.join(letters[row])), id=f'{prefix}{i}', name=f'{prefix}{i}',
                         description=f'OS=Species {i} OX={i + 1000}')
               for i, row in enumerate(np.asarray(matrix))]
    return TextAlignment(records)


# -- detection and removal on BinaryAlignment ---------------------------------

def test_shared_source_species_are_detected_and_removed():
    shared = (3, 7, 21, 40)
    pair = _pair(shared=shared)
    assert pair.Is_Shared_Source()
    assert pair.shared_source_share == pytest.approx(len(shared) / 40)

    pair.Reweight_Sequence(x=0.8)
    query = pair.matrix[0].copy()
    assert pair.Remove_Shared_Source() == len(shared)
    assert pair.matrix.shape == (41 - len(shared), LEN_A + LEN_B)
    np.testing.assert_array_equal(pair.matrix[0], query)
    assert not {f'sp{i}' for i in shared} & set(pair.species_list)
    assert pair.species_list[0] == 'Query'
    assert pair.species_index_map == list(range(pair.matrix.shape[0]))
    assert all(pair.species_index_map[pair.species_list_map[s]] == i
               for i, s in enumerate(pair.species_list))
    # weights of the full alignment no longer apply
    assert pair.weights is None and pair.Beff is None
    assert not pair.Is_Shared_Source() and pair.shared_source_share == 0


def test_gaps_and_lower_case_do_not_hide_a_shared_run():
    gapped = _pair(shared=(1, 2, 3), gap_in_copy=True)
    assert gapped.Is_Shared_Source() and gapped.shared_source_share == pytest.approx(3 / 40)
    lower = _pair(shared=(1, 2, 3), lower_case_copy=True)
    assert lower.Is_Shared_Source() and lower.shared_source_share == pytest.approx(3 / 40)


def test_unrelated_rows_and_a_low_share_are_not_flagged():
    clean = _pair()
    assert not clean.Is_Shared_Source() and clean.shared_source_share == 0
    assert clean.Remove_Shared_Source() == 0
    one = _pair(shared=(5,))                        # 1 of 40 species: below 5%
    assert not one.Is_Shared_Source() and one.shared_source_share == pytest.approx(1 / 40)
    assert one.Is_Shared_Source(threshold=0.02)


def test_pairs_with_dropped_gap_columns_are_split_by_their_saved_columns():
    pair = _pair(shared=(2, 4, 6))
    pair.matrix[1:30, 0] = 0                        # a mostly-gap column in A ...
    pair.matrix[1:30, LEN_A + 1] = 0                # ... and one in B
    pair.Mask_Gap_Columns(0.5)
    pair.Drop_Masked_Gap_Columns()
    assert pair.matrix.shape[1] == LEN_A + LEN_B - 2
    assert pair.Is_Shared_Source() and pair.Remove_Shared_Source() == 3


def test_a_real_pair_of_unrelated_proteins_is_clean():
    pair = BinaryAlignment(TEST1_ZARR) + BinaryAlignment(TEST2_ZARR)
    assert not pair.Is_Shared_Source() and pair.shared_source_share == 0


# -- only for alignments concatenated from two proteins ------------------------

def test_shared_source_checks_need_a_pair():
    monomer, _ = _pair_monomers()
    with pytest.raises(ValueError, match='two proteins'):
        monomer.Is_Shared_Source()
    with pytest.raises(ValueError, match='two proteins'):
        monomer.Remove_Shared_Source()
    triple = _pair() + _pair_monomers()[0]
    with pytest.raises(ValueError, match='two proteins'):
        triple.Is_Shared_Source()

    text = _text_monomer('A', _pair_monomers()[0].matrix)
    with pytest.raises(ValueError, match='two proteins'):
        text.Is_Shared_Source()
    with pytest.raises(ValueError, match='two proteins'):
        text.Remove_Shared_Source()


# -- TextAlignment -------------------------------------------------------------

def test_text_alignment_pair():
    m1, m2 = _pair_monomers(shared=(2, 9, 30))
    pair = _text_monomer('A', m1.matrix) + _text_monomer('B', m2.matrix)
    assert pair.lengthes == [LEN_A, LEN_B]
    assert pair.Is_Shared_Source() and pair.shared_source_share == pytest.approx(3 / 40)
    assert pair.Remove_Shared_Source() == 3
    assert len(pair) == 38 and pair.lengthes == [LEN_A, LEN_B]
    assert not pair.Is_Shared_Source()

    # the two widths survive a round trip from the binary pair
    text = _pair(shared=(2, 9, 30)).To_Text(restore_gaps=True)
    assert text.lengthes == [LEN_A, LEN_B]
    assert text.Is_Shared_Source() and text.Remove_Shared_Source() == 3


# -- the strategy --------------------------------------------------------------

def test_resolve_shared_source():
    with pytest.raises(SharedSourceError, match='shared-source'):
        resolve_shared_source(_pair(shared=(1, 2, 3)), 'skip', 'A-B')
    fixed = resolve_shared_source(_pair(shared=(1, 2, 3)), 'FIX', 'A-B')
    assert fixed.matrix.shape[0] == 38
    clean = _pair()
    assert resolve_shared_source(clean, 'skip').matrix.shape[0] == 41
    with pytest.raises(SharedSourceError, match='no species is left'):
        resolve_shared_source(_pair(n_species=5, shared=range(1, 6)), 'fix', 'A-B')
    with pytest.raises(ValueError, match='shared-source_strategy'):
        check_shared_source_strategy('drop')


# -- the concatenation stage -----------------------------------------------------

def _project(tmp_path, strategy=None, n_pairs=2, shared_pairs=(1,)):
    """Monomers A<i>, B<i> on disk and a job database with pairs A<i>-B<i>;
    the pairs in `shared_pairs` are shared-source."""
    concatenate = {'downsample': {'enabled': False, 'to': 30000}}
    if strategy is not None:
        concatenate['shared-source_strategy'] = strategy
    cfg = {
        'project': {'project_path': str(tmp_path)},
        'paths': {'monomers': 'msa', 'couplings': 'ecs'},
        'database': {'backend': 'sqlite', 'filename': 'job.db'},
        'align': {'db_table': 'alignments', 'col_gap_threshold': 0.5},
        'concatenate': concatenate,
        'couple': {'db_table': 'couplings', 'claim_batch_size': 8},
    }
    os.makedirs(tmp_path / 'msa', exist_ok=True)
    os.makedirs(tmp_path / 'ecs', exist_ok=True)
    for i in range(n_pairs):
        m1, m2 = _pair_monomers(shared=(1, 2, 3, 4) if i in shared_pairs else (), seed=i)
        m1.To_Zarr(str(tmp_path / 'msa' / f'A{i}'))
        m2.To_Zarr(str(tmp_path / 'msa' / f'B{i}'))
    create_job_db(cfg)
    conn = sqlite3.connect(tmp_path / 'job.db')
    conn.executemany('INSERT INTO alignments (pid, length) VALUES (?, ?)',
                     [(f'A{i}', LEN_A) for i in range(n_pairs)]
                     + [(f'B{i}', LEN_B) for i in range(n_pairs)])
    conn.executemany('INSERT INTO couplings (name, pid1, pid2, status, length) VALUES (?, ?, ?, ?, ?)',
                     [(f'A{i}-B{i}', f'A{i}', f'B{i}', STATUS['NOOPT'], LEN_A + LEN_B)
                      for i in range(n_pairs)])
    conn.commit()
    conn.close()
    return cfg


def _rows(tmp_path):
    conn = sqlite3.connect(tmp_path / 'job.db')
    conn.row_factory = sqlite3.Row
    rows = {row['name']: dict(row) for row in conn.execute('SELECT * FROM couplings')}
    conn.close()
    return rows


@pytest.mark.parametrize('strategy', ['skip', 'fix'])
def test_concatenation_stage(tmp_path, strategy):
    from massivora.executors import concatenate
    cfg = _project(tmp_path, strategy)
    results = [concatenate._run_concatenate(
        {'name': f'A{i}-B{i}', 'pid1': f'A{i}', 'pid2': f'B{i}'},
        str(tmp_path / 'msa'), str(tmp_path / 'ecs'), reweight=False,
        shared_source_strategy=strategy) for i in range(2)]
    concatenate._db_apply_results(cfg, results)
    rows = _rows(tmp_path)

    assert rows['A0-B0']['number'] == 41 and rows['A0-B0']['status'] == STATUS['NOOPT']
    if strategy == 'skip':
        assert results[1] == {'name': 'A1-B1', 'status': STATUS['FAILED']}
        assert rows['A1-B1']['status'] == STATUS['FAILED']
        assert rows['A1-B1']['file_hash'] is None
    else:
        assert rows['A1-B1']['number'] == 41 - 4
        assert rows['A1-B1']['status'] == STATUS['NOOPT']


# -- the coupling stage ------------------------------------------------------------

@pytest.fixture
def couple():
    pytest.importorskip('ost', exc_type=ImportError)    # the executors import the analyzer
    from massivora.executors import couple
    return couple


def _concatenating_executor(couple, cfg, calls):
    """A coordinator-driven executor that concatenates each pair on the fly and
    returns a placeholder score, recording every attempt in `calls`."""
    from dask import delayed

    class ConcatenatingExecutor(couple.BaseCouplingExecutor):
        def collect(self, pair):
            with open(calls, 'a') as f:
                f.write(pair['name'] + '\n')
            align = self.concatenate(pair, self.alignment_dir)
            n = int(sum(align.lengthes))
            params = {'name': pair['name'], 'saved_columns': align.saved_columns,
                      'lengthes': align.lengthes}
            return {'id': pair['id'], 'effnumber': 1.0, 'length': n,
                    'number': int(align.matrix.shape[0]), 'file_hash': None, 'pcontact': 0.0,
                    **self._score_fields(np.zeros((n, n), dtype=np.float16), params)}

        def calc_graph(self, pair, priority=0, **kwargs):
            return delayed(self.collect)(pair)

    return ConcatenatingExecutor(cfg)


@pytest.mark.parametrize('strategy', [None, 'fix'])
def test_coupling_stage(tmp_path, couple, strategy):
    import signal

    from dask.distributed import Client
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    cfg = _project(tmp_path, strategy)
    calls = tmp_path / 'calls.txt'
    executor = _concatenating_executor(couple, cfg, calls)
    try:
        with Client(processes=False, dashboard_address=None) as client:
            executor.run(client=client)
    finally:
        for sig, handler in saved.items():
            signal.signal(sig, handler)

    rows = _rows(tmp_path)
    attempts = calls.read_text().split()
    assert rows['A0-B0']['status'] == STATUS['DONE'] and rows['A0-B0']['number'] == 41
    if strategy is None:                             # skip is the default
        assert rows['A1-B1']['status'] == STATUS['FAILED']
        assert attempts.count('A1-B1') == 1          # marked FAILED, not retried
    else:
        assert rows['A1-B1']['status'] == STATUS['DONE']
        assert rows['A1-B1']['number'] == 41 - 4
