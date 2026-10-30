"""The B-cap draw (concatenate.downsample) is reproducible.

A pair concatenated from its monomers with more than `downsample.to` rows keeps
the query row and `to` - 1 others. The draw used to be unseeded, so a rerun (or
a second worker) scored a different subsample. It is now seeded by
`concatenate.downsample.seed` and the pair's name alone (`utils.downsample_rng`),
so the kept rows do not depend on the process, the order or the machine.

A stored pair alignment is used as stored and never downsampled: the screen's
calibration and reference scored their B > 30,000 pairs uncapped from the store,
and this change does not touch that path.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

from massivora.alignment import BinaryAlignment, _downsample_rows
from massivora.utils import DOWNSAMPLE_SEED, downsample_rng

from test_shared_source import _pair, _project, _text_monomer


def _rows_kept(n_rows, to, seed):
    return _downsample_rows(n_rows, to, seed).tolist()


def test_rows_are_the_query_and_an_ascending_draw():
    rows = _rows_kept(1000, 50, downsample_rng(0, 'A-B'))
    assert rows[0] == 0 and len(rows) == 50 and len(set(rows)) == 50
    assert rows == sorted(rows) and 0 not in rows[1:] and max(rows) < 1000


def test_same_seed_and_name_give_the_same_rows():
    assert _rows_kept(1000, 50, downsample_rng(0, 'A-B')) == _rows_kept(1000, 50, downsample_rng(0, 'A-B'))
    assert _rows_kept(1000, 50, 7) == _rows_kept(1000, 50, 7)


def test_seed_and_name_both_change_the_rows():
    base = _rows_kept(1000, 50, downsample_rng(0, 'A-B'))
    assert base != _rows_kept(1000, 50, downsample_rng(1, 'A-B'))
    assert base != _rows_kept(1000, 50, downsample_rng(0, 'A-C'))


def test_the_draw_does_not_depend_on_the_process():
    """Python's str hash is salted per process; the pair seed must not be."""
    code = ('from massivora.utils import downsample_rng; '
            'print(downsample_rng(0, "O24939_HELPY-O25518_HELPY").integers(0, 2**31, 4).tolist())')
    env = {**os.environ, 'PYTHONHASHSEED': 'random'}
    outs = {subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True,
                           check=True).stdout for _ in range(2)}
    here = str(downsample_rng(0, 'O24939_HELPY-O25518_HELPY').integers(0, 2**31, 4).tolist())
    assert outs == {here + '\n'}


def test_negative_seed_is_rejected():
    with pytest.raises(ValueError, match='seed'):
        downsample_rng(-1, 'A-B')


def test_default_seed_is_0():
    assert DOWNSAMPLE_SEED == 0


def test_binary_alignment_downsample_is_reproducible():
    a, b = _pair(), _pair()
    assert a.matrix.shape[0] == 41
    a.Downsample_Randomly(to=20, seed=downsample_rng(0, 'A0-B0'))
    b.Downsample_Randomly(to=20, seed=downsample_rng(0, 'A0-B0'))
    assert a.matrix.shape[0] == 20
    np.testing.assert_array_equal(a.matrix, b.matrix)
    assert a.species_list == b.species_list and a.species_list[0] == 'Query'
    full = _pair()
    kept = _rows_kept(41, 20, downsample_rng(0, 'A0-B0'))
    np.testing.assert_array_equal(a.matrix, full.matrix[kept])
    assert a.species_list[1:] == [full.species_list[i] for i in kept[1:]]


def test_binary_alignment_no_op_when_not_larger():
    a = _pair()
    before = a.matrix.copy()
    a.Downsample_Randomly(to=41, seed=0)
    np.testing.assert_array_equal(a.matrix, before)


def test_text_alignment_downsample_is_reproducible():
    rng = np.random.default_rng(3)
    m = rng.integers(1, 21, size=(30, 12))
    a, b = _text_monomer('x', m), _text_monomer('x', m)
    a.Downsample_Randomly(to=10, seed=downsample_rng(5, 'x'))
    b.Downsample_Randomly(to=10, seed=downsample_rng(5, 'x'))
    ids = [r.id for r in a._records]
    assert ids == [r.id for r in b._records] and len(ids) == 10 and ids[0] == 'x0'
    assert ids == [f'x{i}' for i in _rows_kept(30, 10, downsample_rng(5, 'x'))]


def _stored(path):
    align = BinaryAlignment(path)
    return align.matrix.copy(), list(align.species_list)


@pytest.mark.parametrize('seed, same', [(0, True), (1, False)])
def test_concatenation_stage_stores_the_same_draw(tmp_path, seed, same):
    """`_run_concatenate` (the concatenate stage) twice into two trees."""
    from massivora.db import get_hashed_file_path
    from massivora.executors import concatenate
    _project(tmp_path, 'skip', n_pairs=1, shared_pairs=())
    out = []
    for i, s in enumerate((0, seed)):
        ecs = tmp_path / f'ecs{i}'
        os.makedirs(ecs)
        rec = concatenate._run_concatenate({'name': 'A0-B0', 'pid1': 'A0', 'pid2': 'B0'},
                                           str(tmp_path / 'msa'), str(ecs), downsample=True,
                                           downsample_to=20, reweight=False, downsample_seed=s)
        assert rec['number'] == 20
        out.append(_stored(get_hashed_file_path('A0-B0', rec.get('file_hash'), str(ecs))[0]))
    assert (np.array_equal(out[0][0], out[1][0]) and out[0][1] == out[1][1]) is same


def test_coupling_stage_concatenates_with_the_configured_seed(tmp_path):
    """`BaseCouplingExecutor.concatenate` (fused mode, a pair with no stored
    alignment): the configured seed and the name fix the rows."""
    pytest.importorskip('ost', exc_type=ImportError)    # the executors import the analyzer
    from massivora.executors import couple
    cfg = _project(tmp_path, 'skip', n_pairs=1, shared_pairs=())
    cfg['concatenate']['downsample'] = {'enabled': True, 'to': 20, 'seed': 7}
    pair = {'name': 'A0-B0', 'pid1': 'A0', 'pid2': 'B0'}
    ex = couple.BaseCouplingExecutor(cfg)
    try:
        assert ex.downsample_seed == 7
        a = ex.concatenate(pair, ex.alignment_dir, downsample=True, downsample_to=20)
        b = ex.concatenate(pair, ex.alignment_dir, downsample=True, downsample_to=20)
    finally:
        ex.close_db_connection()
    full = _pair(seed=0)
    kept = _rows_kept(41, 20, downsample_rng(7, 'A0-B0'))
    np.testing.assert_array_equal(a.matrix, b.matrix)
    np.testing.assert_array_equal(a.matrix, full.matrix[kept])
