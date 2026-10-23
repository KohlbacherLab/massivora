"""A degenerate alignment (B <= 3 on the screen: one or two distinct sequences)
gives an all-zero inter-protein block, which cannot be normalised. The
calibrator must then return a defined, score-less result instead of raising
through `collect_score` (the pair ends DONE without a score, not FAILED).

What is stored is a DONE row with NULL log_lr_raw/pcontact: the reason is
logged, not persisted, so the calibration dict carries no key that is not a
column of the couplings table (`massiveilance._coupling_result_info` spreads it
into a row update, and a key with no column aborts the recovery)."""
import logging
import math
import sqlite3

import numpy as np
import pytest

from massivora.analyzer import (BuiltinGaussCalibrator, BuiltinPLMCalibrator, ComplexAnalyzer,
                                DegenerateCouplingsError)
from massivora.db import STATUS
from massivora.utils import saved_columns_to_bitmask

L1, L2 = 7, 5
SAVED = [list(range(L1)), list(range(L2))]      # kept columns per protein, as the store holds them


def _analyzer(score):
    return ComplexAnalyzer(score, saved_columns=SAVED, chainA_length=L1, EC_threshold=0)


def _zeros():
    return np.zeros((L1 + L2, L1 + L2), dtype=np.float16)


def test_all_zero_matrix_raises_a_runtime_error_subclass():
    with pytest.raises(DegenerateCouplingsError):
        _analyzer(_zeros()).NormalizedCouplings(1.0)
    assert issubclass(DegenerateCouplingsError, RuntimeError)


@pytest.mark.parametrize('calibrator', [BuiltinPLMCalibrator(), BuiltinGaussCalibrator()])
def test_calibrate_returns_a_scoreless_result_for_an_all_zero_matrix(calibrator):
    result = calibrator.Calibrate(_analyzer(_zeros()), 1.0)
    assert math.isnan(result['log_lr_raw']) and math.isnan(result['pcontact'])
    assert result['high_precision_hit'] is False and result['confidence_flag'] is False
    assert result['neffoverL'] == pytest.approx(1.0 / (L1 + L2))


@pytest.mark.parametrize('calibrator', [BuiltinPLMCalibrator(), BuiltinGaussCalibrator()])
def test_calibration_keys_are_all_couplings_columns(calibrator):
    """The contract the recovery path depends on: every key is a column.

    `_coupling_result_info` spreads this dict into the row update and indexes
    the task row by each key, so a key with no column raises IndexError and
    aborts `fix_zombie_tasks` uncommitted -- for exactly the degenerate pair
    this guard exists to rescue."""
    columns = {'id', 'name', 'pid1', 'pid2', 'length', 'number', 'effnumber', 'pcontact',
               'neffoverL', 'log_lr_raw', 'confidence_flag', 'high_precision_hit', 'status',
               'job_id', 'claimed_at', 'file_hash', 'score_file_id', 'score_offset', 'score_len'}
    for score, n_eff in ((_zeros(), 1.0), (_ordinary(), 100.0)):
        assert set(calibrator.Calibrate(_analyzer(score), n_eff)) <= columns


def _ordinary(seed=0):
    rng = np.random.default_rng(seed)
    S = rng.normal(size=(L1 + L2, L1 + L2)).astype(np.float16)
    return ((S + S.T) / 2).astype(np.float16)


def test_calibrate_scores_an_ordinary_matrix():
    result = BuiltinPLMCalibrator().Calibrate(_analyzer(_ordinary()), 100.0)
    assert math.isfinite(result['log_lr_raw']) and 0.0 <= result['pcontact'] <= 1.0


def test_the_detector_call_is_guarded_too(caplog):
    """Both models normalise the same matrix, so the RANKER is not the only
    call that can raise; a guard around it alone leaves the DETECTOR open."""
    calibrator = BuiltinPLMCalibrator()
    calls = []
    original = calibrator._model_log_lr

    def ranker_ok_detector_degenerate(analyzer, model, n_eff):
        calls.append(model is calibrator.DETECTOR)
        if model is calibrator.DETECTOR:
            raise DegenerateCouplingsError('Cannot normalize: minimum absolute score is 0')
        return original(analyzer, model, n_eff)

    calibrator._model_log_lr = ranker_ok_detector_degenerate
    with caplog.at_level(logging.WARNING):
        result = calibrator.Calibrate(_analyzer(_ordinary()), 100.0)
    assert any(calls), 'the detector was never called'
    assert math.isnan(result['log_lr_raw']) and result['high_precision_hit'] is False
    assert 'degenerate alignment' in caplog.text


# -- through the real write path --------------------------------------------

def test_collect_score_writes_a_degenerate_pair_as_done_without_a_score(tmp_path, caplog):
    """The pipeline claim: a degenerate pair ends DONE with a NULL score.

    Drives an all-zero score through the GPU plmDCA executor's `collect_score`
    and its own `_db_apply_results` into a real couplings table, so the schema,
    the SET list and the NaN -> NULL conversion are all exercised."""
    from test_msv import _couple_project, _couplings
    from massivora.executors.couple import PLMCouplingExecutorGPU

    cfg = _couple_project(tmp_path, n_pairs=0)
    conn = sqlite3.connect(tmp_path / 'job.db')
    conn.executemany('INSERT INTO alignments (pid, length) VALUES (?, ?)', [('A0', L1), ('B0', L2)])
    conn.execute("INSERT INTO couplings (name, pid1, pid2, status) VALUES ('A0-B0', 'A0', 'B0', ?)",
                 (STATUS['RUNNING'],))
    conn.commit()
    pair_id = conn.execute("SELECT id FROM couplings WHERE name = 'A0-B0'").fetchone()[0]
    conn.close()

    executor = PLMCouplingExecutorGPU(cfg)
    try:
        params = {'pair_id': pair_id, 'name': 'A0-B0', 'B': 2, 'N': L1 + L2, 'Beff': 1.0,
                  'file_hash': None, 'saved_columns': SAVED, 'lengthes': [L1, L2],
                  'bitmask': saved_columns_to_bitmask(SAVED, [L1, L2])}
        with caplog.at_level(logging.WARNING):
            result = executor.collect_score(_zeros(), dict(params))
        assert 'degenerate alignment' in caplog.text
        executor._db_apply_results([result], [])
        executor.db_conn.commit()
    finally:
        executor.close_db_connection()

    row = _couplings(tmp_path)['A0-B0']
    assert row['status'] == STATUS['DONE']
    assert row['log_lr_raw'] is None and row['pcontact'] is None
    assert row['high_precision_hit'] == 0 and row['confidence_flag'] == 0
    assert row['number'] == 2 and row['effnumber'] == pytest.approx(1.0)


def test_recovery_of_a_dead_job_on_a_degenerate_pair(tmp_path, monkeypatch):
    """`fix_zombie_tasks` must survive the pair the guard was written for.

    The recovery path recalibrates from the stored shard and backfills every
    calibration key onto the task row; a degenerate pair exercises the
    score-less branch there."""
    pytest.importorskip('ost', exc_type=ImportError)   # recovery calibrates through the analyzer
    import importlib
    import sys
    importlib.import_module('massivora.massiveilance')
    massiveilance = sys.modules['massivora.massiveilance']
    from test_msv import _couple_project, _couplings
    from massivora.utils import ShardWriter

    cfg = _couple_project(tmp_path, n_pairs=0)
    with ShardWriter(str(tmp_path / 'ecs'), 'couplings') as writer:
        location = writer.append('A0-B0', _zeros(), saved_columns_to_bitmask(SAVED, [L1, L2]))
    conn = sqlite3.connect(tmp_path / 'job.db')
    conn.executemany('INSERT INTO alignments (pid, length) VALUES (?, ?)', [('A0', L1), ('B0', L2)])
    conn.execute(
        'INSERT INTO couplings (name, pid1, pid2, status, job_id, claimed_at, effnumber) '
        "VALUES ('A0-B0', 'A0', 'B0', ?, '999999', 1758000000.0, 1.0)", (STATUS['RUNNING'],))
    conn.commit()
    conn.close()
    monkeypatch.setattr(massiveilance, '_slurm_queuing_job_ids', lambda: [])

    assert massiveilance.fix_zombie_tasks(cfg, 'couple') == 1

    row = _couplings(tmp_path)['A0-B0']
    assert row['status'] == STATUS['DONE']
    assert (row['score_file_id'], row['score_offset'], row['score_len']) == location
    assert row['log_lr_raw'] is None and row['pcontact'] is None
    assert (row['job_id'], row['claimed_at']) == (None, None)
