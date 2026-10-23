"""A degenerate alignment (B <= 3 on the screen: one or two distinct sequences)
gives an all-zero inter-protein block, which cannot be normalised. The
calibrator must then return a defined, score-less result instead of raising
through `collect_score` (the pair ends DONE without a score, not FAILED)."""
import math
import sqlite3

import numpy as np
import pytest

from massivora.analyzer import (BuiltinGaussCalibrator, BuiltinPLMCalibrator, ComplexAnalyzer,
                                DegenerateCouplingsError)

L1, L2 = 7, 5
SAVED = [list(range(L1)), list(range(L2))]      # kept columns per protein, as the store holds them


def _analyzer(score):
    return ComplexAnalyzer(score, saved_columns=SAVED, chainA_length=L1, EC_threshold=0)


def test_all_zero_matrix_raises_a_runtime_error_subclass():
    with pytest.raises(DegenerateCouplingsError):
        _analyzer(np.zeros((L1 + L2, L1 + L2), dtype=np.float16)).NormalizedCouplings(1.0)
    assert issubclass(DegenerateCouplingsError, RuntimeError)


@pytest.mark.parametrize('calibrator', [BuiltinPLMCalibrator(), BuiltinGaussCalibrator()])
def test_calibrate_returns_a_scoreless_result_for_an_all_zero_matrix(calibrator):
    result = calibrator.Calibrate(_analyzer(np.zeros((L1 + L2, L1 + L2), dtype=np.float16)), 1.0)
    assert result['reason'] == 'degenerate alignment'
    assert math.isnan(result['log_lr_raw']) and math.isnan(result['pcontact'])
    assert result['high_precision_hit'] is False and result['confidence_flag'] is False
    assert result['neffoverL'] == pytest.approx(1.0 / (L1 + L2))


def test_calibrate_scores_an_ordinary_matrix():
    rng = np.random.default_rng(0)
    S = rng.normal(size=(L1 + L2, L1 + L2)).astype(np.float16)
    S = (S + S.T) / 2
    result = BuiltinPLMCalibrator().Calibrate(_analyzer(S), 100.0)
    assert result['reason'] is None
    assert math.isfinite(result['log_lr_raw']) and 0.0 <= result['pcontact'] <= 1.0


def test_nan_score_is_stored_as_null():
    # what makes the pair DONE-with-no-score in the couplings table
    conn = sqlite3.connect(':memory:')
    conn.execute("CREATE TABLE t (log_lr_raw REAL, pcontact REAL)")
    conn.execute("INSERT INTO t VALUES (?, ?)", (float('nan'), float('nan')))
    assert conn.execute("SELECT log_lr_raw IS NULL, pcontact IS NULL FROM t").fetchone() == (1, 1)
