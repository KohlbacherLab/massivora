"""The GaussDCA stage's depth pre-gate (couple.cascade.min_neffoverL): a pair
whose Neff/L after reweighting is below the gate is never handed to plmDCA
by the cascade, so the fused GPU stage leaves it unscored and records it as
SKIPPED with its depth. Checked without a GPU, on a fake session."""
import types

import numpy as np
import pytest

from massivora.db import STATUS
from massivora.executors.couple import GaussCouplingExecutorGPU, SessionGPUExecutor


class _Session:
    """neighbour_counts gives every row `count` neighbours; score records its calls."""

    def __init__(self, count):
        self.count, self.scored = count, []
        self.last_pair_bytes = 0

    def neighbour_counts(self, MSA, threshold):
        return np.full(MSA.shape[1], self.count, dtype=np.int32)

    def score(self, MSA, W, q, pseudocount):
        self.scored.append((MSA.shape, float(W.sum()), q, pseudocount))
        return np.zeros((MSA.shape[0], MSA.shape[0]), dtype=np.float32)


def _executor(gate):
    return types.SimpleNamespace(min_neffoverL=gate, pseudocount=0.8)


def _params(N, B):
    return {'name': 'A-B', 'pair_id': 7, 'B': B, 'N': N, 'q': 21, 'Beff': None,
            'file_hash': b'h'}


def test_depth_gate_is_read_from_the_cascade_block():
    gate = GaussCouplingExecutorGPU._depth_gate
    assert gate({}) is None
    assert gate({'cascade': None}) is None
    assert gate({'cascade': {'keep_fraction': 0.2}}) is None
    assert gate({'cascade': {'min_neffoverL': None}}) is None
    assert gate({'cascade': {'min_neffoverL': 'null'}}) is None
    assert gate({'cascade': {'min_neffoverL': 0.1}}) == pytest.approx(0.1)
    assert gate({'cascade': {'min_neffoverL': '0.05'}}) == pytest.approx(0.05)
    with pytest.raises(ValueError, match='min_neffoverL'):
        gate({'cascade': {'min_neffoverL': -1}})


@pytest.mark.parametrize('gate, count, scored', [
    (None, 50, True),        # no gate: every pair is scored
    (0.1, 1, True),          # Beff 100 of B 100: Neff/L = 100/200 = 0.5 >= 0.1
    (0.1, 10, False),        # Beff 10: 10/200 = 0.05 < 0.1: skipped
    (0.1, 50, False),        # Beff 2: 0.01: skipped
    (0.0, 1000, True),       # a zero gate keeps everything
])
def test_pair_below_the_gate_is_not_scored(gate, count, scored):
    N, B = 200, 100
    MSA = np.zeros((N, B), dtype=np.int8)
    sess = _Session(count)
    params = _params(N, B)
    kwargs = {'reweighting_threshold': 0.8}
    score = GaussCouplingExecutorGPU._run_on_session(_executor(gate), sess, MSA, None, params, kwargs)
    assert params['Beff'] == pytest.approx(B / count)
    if scored:
        assert score is not None and len(sess.scored) == 1
        assert 'skipped' not in params and params['weights'].shape == (B,)
    else:
        assert score is None and sess.scored == []
        assert 'min_neffoverL' in params['skipped'] and 'weights' not in params


def test_stored_weights_are_gated_too():
    """A pair with weights from the store is gated on their Beff."""
    N, B = 200, 100
    MSA = np.zeros((N, B), dtype=np.int8)
    W = np.full(B, 0.1)                                    # Beff 10: Neff/L 0.05
    params = {**_params(N, B), 'Beff': 10.0}
    sess = _Session(1)
    assert GaussCouplingExecutorGPU._run_on_session(_executor(0.1), sess, MSA, W, params,
                                                     {'reweighting_threshold': 0.8}) is None
    assert sess.scored == []
    assert GaussCouplingExecutorGPU._run_on_session(_executor(0.04), sess, MSA, W, params,
                                                     {'reweighting_threshold': 0.8}) is not None


def test_skipped_row_carries_the_depth_and_no_score():
    params = {**_params(300, 120), 'Beff': 12.0, 'skipped': 'Neff/L 0.040 below ...'}
    row = SessionGPUExecutor._skipped_row(types.SimpleNamespace(), params)
    assert row['id'] == 7 and row['status'] == STATUS['SKIPPED']
    assert row['number'] == 120 and row['length'] == 300 and row['effnumber'] == 12.0
    assert row['neffoverL'] == pytest.approx(0.04)
    assert row['log_lr_raw'] is None and row['score_file_id'] is None
    assert row['high_precision_hit'] is False and row['confidence_flag'] is False
    assert row['file_hash'] == b'h'
    # every column _db_apply_results writes is there
    for key in ('status', 'effnumber', 'number', 'length', 'pcontact', 'neffoverL', 'log_lr_raw',
                'confidence_flag', 'high_precision_hit', 'file_hash', 'score_file_id',
                'score_offset', 'score_len', 'id'):
        assert key in row
