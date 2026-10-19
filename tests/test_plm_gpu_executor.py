"""The plmDCA GPU executor on stand-in Sessions.

The dispatch loop it shares with GaussDCA -- admission, host processes, a lost
device, signals -- is covered in test_msv.py; these cover what plmDCA adds:
its own loading and analysis, fitting a pair's site batch to the device
budget, and reweighting on the GPU before the gap columns are dropped.
"""
import sys
import types

import numpy as np
import pytest

from massivora.alignment import BinaryAlignment
from massivora.db import STATUS, get_hashed_file_path
from test_msv import (_PAIR_MASK, _check_stored, _couple_project, _couplings,  # noqa: F401
                      _symmetric, couple)


def _fake_plm_cuda(couple, monkeypatch, free_mb=1024,
                   pair_bytes=lambda B, N, q, site_batch, device=0, l2_share=1.0: 1 << 20):
    """Stand-in for the plm_cuda module: `score` returns a seeded symmetric matrix."""

    class DeviceLostError(RuntimeError):
        pass

    class Session:
        last_pair_bytes = 0

        def __init__(self, device=0, l2_share=1.0):
            self.device, self.l2_share = device, l2_share
            Session.made.append(self)

        def score(self, MSA, W, q, lambdaH, lambdaJ, maxeval, site_batch):
            Session.calls.append({'B': MSA.shape[0], 'N': MSA.shape[1], 'q': q, 'W': W,
                                  'maxeval': maxeval, 'site_batch': site_batch,
                                  'lambdas': (lambdaH, lambdaJ)})
            return _symmetric(MSA.shape[1], seed=int(MSA[0, 0])).astype(np.float32)

        def neighbour_counts(self, MSA, identical_threshold):
            from massivora import cpp_bindings      # the counts the device computes
            Session.reweighted.append(MSA.shape)
            return np.asarray(cpp_bindings.reweightNeighbourCounts(
                np.ascontiguousarray(MSA), identical_threshold, 1), dtype=np.int32)

    Session.made, Session.calls, Session.reweighted = [], [], []
    fake = types.SimpleNamespace(
        Session=Session, DeviceLostError=DeviceLostError, pair_bytes=pair_bytes,
        device_memory=lambda d: {'free_mb': free_mb, 'total_mb': free_mb})
    monkeypatch.setitem(sys.modules, 'massivora.cpp_bindings.plm_cuda', fake)
    monkeypatch.setattr(couple.cpp_bindings, 'plm_cuda', fake, raising=False)
    monkeypatch.setattr(couple, 'visible_gpu_devices', lambda: ['0'])
    return fake


def _store_alignments(tmp_path, n_pairs, B=50, weights=True, gap_columns=()):
    """Pair alignments in the project's store; matrix[0, 0] = id seeds the fake score."""
    saved, lengthes = _PAIR_MASK
    files, want = {}, {}
    for i in range(n_pairs):
        name = f'A{i}-B{i}'
        matrix = np.random.default_rng(i).integers(1, 21, (B, sum(lengthes)), dtype=np.int8)
        matrix[B // 2:] = matrix[:B // 2]           # duplicates, so the weights are not all 1
        for c in gap_columns:
            matrix[:int(0.8 * B), c] = 0            # mostly gaps: the mask drops it
        matrix[0, 0] = i + 1
        ref = BinaryAlignment()
        ref.matrix = matrix.copy()
        ref.Reweight_Sequence(0.8)
        want[name] = (ref.weights, ref.Beff)
        align = BinaryAlignment()
        align.matrix = matrix
        align.species_list = ['Query'] + [f's{k}' for k in range(1, B)]
        align.species_index_map = list(range(B))
        align.saved_columns, align.lengthes = saved, lengthes
        if weights:
            align.Reweight_Sequence(0.8)
        files[name] = get_hashed_file_path(name, None, str(tmp_path / 'ecs'))[0]
        align.To_Zarr(files[name])
    return files, want


def test_gpu_plmdca_threads_store_their_scores(tmp_path, couple, monkeypatch):
    """Sessions in this process: one per GPU with all of its L2, whatever
    gpu_sessions -- GaussDCA's setting -- says."""
    cfg = _couple_project(tmp_path, n_pairs=4, gpu_sessions=4, gpu_workers=3,
                          gpu_host_processes=0, lambda_h=0.02, lambda_J=0.03, maxit=77)
    _, want = _store_alignments(tmp_path, 4)
    fake = _fake_plm_cuda(couple, monkeypatch)
    executor = couple.PLMCouplingExecutorGPU(cfg)

    executor.run()

    rows = _couplings(tmp_path)
    assert all(row['status'] == STATUS['DONE'] for row in rows.values())
    assert all(rows[name]['effnumber'] == beff for name, (_, beff) in want.items())
    _check_stored(tmp_path, rows)
    assert [s.l2_share for s in fake.Session.made] == [1.0]
    for call in fake.Session.calls:
        assert call['site_batch'] == 24 and call['maxeval'] == 77
        assert call['lambdas'] == pytest.approx((0.02, 0.03))
        assert call['W'].sum() == pytest.approx(1.0)       # the weights divided by Beff
    assert fake.Session.reweighted == []                    # the store had them


def test_gpu_plmdca_halves_the_site_batch_to_fit_the_device(tmp_path, couple, monkeypatch):
    cfg = _couple_project(tmp_path, n_pairs=3, gpu_workers=2,
                          gpu_host_processes=0, gpu_memory_limit=1.0)
    _store_alignments(tmp_path, 3)
    big = {'id': None}

    def pair_bytes(B, N, q, site_batch, device=0, l2_share=1.0):
        if big['id'] is not None and B == 80:
            return 10 * 2**30                       # does not fit at any batch
        return site_batch * 10 * 2**20

    fake = _fake_plm_cuda(couple, monkeypatch, free_mb=100, pair_bytes=pair_bytes)
    executor = couple.PLMCouplingExecutorGPU(cfg)
    real_load = executor.load_pair_arrays

    def load_pair_arrays(pair, **kwargs):
        MSA, W, params = real_load(pair, **kwargs)
        if pair['name'] == 'A1-B1':                 # an alignment far too deep for the GPU
            big['id'] = pair['id']
            MSA = np.repeat(MSA, 2, axis=0)[:80]
            W = np.full(80, 1.0 / 80)
            params['B'] = 80
        return MSA, W, params

    monkeypatch.setattr(executor, 'load_pair_arrays', load_pair_arrays)

    executor.run()

    rows = _couplings(tmp_path)
    assert rows['A0-B0']['status'] == rows['A2-B2']['status'] == STATUS['DONE']
    assert rows['A1-B1']['status'] == STATUS['FAILED']
    assert [c['site_batch'] for c in fake.Session.calls] == [6, 6]   # 24 -> 12 -> 6 fits 100 MiB
    assert all(c['B'] != 80 for c in fake.Session.calls)            # never reached a session


def test_gpu_plmdca_reweights_on_the_gpu_before_dropping_gap_columns(tmp_path, couple,
                                                                    monkeypatch):
    """A store without weights is reweighted by the Session over every column, as
    Reweight_Sequence would; only then are the gap columns dropped, and the
    store ends up with the weights the CPU path would leave."""
    cfg = _couple_project(tmp_path, n_pairs=3, gpu_workers=3,
                          gpu_host_processes=0, fast_approximation=True)
    files, want = _store_alignments(tmp_path, 3, B=60, weights=False, gap_columns=(1, 6))
    fake = _fake_plm_cuda(couple, monkeypatch)
    executor = couple.PLMCouplingExecutorGPU(cfg)

    executor.run()

    rows = _couplings(tmp_path)
    full = sum(_PAIR_MASK[1])
    assert fake.Session.reweighted == [(60, full)] * 3              # every column
    assert [c['N'] for c in fake.Session.calls] == [full - 2] * 3   # without the gap columns
    for name, (weights, beff) in want.items():
        assert rows[name]['status'] == STATUS['DONE']
        assert rows[name]['effnumber'] == beff
        assert rows[name]['length'] == full                         # expanded back
        stored = BinaryAlignment(files[name])
        assert stored.weights == weights and stored.Beff == beff
