"""The batched multi-site GPU L-BFGS: plm_opt_site and the plm_cuda Session.

Both paths of `plm_opt_site --use-gpu` run the same L-BFGS; the batched one
(the default, `site_batch` > 0) only stacks many sites into each GEMM. They
must agree to within the per-site path's own run-to-run noise (float atomics),
and both must match the stored reference scores. The Session runs the batched
path in-process and computes the score on the device as well.
"""
import os
import re
import subprocess
import threading
from multiprocessing import shared_memory

import numpy as np
import pytest

from massivora import BinaryAlignment, cpp_bindings
from massivora.analyzer import ComplexAnalyzer
from massivora.executors.couple import BaseCouplingExecutor
from massivora.utils import massivora_pkg_dir

TEST_ZARR = os.path.join(os.path.dirname(__file__), "RL4_ECOLI-RL31_ECOLI")
STANDARD_SCORE = os.path.join(os.path.dirname(__file__), "score.npy")
PAIR = "plm_batched_test"
PLM_OPT_EXE = os.environ.get(
    "MASSIVORA_PLM_OPT_EXE", os.path.join(massivora_pkg_dir(), "bin", "plm_opt_site"))


def _write_plm_shm(name, MSA, W):
    w_offset = (MSA.nbytes + 7) & ~7
    shm = shared_memory.SharedMemory(name=name, create=True, size=w_offset + W.nbytes)
    np.copyto(np.ndarray(MSA.shape, dtype=np.int8, buffer=shm.buf), MSA)
    np.copyto(np.ndarray(W.shape, dtype=np.float64, buffer=shm.buf[w_offset:]), W)
    return shm


@pytest.fixture(scope="module")
def runs():
    try:
        import cupy  # noqa: F401  -- GPU-availability gate
    except ImportError:
        pytest.skip("CuPy / GPU not available")

    align = BinaryAlignment(TEST_ZARR)
    MSA = np.ascontiguousarray(align.matrix, dtype=np.int8)
    B, N = MSA.shape
    W = np.array(align.weights, dtype=np.float64) / float(align.Beff)
    q = int(MSA.max()) + 1

    msa_shm = _write_plm_shm("Massivora_" + PAIR, MSA, W)
    J_shm = shared_memory.SharedMemory(name="Massivora_" + PAIR + "_J", create=True,
                                       size=N * N * q * q * 4)
    J = np.ndarray((N, N * q * q), dtype=np.float32, buffer=J_shm.buf)
    out = {}
    try:
        for label, site_batch in (("streams", 0), ("batched", 16)):
            J[:] = np.nan
            rv = subprocess.run(
                [PLM_OPT_EXE, "--use-gpu", PAIR, str(B), str(N), str(q), "0.01", "0.01",
                 "3e-4", "500", "4", str(site_batch)],
                capture_output=True, text=True,
                env=dict(os.environ, MASSIVORA_PLM_STATS="1"))
            assert rv.returncode == 0, f"{label}: rc={rv.returncode}\n{rv.stderr}"
            assert not np.isnan(J).any(), f"{label}: J not fully written"
            stats = dict(re.findall(r"(\w+)=(\S+)", rv.stderr.split("PLM_STATS", 1)[1]))
            Jg = J.copy()
            cpp_bindings.applyIsingGauge(Jg, q)
            score = BaseCouplingExecutor.compute_score(Jg.T, q=q).astype(np.float16)
            out[label] = (score, stats)
    finally:
        del J
        J_shm.close()
        J_shm.unlink()
        msa_shm.close()
        msa_shm.unlink()
    return out


def test_batched_matches_reference(runs):
    standard = ComplexAnalyzer(STANDARD_SCORE, chainA_length=201)
    batched = ComplexAnalyzer(runs["batched"][0], chainA_length=201)
    spearman, _ = standard.SpearmanCorrelation(batched)
    assert spearman > 0.99, f"Spearman vs reference too low: {spearman}"


def test_batched_matches_per_site(runs):
    (s_ref, st_ref), (s_new, st_new) = runs["streams"], runs["batched"]
    iu = np.triu_indices(s_ref.shape[0], 1)
    rho = np.corrcoef(np.argsort(np.argsort(s_ref[iu])), np.argsort(np.argsort(s_new[iu])))[0, 1]
    assert rho > 0.999, f"batched vs per-site Spearman too low: {rho}"
    # Same algorithm: same objective reached, same amount of work. Run-to-run
    # spread of f_sum on this alignment is ~5e-5 relative on either path.
    f_ref, f_new = float(st_ref["f_sum"]), float(st_new["f_sum"])
    assert abs(f_new - f_ref) <= 3e-4 * abs(f_ref), (f_ref, f_new)
    it_ref, it_new = int(st_ref["iters"]), int(st_new["iters"])
    assert abs(it_new - it_ref) <= 0.05 * it_ref, (it_ref, it_new)


# ---------------------------------------------------------------------------
# plm_cuda.Session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def plm_cuda():
    try:
        from massivora.cpp_bindings import plm_cuda
    except ImportError:
        pytest.skip("massivora was built without CUDA (no plm_cuda)")
    if plm_cuda.device_count() == 0:
        pytest.skip("no CUDA device")
    return plm_cuda


@pytest.fixture(scope="module")
def rl4():
    align = BinaryAlignment(TEST_ZARR)
    MSA = np.ascontiguousarray(align.matrix, dtype=np.int8)
    W = np.array(align.weights, dtype=np.float64) / float(align.Beff)
    return MSA, W, int(MSA.max()) + 1


def _spearman(a, b):
    iu = np.triu_indices(a.shape[0], 1)
    return np.corrcoef(np.argsort(np.argsort(a[iu])), np.argsort(np.argsort(b[iu])))[0, 1]


def test_session_scores_as_compute_score_does_on_the_gauged_couplings(plm_cuda, rl4):
    """The device score path, on the very couplings, against the CPU one."""
    MSA, W, q = rl4
    sess = plm_cuda.Session(0)
    J = sess.couplings(MSA, W, q, maxeval=50)
    Jg = J.copy()
    cpp_bindings.applyIsingGauge(Jg, q)
    want = BaseCouplingExecutor.compute_score(Jg.T, q=q)
    got = sess.score_couplings(J, q)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-5 * np.abs(want).max())


def test_session_score_matches_reference_and_executable(plm_cuda, rl4, runs):
    MSA, W, q = rl4
    sess = plm_cuda.Session(0)
    score = sess.score(MSA, W, q, maxeval=500)
    assert sess.last_stats["slots"] >= 8 and sess.last_stats["converged"] > 0
    standard = ComplexAnalyzer(STANDARD_SCORE, chainA_length=201)
    spearman, _ = standard.SpearmanCorrelation(
        ComplexAnalyzer(score.astype(np.float16), chainA_length=201))
    assert spearman > 0.99, f"Spearman vs reference too low: {spearman}"
    assert _spearman(score, runs["batched"][0].astype(np.float32)) > 0.999


def test_session_neighbour_counts_match_the_cpu(plm_cuda, rl4):
    MSA = rl4[0]
    threshold = 0.8 * MSA.shape[1]
    got = plm_cuda.Session(0).neighbour_counts(MSA, threshold)
    np.testing.assert_array_equal(got, cpp_bindings.reweightNeighbourCounts(MSA, threshold, 0))


def test_sessions_share_a_gpu_and_hand_its_memory_back(plm_cuda, rl4):
    """Two Sessions on one GPU from two threads, each within what admission
    reserves for its pair, and nothing left held on the device once they are
    done: idle Sessions must not sit on memory another process could use."""
    MSA, W, _ = rl4
    sub = np.ascontiguousarray(MSA[:, :80])
    q = int(sub.max()) + 1
    sessions = [plm_cuda.Session(0, l2_share=0.5) for _ in range(2)]
    free_before = plm_cuda.device_memory(0)["free_mb"]
    scores = [None, None]

    def run(k):
        scores[k] = sessions[k].score(sub, W, q, maxeval=100)

    threads = [threading.Thread(target=run, args=(k,)) for k in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert _spearman(scores[0], scores[1]) > 0.999
    reserved = plm_cuda.pair_bytes(sub.shape[0], sub.shape[1], q, 24, 0, 0.5)
    assert all(0 < s.last_pair_bytes <= reserved for s in sessions)
    assert plm_cuda.device_memory(0)["free_mb"] >= free_before - 256


def test_a_state_outside_the_alphabet_is_refused_and_the_device_survives(plm_cuda, rl4):
    """On the device it would be an illegal address, which takes the CUDA
    context -- every Session in the process -- down with it."""
    MSA, W, q = rl4
    sess = plm_cuda.Session(0)
    bad = np.ascontiguousarray(MSA[:, :20])
    bad[3, 4] = q + 5
    with pytest.raises(ValueError):
        sess.score(bad, W, q)
    good = np.ascontiguousarray(MSA[:, :20])
    assert np.isfinite(sess.score(good, W, int(good.max()) + 1, maxeval=20)).all()
