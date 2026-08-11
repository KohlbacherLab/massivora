"""Validate the GaussDCA C++ port against the reference Julia package.

The reference artifacts in ``tests/gauss_ref/`` were produced by running the
original Julia GaussDCA pipeline (temp_code/dump_julia_gdca.jl) on the small
alignment ``mini.a2m`` (200 sequences x 30 columns):

    mJ.npy       inv(cholesky(C))                     -- native core output
    S_fn.npy     Frobenius-norm score (pre-APC)
    S_apc.npy    APC-corrected score (final matrix)   -- Python compute_score output
    ranking.npy  sorted (i, j, score)                 -- final coupling ranking
    Z.npy, W.npy, meta.npy                             -- inputs / metadata

Three layers are checked:
  * compute_score (Python FN + APC) vs Julia            -- always runs
  * cpp_bindings.gaussCouplings (native core) vs Julia  -- needs the native module
  * full native+Python chain vs Julia                   -- needs the native module

All comparisons are to (near) machine precision.
"""
import os
import subprocess
import sysconfig

import numpy as np
import pytest

from massivora.executors.couple import GaussCouplingExecutor
from massivora.utils import massivora_pkg_dir

REF_DIR = os.path.join(os.path.dirname(__file__), "gauss_ref")


def _load(name):
    return np.load(os.path.join(REF_DIR, name))


try:
    from massivora import cpp_bindings
    HAS_BINDING = hasattr(cpp_bindings, "gaussCouplings")
except Exception:  # pragma: no cover - import guard
    cpp_bindings = None
    HAS_BINDING = False

needs_binding = pytest.mark.skipif(
    not HAS_BINDING,
    reason="cpp_bindings.gaussCouplings not built (rebuild the native extension)",
)


def _find_gauss_infer():
    """Locate the gauss_infer executable (the executor uses massivora_pkg_dir())."""
    candidates = [
        os.path.join(massivora_pkg_dir(), "bin", "gauss_infer"),
        os.path.join(sysconfig.get_paths()["platlib"], "massivora", "bin", "gauss_infer"),
    ]
    return next((c for c in candidates if os.path.exists(c)), None)


GAUSS_INFER = _find_gauss_infer()

needs_gauss_infer = pytest.mark.skipif(
    GAUSS_INFER is None,
    reason="gauss_infer executable not built (rebuild the native extension)",
)


def _ranking(S, min_separation=5):
    """Reproduce GaussDCA.compute_ranking as (i, j, S[j,i]), 1-based, desc."""
    N = S.shape[0]
    out = []
    for i in range(N - min_separation):
        for j in range(i + min_separation, N):
            out.append((i + 1, j + 1, float(S[j, i])))
    out.sort(key=lambda x: x[2], reverse=True)
    return out


def test_compute_score_matches_julia():
    """Python FN + APC (compute_score) reproduces Julia's final score matrix."""
    mJ = _load("mJ.npy")
    S_apc_ref = _load("S_apc.npy")

    score = GaussCouplingExecutor.compute_score(mJ, q=21)

    assert score.shape == S_apc_ref.shape
    assert np.allclose(score, S_apc_ref, atol=1e-10, rtol=0), (
        "compute_score max abs diff vs Julia = %.3e"
        % np.max(np.abs(score - S_apc_ref))
    )
    # Score matrix is symmetric (as in GaussDCA after APC).
    assert np.allclose(score, score.T, atol=1e-10)


def test_ranking_matches_julia():
    """The coupling ranking derived from compute_score matches Julia exactly."""
    mJ = _load("mJ.npy")
    ranking_ref = _load("ranking.npy")

    score = GaussCouplingExecutor.compute_score(mJ, q=21)
    ranking = _ranking(score, min_separation=5)

    assert len(ranking) == len(ranking_ref)
    # Every ranked pair appears in the same position with the same score.
    for k, (i, j, sc) in enumerate(ranking):
        assert (i, j) == (int(ranking_ref[k, 0]), int(ranking_ref[k, 1]))
        assert abs(sc - ranking_ref[k, 2]) < 1e-10


@needs_binding
def test_gauss_couplings_core_matches_julia():
    """Native gaussCouplings reproduces Julia's mJ = inv(cholesky(C))."""
    Z = _load("Z.npy")          # (N, M), GaussDCA encoding (gap = 21)
    W = _load("W.npy")          # (M,)
    mJ_ref = _load("mJ.npy")

    MSA = np.ascontiguousarray(Z.T)   # (M, N) = (B, N)
    mJ = cpp_bindings.gaussCouplings(MSA, W, 21, 0.8)

    assert mJ.shape == mJ_ref.shape
    assert np.allclose(mJ, mJ_ref, atol=1e-8, rtol=0), (
        "gaussCouplings mJ max abs diff vs Julia = %.3e"
        % np.max(np.abs(mJ - mJ_ref))
    )


@needs_binding
def test_encoding_agnostic():
    """massivora encoding (gap = 0) gives the same couplings as GaussDCA (gap = 21)."""
    Z = _load("Z.npy")
    W = _load("W.npy")
    MSA_g21 = np.ascontiguousarray(Z.T)
    MSA_g0 = MSA_g21.copy()
    MSA_g0[MSA_g0 == 21] = 0          # remap gap 21 -> 0 (massivora convention)

    mJ_g21 = cpp_bindings.gaussCouplings(MSA_g21, W, 21, 0.8)
    mJ_g0 = cpp_bindings.gaussCouplings(MSA_g0.astype(np.int8), W, 21, 0.8)
    assert np.array_equal(mJ_g21, mJ_g0)


@needs_binding
def test_full_native_python_chain_matches_julia():
    """End-to-end: native core -> Python compute_score vs Julia final output."""
    Z = _load("Z.npy")
    W = _load("W.npy")
    S_apc_ref = _load("S_apc.npy")
    ranking_ref = _load("ranking.npy")

    MSA = np.ascontiguousarray(Z.T)
    mJ = cpp_bindings.gaussCouplings(MSA, W, 21, 0.8)
    score = GaussCouplingExecutor.compute_score(mJ, q=21)

    assert np.allclose(score, S_apc_ref, atol=1e-8, rtol=0)
    ranking = _ranking(score, min_separation=5)
    top = 25
    for k in range(min(top, len(ranking))):
        assert (ranking[k][0], ranking[k][1]) == (
            int(ranking_ref[k, 0]), int(ranking_ref[k, 1]))


@needs_gauss_infer
def test_gauss_infer_shared_memory_matches_julia():
    """Exercise the executor's data path: publish MSA+W to shared memory, run the
    gauss_infer executable, read J back, and score it — all as GaussCouplingExecutor
    (load_pair -> infer -> collect_results) does, and compare to Julia."""
    from multiprocessing import shared_memory

    Z = _load("Z.npy")                    # (N, M), GaussDCA encoding (gap = 21)
    W = _load("W.npy").astype(np.float64)  # raw weights (Meff = sum(W))
    S_apc_ref = _load("S_apc.npy")
    ranking_ref = _load("ranking.npy")
    q = 21
    N, M = Z.shape
    NQ = N * (q - 1)

    # massivora BinaryAlignment encoding: gap = 0 (what load_pair feeds).
    MSA = np.ascontiguousarray(Z.T.astype(np.int32))
    MSA[MSA == 21] = 0

    prefix = "Massivora_"
    pair = "pyg"
    msa_bytes = MSA.nbytes
    w_bytes = W.nbytes

    in_shm = shared_memory.SharedMemory(name=prefix + pair, create=True,
                                        size=msa_bytes + w_bytes)
    j_shm = shared_memory.SharedMemory(name=prefix + pair + "_J", create=True,
                                       size=NQ * NQ * 8)
    try:
        # load_pair: MSA (M, N) int32 then W (M,) float64, contiguously.
        np.ndarray(MSA.shape, dtype=np.int32, buffer=in_shm.buf)[:] = MSA
        np.ndarray(W.shape, dtype=np.float64, buffer=in_shm.buf[msa_bytes:])[:] = W

        # infer: gauss_infer <pair> <M> <N> <q> <pseudocount> <n_threads>
        rv = subprocess.run(
            [GAUSS_INFER, pair, str(M), str(N), str(q), "0.8", "2"])
        assert rv.returncode == 0, f"gauss_infer failed rc={rv.returncode}"

        # collect_results: read J (N*Q, N*Q) float64 row-major and score it.
        J = np.ndarray((NQ, NQ), dtype=np.float64, buffer=j_shm.buf).copy()
        score = GaussCouplingExecutor.compute_score(J, q=q)

        assert np.allclose(score, S_apc_ref, atol=1e-8, rtol=0), (
            "score max abs diff vs Julia = %.3e" % np.max(np.abs(score - S_apc_ref)))
        ranking = _ranking(score, min_separation=5)
        for k in range(min(25, len(ranking))):
            assert (ranking[k][0], ranking[k][1]) == (
                int(ranking_ref[k, 0]), int(ranking_ref[k, 1]))
    finally:
        in_shm.close()
        in_shm.unlink()
        j_shm.close()
        j_shm.unlink()
