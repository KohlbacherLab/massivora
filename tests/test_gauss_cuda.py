"""Validate the CUDA GaussDCA core against the Julia reference and the CPU path.

The GPU is reached only through ``massivora.gauss_cuda``; there is no CUDA
executable. The reference artifacts in ``tests/gauss_ref/`` come from the
original Julia GaussDCA pipeline -- see test_gauss_coupling.py for how they were
produced.

The whole module skips unless a CUDA build and a device are both present, so a
CPU-only checkout is unaffected.
"""
import os
import subprocess
import sys
import sysconfig
import threading
from multiprocessing import shared_memory

import numpy as np
import pytest

from massivora.utils import massivora_pkg_dir

REF_DIR = os.path.join(os.path.dirname(__file__), "gauss_ref")
PREFIX = "Massivora_"


def _load(name):
    return np.load(os.path.join(REF_DIR, name))


def _import_gauss_cuda():
    """The installed module, or a build tree via MASSIVORA_GAUSS_CUDA_PATH."""
    extra = os.environ.get("MASSIVORA_GAUSS_CUDA_PATH")
    if extra and extra not in sys.path:
        sys.path.insert(0, extra)
    try:
        import gauss_cuda  # noqa: PLC0415  (build tree)
        return gauss_cuda
    except ImportError:
        pass
    try:
        from massivora.cpp_bindings import gauss_cuda  # noqa: PLC0415
        return gauss_cuda
    except ImportError:
        return None


GC = _import_gauss_cuda()


def _has_device():
    try:
        return GC is not None and GC.device_count() > 0
    except Exception:  # pragma: no cover - probe guard
        return False


needs_cuda = pytest.mark.skipif(
    not _has_device(),
    reason="massivora.gauss_cuda not built, or no CUDA device visible",
)


def _find_gauss_infer():
    """The CPU executable, which is what the GPU result is cross-checked against."""
    env = os.environ.get("MASSIVORA_GAUSS_INFER")
    candidates = [env] if env else []
    candidates += [
        os.path.join(massivora_pkg_dir(), "bin", "gauss_infer"),
        os.path.join(sysconfig.get_paths()["platlib"], "massivora", "bin",
                     "gauss_infer"),
    ]
    return next((c for c in candidates if c and os.path.exists(c)), None)


GAUSS_INFER = _find_gauss_infer()
needs_gauss_infer = pytest.mark.skipif(
    GAUSS_INFER is None, reason="CPU gauss_infer not built")


def _cpu_score(pair, Z, W, q=21, pseudocount=0.8, threads=4):
    """Run the CPU executable over its shared-memory contract."""
    N, M = Z.shape
    msa = np.ascontiguousarray(Z, dtype=np.int8)
    w = np.ascontiguousarray(W, dtype=np.float64)
    for nm in (PREFIX + pair, PREFIX + pair + "_J"):
        try:
            old = shared_memory.SharedMemory(name=nm); old.close(); old.unlink()
        except FileNotFoundError:
            pass
    inp = shared_memory.SharedMemory(name=PREFIX + pair, create=True,
                                     size=msa.nbytes + w.nbytes)
    out = shared_memory.SharedMemory(name=PREFIX + pair + "_J", create=True,
                                     size=N * N * 4)
    try:
        np.ndarray(msa.shape, dtype=np.int8, buffer=inp.buf)[:] = msa
        np.ndarray(w.shape, dtype=np.float64, buffer=inp.buf[msa.nbytes:])[:] = w
        rv = subprocess.run([GAUSS_INFER, pair, str(M), str(N), str(q),
                             str(pseudocount), str(threads)], capture_output=True)
        assert rv.returncode == 0, rv.stderr.decode()
        return np.ndarray((N, N), dtype=np.float32, buffer=out.buf).copy()
    finally:
        inp.close(); inp.unlink()
        out.close(); out.unlink()


def _ranking(S, min_separation=5, pairs_only=False):
    """Reproduce GaussDCA.compute_ranking as (i, j, S[j,i]), 1-based, desc."""
    N = S.shape[0]
    out = [(i + 1, j + 1, float(S[j, i]))
           for i in range(N - min_separation)
           for j in range(i + min_separation, N)]
    out.sort(key=lambda x: x[2], reverse=True)
    return [(i, j) for i, j, _ in out] if pairs_only else out


@pytest.fixture(scope="module")
def reference():
    Z = _load("Z.npy")                  # (N, M), GaussDCA encoding (gap = 21)
    Z0 = Z.copy()
    Z0[Z0 == 21] = 0                    # massivora encoding (gap = 0)
    return {"Z": Z, "Z0": Z0, "W": _load("W.npy").astype(np.float64),
            "S_apc": _load("S_apc.npy"), "mJ": _load("mJ.npy"),
            "ranking": _load("ranking.npy")}


@pytest.fixture(scope="module")
def session():
    return GC.Session(0)


@needs_cuda
def test_device_visible():
    """The module reports the devices CUDA can see, and nothing else."""
    assert GC.device_count() >= 1
    info = GC.device_info(0)
    assert info["name"] and info["sm"] >= 60
    assert GC.device_memory(0)["total_mb"] > 0


@needs_cuda
def test_score_matches_julia(reference, session):
    """The float32 score reproduces Julia's APC-corrected matrix."""
    score = session.score(reference["Z0"], reference["W"])
    ref = reference["S_apc"]
    assert score.shape == ref.shape
    assert np.allclose(score, ref, atol=1e-2, rtol=0), (
        "max abs diff vs Julia = %.3e" % np.max(np.abs(score - ref)))
    assert np.allclose(score, score.T, atol=1e-6)


@needs_cuda
def test_ranking_matches_julia(reference, session):
    """The top contacts come out in Julia's order."""
    score = session.score(reference["Z0"], reference["W"])
    ref = reference["ranking"]
    ranking = _ranking(score, min_separation=5)
    for k in range(min(15, len(ranking))):
        assert (ranking[k][0], ranking[k][1]) == (int(ref[k, 0]), int(ref[k, 1]))


@needs_cuda
def test_couplings_float64_matches_julia(reference, session):
    """The float64 path reproduces Julia's inverse to near machine precision.

    This is the reason couplings() factorises in float64 and accumulates the
    pair histograms in double regardless of what the score path does.
    """
    mJ = session.couplings(reference["Z0"], reference["W"])
    assert mJ.shape == reference["mJ"].shape
    assert np.allclose(mJ, reference["mJ"], atol=1e-8, rtol=0), (
        "max abs diff vs Julia = %.3e" % np.max(np.abs(mJ - reference["mJ"])))


@needs_cuda
def test_encoding_agnostic(reference, session):
    """gap = 0 and gap = 21 encodings give bit-identical scores."""
    a = session.score(reference["Z0"], reference["W"])
    b = session.score(reference["Z"], reference["W"])
    assert np.array_equal(a, b)


@needs_cuda
def test_repeated_and_varying_sizes(reference, session):
    """One Session over many pairs must match one Session per pair.

    Regression test for device buffers that survive between calls: FN is only
    written off-diagonal (the CPU core relies on a fresh zero-initialised
    vector), so a reused buffer would leave the previous pair's values on the
    diagonal and the APC row sums would pick them up. It only shows up once N
    varies, which a single-size smoke test never does.
    """
    Z0, W = reference["Z0"], reference["W"]
    sizes = [Z0.shape[0], Z0.shape[0] // 2, Z0.shape[0], Z0.shape[0] // 3]
    expected = [GC.Session(0).score(np.ascontiguousarray(Z0[:n]), W)
                for n in sizes]
    for n, want in zip(sizes, expected):
        got = session.score(np.ascontiguousarray(Z0[:n]), W)
        assert np.array_equal(got, want), f"reused Session differs at N={n}"


@needs_cuda
def test_session_pool_is_thread_safe(reference):
    """A pool of Sessions driven by threads gives the per-Session answers.

    This is how GaussCouplingExecutorGPU runs: score() releases the GIL, so the
    pool has to hand each Session to one thread at a time and each Session has
    to reselect its own device.
    """
    Z0, W = reference["Z0"], reference["W"]
    sizes = [Z0.shape[0], Z0.shape[0] // 2, Z0.shape[0] // 3] * 4
    want = {n: GC.Session(0).score(np.ascontiguousarray(Z0[:n]), W) for n in set(sizes)}

    import queue
    pool = queue.Queue()
    for _ in range(3):
        pool.put(GC.Session(0))
    work = queue.Queue()
    for n in sizes:
        work.put(n)
    bad = []

    def worker():
        while True:
            try:
                n = work.get_nowait()
            except queue.Empty:
                return
            sess = pool.get()
            try:
                got = sess.score(np.ascontiguousarray(Z0[:n]), W)
            finally:
                pool.put(sess)
            if not np.array_equal(got, want[n]):
                bad.append(n)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not bad, f"pooled Sessions disagreed at N={sorted(set(bad))}"


@needs_cuda
@needs_gauss_infer
def test_matches_cpu_executable(reference, session):
    """GPU module and CPU executable agree, and rank identically.

    On a tensor-core device the GPU runs TF32 potrf/trtri and an fp16 lauum,
    which moves the score by a few 1e-3 relative. The bound below is on that;
    the assertion that matters is the ranking, which is what the pipeline
    consumes and which must not move.
    """
    gpu = session.score(reference["Z0"], reference["W"])
    cpu = _cpu_score("cudavcpu", reference["Z0"], reference["W"])
    scale = max(np.max(np.abs(cpu)), 1e-30)
    assert np.max(np.abs(gpu - cpu)) / scale < 1e-2
    assert (_ranking(gpu, 5, pairs_only=True)[:25]
            == _ranking(cpu, 5, pairs_only=True)[:25])


def _family(B, N, seed, q=21):
    """(B, N) sequences drawn from a few ancestors at assorted mutation rates, so
    their pairwise identities spread across the reweighting thresholds."""
    rng = np.random.default_rng(seed)
    parents = rng.integers(0, q, (max(1, B // 20), N), dtype=np.int8)
    A = parents[rng.integers(0, len(parents), B)]
    flip = rng.random((B, N)) < rng.uniform(0.0, 0.4, (B, 1))
    A[flip] = rng.integers(0, q, int(flip.sum()), dtype=np.int8)
    return A


@needs_cuda
@pytest.mark.parametrize("B,N", [(1, 7), (2, 1), (33, 3), (31, 4), (100, 5),
                                 (1000, 131), (2500, 575)])
def test_neighbour_counts_match_the_cpu_core(session, B, N):
    """GPU reweighting counts are the CPU core's, exactly, at any threshold."""
    from massivora import cpp_bindings
    A = _family(B, N, seed=7 * B + N)
    k = round(0.7 * N)
    for t in (0.8 * N, 0.5 * N, 0.0, -1.0, float(N), N + 1.0, k + 1e-9, k - 1e-9):
        want = np.asarray(cpp_bindings.reweightNeighbourCounts(A, t, 0))
        got = session.neighbour_counts(np.ascontiguousarray(A.T), t)
        assert np.array_equal(got, want), f"B={B} N={N} threshold {t!r}"


@needs_cuda
def test_neighbour_counts_fit_in_the_pair_reservation(session):
    """The executor reweights inside the memory it reserved for the pair's score."""
    for B, N in ((60000, 300), (3000, 2600), (5, 2)):
        session.neighbour_counts(np.zeros((N, B), dtype=np.int8), 0.8 * N)
        assert 0 < session.last_pair_bytes <= GC.pair_bytes(N, B)
