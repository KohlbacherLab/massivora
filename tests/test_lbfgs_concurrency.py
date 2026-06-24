"""Multi-stream concurrency / correctness test for the unconstrained L-BFGS.

This exercises the new in-tree GPU L-BFGS optimizer (optimizer_type == 2, the
default in cudaOptimizeSite) across many CUDA streams running concurrently —
the scenario phase 4 of the rewrite must validate with compute-sanitizer.

Two things are checked:

  1. Functional correctness: the coupling score recovered from the optimized
     parameters still correlates with the reference score matrix
     (Spearman > 0.99), the same bar as test_coupling.py.

  2. Stream isolation: running every site across `--streams` concurrent
     streams produces (near) identical couplings to a single-stream baseline.
     If the per-site L-BFGS history buffers ever leaked across streams, the
     two runs would diverge.

Run it directly under compute-sanitizer to catch races / OOB accesses:

    compute-sanitizer --tool memcheck  python tests/test_lbfgs_concurrency.py --streams 8
    compute-sanitizer --tool racecheck python tests/test_lbfgs_concurrency.py --streams 8

or as a plain functional check:

    python tests/test_lbfgs_concurrency.py --streams 8
"""
import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from massivora import BinaryAlignment, cpp_bindings
from massivora.analyzer import ComplexAnalyzer
from massivora.executors.couple import BaseCouplingExecutor

TEST_ZARR = os.path.join(os.path.dirname(__file__), "RL4_ECOLI-RL31_ECOLI")
STANDARD_SCORE = os.path.join(os.path.dirname(__file__), "score.npy")
CHAIN_A_LENGTH = 201


def _optimize_all_sites(MSA_pad, MSA, W, B, N, q, q_pad, n_streams, hyperparams):
    """Optimize every site, fanned out over `n_streams` concurrent streams.

    Each worker thread owns exactly one stream and walks a disjoint slice of
    sites (r = tid, tid + n_streams, ...), so no stream is ever shared by two
    simultaneously-running sites. Returns a fresh (N, n_params) fp16 array of
    optimized parameters.
    """
    import cupy as cp

    Jr_pad_total_params = N * q_pad * q_pad
    n_params = q_pad + Jr_pad_total_params
    x0_pad = cp.zeros((N, n_params), dtype=cp.float16)

    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(n_streams)]

    def run_stream(tid):
        stream = streams[tid]
        out = []
        for r in range(tid, N, n_streams):
            iters, pll_r = cpp_bindings.cudaOptimizeSite(
                MSA_pad.data.ptr,
                MSA.data.ptr,
                W.data.ptr,
                np.int32(r), np.int32(B), np.int32(N),
                np.int32(q), np.int32(q_pad),
                np.float32(0.01), np.float32(0.01),
                np.float32(3e-4), np.int32(500),
                x0_pad[r].data.ptr,
                hyperparams, 0, stream.ptr,
            )
            out.append((r, int(iters), float(pll_r)))
        return out

    results = {}
    with ThreadPoolExecutor(max_workers=n_streams) as pool:
        futures = [pool.submit(run_stream, tid) for tid in range(n_streams)]
        for future in as_completed(futures):
            for r, iters, pll_r in future.result():
                results[r] = (iters, pll_r)
    for s in streams:
        s.synchronize()
    return x0_pad, results


def _score_from_params(x0_pad, N, q, q_pad):
    import cupy as cp

    from massivora.utils import get_cuda_module

    cuda_module = get_cuda_module()
    J_result = (x0_pad[:, q_pad:]
                .reshape(N, N, q_pad, q_pad)[:, :, :q, :q]
                .astype(cp.float32))
    cuda_module.get_function("apply_ising_gauge")(
        (N, N), (q, q), (J_result, np.int32(N), np.int32(q)))
    J = cp.asnumpy(J_result.reshape(N, N * q * q)).T
    return BaseCouplingExecutor.compute_score(J, q=q).astype("float16")


def run(n_streams, hyperparams=None):
    import cupy as cp

    hyperparams = hyperparams or {}

    align = BinaryAlignment(TEST_ZARR)
    align.Reweight_Sequence(0.8, True)
    MSA = cp.asarray(align.matrix, dtype=cp.int8)
    B, N = MSA.shape
    W = cp.asarray(align.weights, dtype=np.float32)
    W /= float(align.Beff)
    q = int(MSA.max()) + 1
    q_pad = ((q + 7) // 8) * 8

    b_idx = cp.arange(B)[:, None]
    i_idx = cp.arange(N)[None, :]
    MSA_pad = cp.zeros((B, N, q_pad), dtype=cp.float16)
    MSA_pad[b_idx, i_idx, MSA] = 1.0

    print(f"MSA shape: {tuple(MSA.shape)}, Beff: {align.Beff}, q={q}, "
          f"streams={n_streams}", flush=True)

    # --- multi-stream run ---
    x_multi, results = _optimize_all_sites(
        MSA_pad, MSA, W, B, N, q, q_pad, n_streams, hyperparams)
    cp.cuda.Stream.null.synchronize()

    bad = [r for r, (_, pll_r) in results.items()
           if not np.isfinite(pll_r)]
    assert not bad, f"Non-finite PLL at sites: {bad}"

    score_multi = _score_from_params(x_multi, N, q, q_pad)
    ref = ComplexAnalyzer(STANDARD_SCORE, chainA_length=CHAIN_A_LENGTH)

    # Write into the test zarr the same way the pipeline does, then score.
    import zarr
    grp = zarr.open_group(store=TEST_ZARR)
    z = grp.require_array(name="couplings", shape=score_multi.shape,
                          dtype="float16", overwrite=True)
    z[:] = score_multi
    this_score = ComplexAnalyzer(TEST_ZARR, chainA_length=CHAIN_A_LENGTH)
    spearman, _ = ref.SpearmanCorrelation(this_score)
    print(f"[streams={n_streams}] Spearman vs reference: {spearman:.6f}",
          flush=True)
    assert spearman > 0.99, f"Spearman too low: {spearman}"

    # --- single-stream baseline for stream-isolation check ---
    if n_streams > 1:
        x_single, _ = _optimize_all_sites(
            MSA_pad, MSA, W, B, N, q, q_pad, 1, hyperparams)
        cp.cuda.Stream.null.synchronize()
        a = cp.asnumpy(x_multi).astype(np.float32)
        b = cp.asnumpy(x_single).astype(np.float32)
        max_abs = float(np.max(np.abs(a - b)))
        print(f"max|x_multi - x_single| = {max_abs:.3e}", flush=True)
        # fp16 storage + nondeterministic atomics in the PLL kernels give a
        # small but bounded difference; a leak across streams would be large.
        assert max_abs < 1e-2, (
            f"Multi-stream result diverged from single-stream baseline "
            f"(max abs diff {max_abs:.3e}); possible cross-stream corruption.")

    print("OK", flush=True)
    return spearman


def test_lbfgs_concurrency():
    """Pytest entry point (GPU required)."""
    try:
        import cupy  # noqa: F401
    except ImportError:
        import pytest
        pytest.skip("CuPy not installed; GPU concurrency test skipped")
    run(n_streams=8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--streams", type=int, default=8,
                        help="number of concurrent CUDA streams")
    parser.add_argument("--m-corr", type=int, default=None,
                        help="L-BFGS history length (default: optimizer default)")
    args = parser.parse_args()
    hp = {}
    if args.m_corr is not None:
        hp["m_corr"] = float(args.m_corr)
    run(args.streams, hyperparams=hp)
