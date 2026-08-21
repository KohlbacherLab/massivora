import os
import numpy as np
from multiprocessing import shared_memory, cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
from massivora.utils import massivora_pkg_dir, get_cuda_module
from massivora import cpp_bindings
from massivora import BinaryAlignment
from massivora.executors.couple import BaseCouplingExecutor
from massivora.analyzer import ComplexAnalyzer, MonomerAnalyzer
import zarr

J_file = os.path.join(os.path.dirname(__file__), "J.npy")
TEST_ZARR = os.path.join(os.path.dirname(__file__), "RL4_ECOLI-RL31_ECOLI")
STANDARD_SCORE = os.path.join(os.path.dirname(__file__), "score.npy")
STANDARD_SCORE_GAUSS = os.path.join(os.path.dirname(__file__), "RL4_ECOLI-RL31_ECOLI.csv")

def test_compute_score():
    J = np.load(J_file).T
    score = BaseCouplingExecutor.compute_score(J, q=21, gap_idx=0)
    real = ComplexAnalyzer(STANDARD_SCORE, chainA_length=201)
    calcd = ComplexAnalyzer(score, chainA_length=201)
    assert real.SpearmanCorrelation(calcd)[0] > 0.99, "Spearman correlation should be > 0.99"

def test_coupling_CPU():
    align = BinaryAlignment(TEST_ZARR)
    MSA = align.matrix.astype(np.int32)
    B, N = MSA.shape
    print(f"MSA shape: {MSA.shape}, Beff: {align.Beff}", flush=True)
    W = np.array(align.weights, dtype=np.float64)
    Beff = float(align.Beff)
    W /= Beff

    q = int(MSA.max()) + 1

    msa_size = MSA.nbytes
    w_size = W.nbytes
    msa_shm = shared_memory.SharedMemory(name="Massivora_RL4_ECOLI-RL31_ECOLI", create=True, size=msa_size+w_size)
    msa_arr = np.ndarray(MSA.shape, dtype=np.int32, buffer=msa_shm.buf)
    w_arr = np.ndarray(W.shape, dtype=np.float64, buffer=msa_shm.buf[msa_size:])
    np.copyto(msa_arr, MSA)
    np.copyto(w_arr, W)

    J_shm = shared_memory.SharedMemory(name="Massivora_RL4_ECOLI-RL31_ECOLI_J", create=True, size=N*N*q*q*8)

    massivora_dir = massivora_pkg_dir()
    plm_opt_exe = os.path.join(massivora_dir, 'bin', 'plm_opt_site')

    def optimize_site(r):
        rv = subprocess.run([plm_opt_exe, "RL4_ECOLI-RL31_ECOLI", str(r), str(B), str(N), str(q), str(0.01), str(0.01), str(1e-4), str(500)])
        return r, rv.returncode
    max_workers = cpu_count()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(optimize_site, r) for r in range(N)]
        for future in futures:
            r, returncode = future.result()

    J = np.ndarray((N*q*q, N), dtype=np.float64, buffer=J_shm.buf)

    score = BaseCouplingExecutor.compute_score(J, q=q).astype(np.float16)

    grp = zarr.open_group(store=TEST_ZARR)
    z = grp.require_array(name="couplings", shape=score.shape, dtype='float16', overwrite=True)
    z[:] = score

    try:
        del J
        J_shm.close()
        J_shm.unlink()
        msa_shm.close()
        msa_shm.unlink()
    except FileNotFoundError:
        pass

    this_score = ComplexAnalyzer(TEST_ZARR, chainA_length=201)
    standard_score = ComplexAnalyzer(STANDARD_SCORE, chainA_length=201)

    spearman, p = standard_score.SpearmanCorrelation(this_score)

    assert spearman > 0.99, f"Spearman correlation is too low: {spearman}"

def test_coupling_GPU_executable():
    """Drive the external CUDA-built plm_opt_site executable.

    Mirrors test_coupling_CPU(), but uses the GPU executable's argument
    signature (no per-site index; whole pair in one process) and reads back
    the float32 J it writes to shared memory.
    """
    try:
        import cupy as cp  # noqa: F401  -- used only as a GPU-availability gate
    except ImportError:
        print("CuPy is not installed. Skipping GPU executable test.")
        assert False, "GPU is not available for testing"

    align = BinaryAlignment(TEST_ZARR)
    MSA = align.matrix.astype(np.int32)
    B, N = MSA.shape
    print(f"MSA shape: {MSA.shape}, Beff: {align.Beff}", flush=True)
    W = np.array(align.weights, dtype=np.float64)
    Beff = float(align.Beff)
    W /= Beff

    q = int(MSA.max()) + 1

    # Shared-memory layout is identical to the CPU pipeline: int32 MSA + float64 W.
    msa_size = MSA.nbytes
    w_size = W.nbytes
    msa_shm = shared_memory.SharedMemory(name="Massivora_RL4_ECOLI-RL31_ECOLI", create=True, size=msa_size+w_size)
    msa_arr = np.ndarray(MSA.shape, dtype=np.int32, buffer=msa_shm.buf)
    w_arr = np.ndarray(W.shape, dtype=np.float64, buffer=msa_shm.buf[msa_size:])
    np.copyto(msa_arr, MSA)
    np.copyto(w_arr, W)

    # The GPU executable writes J as float32 (4 bytes), unlike the CPU's float64.
    J_shm = shared_memory.SharedMemory(name="Massivora_RL4_ECOLI-RL31_ECOLI_J", create=True, size=N*N*q*q*4)

    massivora_dir = massivora_pkg_dir()
    plm_opt_exe = os.path.join(massivora_dir, 'bin', 'plm_opt_site')

    n_streams = 4
    rv = subprocess.run([plm_opt_exe, "--use-gpu", "RL4_ECOLI-RL31_ECOLI", str(B), str(N), str(q),
                         str(0.01), str(0.01), str(3e-4), str(500), str(n_streams)])
    assert rv.returncode == 0, f"plm_opt_site (GPU) failed with rc={rv.returncode}"

    # Row r holds site r's (N, q, q) coupling blocks flattened to (N*q*q,).
    J = np.ndarray((N, N*q*q), dtype=np.float32, buffer=J_shm.buf)
    # The GPU executable no longer applies the Ising gauge; do it here (in place).
    cpp_bindings.applyIsingGauge(J, q)
    score = BaseCouplingExecutor.compute_score(J.T, q=q).astype(np.float16)

    grp = zarr.open_group(store=TEST_ZARR)
    z = grp.require_array(name="couplings", shape=score.shape, dtype='float16', overwrite=True)
    z[:] = score

    try:
        del J
        J_shm.close()
        J_shm.unlink()
        msa_shm.close()
        msa_shm.unlink()
    except FileNotFoundError:
        pass

    this_score = ComplexAnalyzer(TEST_ZARR, chainA_length=201)
    standard_score = ComplexAnalyzer(STANDARD_SCORE, chainA_length=201)

    spearman, p = standard_score.SpearmanCorrelation(this_score)

    assert spearman > 0.99, f"Spearman correlation is too low: {spearman}"

def run_coupling_GPU(zarr_path, *, lambdaH=0.01, lambdaJ=0.01, eps=1e-4,
                     maxeval=500, n_streams=4, reweight_theta=0.8, verbose=True):
    import cupy as cp

    align = BinaryAlignment(zarr_path)
    align.Reweight_Sequence(reweight_theta, True)
    MSA = cp.asarray(align.matrix, dtype=cp.int8)
    B, N = MSA.shape
    W = cp.asarray(align.weights, dtype=np.float32)
    Beff = float(align.Beff)
    W /= Beff

    q = int(MSA.max()) + 1
    if verbose:
        print(f"MSA shape: {MSA.shape}, Beff: {Beff}, q: {q}", flush=True)

    # Get the cached CUDA module (automatically uses device 0 via CUDA_VISIBLE_DEVICES)
    cuda_module = get_cuda_module()

    q_pad = ((q + 7) // 8) * 8
    pll = cp.zeros(N, dtype=cp.float32)
    Jr_pad_total_params = N * q_pad * q_pad
    n_params = q_pad + Jr_pad_total_params
    x0_pad = cp.zeros((N, n_params), dtype=cp.float16)

    b_idx = cp.arange(B)[:, None]
    i_idx = cp.arange(N)[None, :]
    MSA_pad = cp.zeros((B, N, q_pad), dtype=cp.float16)
    MSA_pad[b_idx, i_idx, MSA] = 1.0

    # Create CUDA streams for parallel execution.
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(n_streams)]

    def run_stream(tid):
        stream = streams[tid]
        results = []
        for r in range(tid, N, n_streams):
            iters, pll_r = cpp_bindings.cudaOptimizeSite(
                MSA_pad.data.ptr,
                MSA.data.ptr,
                W.data.ptr,
                np.int32(r), np.int32(B), np.int32(N),
                np.int32(q), np.int32(q_pad),
                np.float32(lambdaH), np.float32(lambdaJ),
                np.float32(eps), np.int32(maxeval),
                x0_pad[r].data.ptr,
                {}, 0, stream.ptr
            )
            results.append((r, iters, pll_r))
        return results

    with ThreadPoolExecutor(max_workers=n_streams) as pool:
        futures = [pool.submit(run_stream, tid) for tid in range(n_streams)]
        for future in as_completed(futures):
            for r, iters, pll_r in future.result():
                pll[r] = pll_r
                if verbose:
                    print(f"site {r} iter {iters}: PLL = {float(pll_r):.6f}")

    J_result = (x0_pad[:, q_pad:]
                .reshape(N, N, q_pad, q_pad)[:, :, :q, :q]
                .astype(cp.float32))
    cuda_module.get_function("apply_ising_gauge")(
        (N, N), (q, q), (J_result, np.int32(N), np.int32(q)))

    J = cp.asnumpy(J_result.reshape(N, N * q * q)).T
    score = BaseCouplingExecutor.compute_score(J, q=q).astype('float16')

    grp = zarr.open_group(store=zarr_path)
    z = grp.require_array(name="couplings", shape=score.shape, dtype='float16', overwrite=True)
    z[:] = score
    return score


def test_coupling_GPU():
    try:
        import cupy as cp  # noqa: F401  -- used only as a GPU-availability gate
    except ImportError:
        print("CuPy is not installed. Skipping GPU test.")
        assert False, "GPU is not available for testing"

    run_coupling_GPU(TEST_ZARR)

    this_score = ComplexAnalyzer(TEST_ZARR, chainA_length=201)
    standard_score = ComplexAnalyzer(STANDARD_SCORE, chainA_length=201)

    spearman, p = standard_score.SpearmanCorrelation(this_score)

    assert spearman > 0.99, f"Spearman correlation is too low: {spearman}"

def test_coupling_gauss():
    align = BinaryAlignment(TEST_ZARR)
    massivora_dir = massivora_pkg_dir()
    gauss_infer_exe = os.path.join(massivora_dir, 'bin', 'gauss_infer')

    B, N = align.matrix.shape
    # RAW reweighting vector
    W = np.asarray(align.weights, dtype=np.float64)
    Beff = float(align.Beff)
    q = 21

    # Hand the alignment over as int8 in residue-major (N, M) order
    Zt = np.ascontiguousarray(align.matrix.T, dtype=np.int8)   # (N, M)
    msa_size = Zt.nbytes
    w_size = W.nbytes
    msa_shm = shared_memory.SharedMemory(name="Massivora_test", create=True, size=msa_size + w_size)
    msa_arr = np.ndarray(Zt.shape, dtype=np.int8, buffer=msa_shm.buf)
    w_arr = np.ndarray(W.shape, dtype=np.float64, buffer=msa_shm.buf[msa_size:])
    np.copyto(msa_arr, Zt)
    np.copyto(w_arr, W)

    score_shm = shared_memory.SharedMemory(name='Massivora_test_J', create=True, size=N * N * 4)

    cmd = [gauss_infer_exe, 'test', str(B), str(N), str(q),
            str(0.8), str(1)]

    rv = subprocess.run(cmd)
    score = np.ndarray((N, N), dtype=np.float32, buffer=score_shm.buf).astype(np.float16)
    grp = zarr.open_group(store=TEST_ZARR)
    z = grp.require_array(name="couplings", shape=score.shape, dtype='float16', overwrite=True)
    z[:] = score
    try:
        score_shm.close()
        score_shm.unlink()
        msa_shm.close()
        msa_shm.unlink()
    except FileNotFoundError:
        pass
    standard_score = MonomerAnalyzer(STANDARD_SCORE_GAUSS)
    this_score = MonomerAnalyzer(TEST_ZARR)

    spearman, p = standard_score.SpearmanCorrelation(this_score)
    
    assert spearman > 0.99, f"Spearman correlation is too low: {spearman}"

if __name__ == "__main__":
    # Drive the GPU coupling computation (the test_coupling_GPU logic) on an
    # arbitrary zarr alignment, e.g.:
    #   python tests/test_coupling.py /path/to/PAIR_alignment
    # The couplings are written back to the alignment's 'couplings' array.
    import argparse
    import time

    ap = argparse.ArgumentParser(
        description="Run the in-process GPU PLM coupling optimizer on a zarr alignment.")
    ap.add_argument("zarr_path", nargs="?", default=TEST_ZARR,
                    help="path to the zarr alignment group "
                         "(default: the bundled RL4/RL31 test pair)")
    ap.add_argument("--lambdaH", type=float, default=0.01, help="field L2 regularization")
    ap.add_argument("--lambdaJ", type=float, default=0.01, help="coupling L2 regularization")
    ap.add_argument("--eps", type=float, default=1e-4, help="convergence tolerance")
    ap.add_argument("--maxeval", type=int, default=500, help="max iterations per site")
    ap.add_argument("--streams", type=int, default=4, help="number of CUDA streams")
    ap.add_argument("--theta", type=float, default=0.8,
                    help="sequence-reweighting identity threshold")
    ap.add_argument("-q", "--quiet", action="store_true",
                    help="suppress per-site progress output")
    args = ap.parse_args()

    # Pure speed run: no correctness/assertion checks, just wall-clock timing.
    t0 = time.perf_counter()
    run_coupling_GPU(args.zarr_path,
                     lambdaH=args.lambdaH, lambdaJ=args.lambdaJ, eps=args.eps,
                     maxeval=args.maxeval, n_streams=args.streams,
                     reweight_theta=args.theta, verbose=not args.quiet)
    dt = time.perf_counter() - t0
    print(f"Done in {dt:.2f}s; couplings written to {args.zarr_path}", flush=True)