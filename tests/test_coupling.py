import os
import numpy as np
from multiprocessing import shared_memory, cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
from massivora.utils import massivora_pkg_dir, get_cuda_module
from massivora import cpp_bindings
from massivora import BinaryAlignment
from massivora.executors.couple import BaseCouplingExecutor
from massivora.analyzer import ComplexAnalyzer
import zarr

J_file = os.path.join(os.path.dirname(__file__), "J.npy")
TEST_ZARR = os.path.join(os.path.dirname(__file__), "RL4_ECOLI-RL31_ECOLI")
STANDARD_SCORE = os.path.join(os.path.dirname(__file__), "score.npy")

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
    msa_shm = shared_memory.SharedMemory(name="RL4_ECOLI-RL31_ECOLI", create=True, size=msa_size+w_size)
    msa_arr = np.ndarray(MSA.shape, dtype=np.int32, buffer=msa_shm.buf)
    w_arr = np.ndarray(W.shape, dtype=np.float64, buffer=msa_shm.buf[msa_size:])
    np.copyto(msa_arr, MSA)
    np.copyto(w_arr, W)

    J_shm = shared_memory.SharedMemory(name="RL4_ECOLI-RL31_ECOLI_J", create=True, size=N*N*q*q*8)

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
    msa_shm = shared_memory.SharedMemory(name="RL4_ECOLI-RL31_ECOLI", create=True, size=msa_size+w_size)
    msa_arr = np.ndarray(MSA.shape, dtype=np.int32, buffer=msa_shm.buf)
    w_arr = np.ndarray(W.shape, dtype=np.float64, buffer=msa_shm.buf[msa_size:])
    np.copyto(msa_arr, MSA)
    np.copyto(w_arr, W)

    # The GPU executable writes J as float32 (4 bytes), unlike the CPU's float64.
    J_shm = shared_memory.SharedMemory(name="RL4_ECOLI-RL31_ECOLI_J", create=True, size=N*N*q*q*4)

    massivora_dir = massivora_pkg_dir()
    plm_opt_exe = os.path.join(massivora_dir, 'bin', 'plm_opt_site')

    n_streams = 1
    rv = subprocess.run([plm_opt_exe, "--use-gpu", "RL4_ECOLI-RL31_ECOLI", str(B), str(N), str(q),
                         str(0.01), str(0.01), str(3e-4), str(500), str(n_streams)])
    assert rv.returncode == 0, f"plm_opt_site (GPU) failed with rc={rv.returncode}"

    # Row r holds site r's (N, q, q) coupling blocks flattened to (N*q*q,).
    J = np.ndarray((N, N*q*q), dtype=np.float32, buffer=J_shm.buf)
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

def test_coupling_GPU():
    try:
        import cupy as cp
    except ImportError:
        print("CuPy is not installed. Skipping GPU test.")
        assert False, "GPU is not available for testing"

    align = BinaryAlignment(TEST_ZARR)
    align.Reweight_Sequence(0.8, True)
    MSA = cp.asarray(align.matrix, dtype=cp.int8)
    B, N = MSA.shape
    W = cp.asarray(align.weights, dtype=np.float32)
    Beff = float(align.Beff)
    W /= Beff

    q = int(MSA.max()) + 1

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

    # Create CUDA streams for parallel execution
    n_streams = 1
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(n_streams)]
    def run_site(r):
        stream = streams[r % n_streams]
        iters, pll_r = cpp_bindings.cudaOptimizeSite(
            MSA_pad.data.ptr,
            MSA.data.ptr,
            W.data.ptr,
            np.int32(r), np.int32(B), np.int32(N),
            np.int32(q), np.int32(q_pad),
            np.float32(0.01), np.float32(0.01),
            np.float32(3e-4), np.int32(500),
            x0_pad[r].data.ptr,
            {}, 0, stream.ptr
        )
        return r, iters, pll_r

    with ThreadPoolExecutor(max_workers=n_streams) as pool:
        futures = {pool.submit(run_site, r): r for r in range(N)}
        for future in as_completed(futures):
            r, iters, pll_r = future.result()
            pll[r] = pll_r
            print(f"site {r} iter {iters}: PLL = {float(pll_r):.6f}")

    J_result = (x0_pad[:, q_pad:]
                .reshape(N, N, q_pad, q_pad)[:, :, :q, :q]
                .astype(cp.float32))
    cuda_module.get_function("apply_ising_gauge")(
        (N, N), (q, q), (J_result, np.int32(N), np.int32(q)))

    J = cp.asnumpy(J_result.reshape(N, N * q * q)).T
    score = BaseCouplingExecutor.compute_score(J, q=q).astype('float16')

    grp = zarr.open_group(store=TEST_ZARR)
    z = grp.require_array(name="couplings", shape=score.shape, dtype='float16', overwrite=True)
    z[:] = score

    this_score = ComplexAnalyzer(TEST_ZARR, chainA_length=201)
    standard_score = ComplexAnalyzer(STANDARD_SCORE, chainA_length=201)

    spearman, p = standard_score.SpearmanCorrelation(this_score)

    assert spearman > 0.99, f"Spearman correlation is too low: {spearman}"