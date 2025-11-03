import numpy as np
import multiprocessing as mp
from scipy.optimize import minimize
import time
import tracemalloc
from functools import partial
import compute
from alignment import TextAlignment
from concurrent.futures import ThreadPoolExecutor
import numba
import math
from numba import cuda
# import cupy as cp

# # num_gpus = torch.cuda.device_count()
# data_type_cpu = np.float32
# data_type_gpu = cp.float32

def compute_similarity_matrix_CPPKernel(MSA, x=0.8):
    B, N = MSA.shape
    identical_threshold = x * N
    simM = np.full((B, B), N, dtype=np.int16)
    compute.compute_similarity_matrix(MSA, B, N, simM)

    # Calculate weights
    m = np.sum(simM >= identical_threshold, axis=0)
    w = 1.0 / m.astype(np.float32)
    Beff = np.sum(w)

    return w, Beff

@cuda.jit
def similarity_matrix_kernel(msa, simM, B, N):
    b = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    if b < B - 1:
        # Process all sequences after the current one
        for i in range(b + 1, B):
            identical_count = 0

            # Compare sequences using equality checks (Numba optimizes this)
            for j in range(N):
                if msa[b, j] == msa[i, j]:  # Equivalent to XOR == 0
                    identical_count += 1

            # Fill similarity matrix (symmetric)
            simM[b, i] = identical_count
            simM[i, b] = identical_count


def reweight_sequence_numba(MSA, x=0.8):
    cuda.select_device(0)
    MSAkernel = cuda.as_cuda_array(MSA)
    B, N = MSA.shape
    identical_threshold = x * N

    # Initialize similarity matrix with N on diagonal
    simM = cp.full((B, B), N, dtype=cp.int16)
    simMkernel = cuda.as_cuda_array(simM)

    # Launch kernel with appropriate block/grid dimensions
    threads_per_block = 128
    blocks_per_grid = math.ceil(B / threads_per_block)

    similarity_matrix_kernel[blocks_per_grid, threads_per_block](MSAkernel, simMkernel, B, N)

    # Calculate weights
    m = cp.sum(simM >= identical_threshold, axis=0)
    w = 1.0 / m.astype(cp.float32)
    Beff = cp.sum(w)

    # cuda.close()

    return w, Beff

def reweight_sequence(MSA, x=0.8):
    B = MSA.shape[0]
    N = MSA.shape[1]
    identical_threshold = x * N
    simM = np.full((B,B), N, dtype=np.int16)

    # TODO: parallelize this
    for b in range(B-1):
        # print(f"Calculating weights for sequence {b+1}/{B}")
        identical_positions = np.equal(MSA[b+1:], MSA[b])
        identity_scores = np.sum(identical_positions, axis=1)
        simM[b, b+1:] = identity_scores
        simM[b+1:, b] = identity_scores
    m = np.sum(simM >= identical_threshold, axis=0)
    w = 1/m
    Beff = w.sum()

    print(f"Effective number of sequences: {Beff}")
    print(f"Effective weights: {w}")
    return w, Beff

def reweight_sequence_torch_cuda(MSA, x=0.8, tocpu=True):
    if isinstance(MSA, np.ndarray):
        MSA = torch.from_numpy(MSA).cuda()
    else:
        MSA = MSA.cuda()

    B, N = MSA.shape
    identical_threshold = x * N
    device = MSA.device

    simM = torch.full((B,B), N, dtype=torch.int16, device=device)

    for b in range(B-1):
        current_seq = MSA[b].unsqueeze(0).unsqueeze(0)
        remaining = MSA[b+1:].unsqueeze(0)
        identical_positions = (current_seq == remaining)
        identity_scores = identical_positions.sum(dim=-1).squeeze(0)
        simM[b, b+1:] = identity_scores
        simM[b+1:, b] = identity_scores
    m = torch.sum(simM >= identical_threshold, dim=0)
    w = 1.0 / m.float()
    Beff = w.sum()

    if tocpu:
        w = w.cpu()
        Beff = Beff.cpu()
    return w, Beff

def reweight_sequence_torch_cuda_memory_efficient(MSA, x=0.8, tocpu=True):
    """Memory efficient version of reweight_sequence_cuda. 
    40% slower than the original, but saves 80% of memory."""
    
    if isinstance(MSA, np.ndarray):
        MSA = torch.from_numpy(MSA).cuda()
    else:
        MSA = MSA.cuda()

    B, N = MSA.shape
    identical_threshold = x * N
    device = MSA.device

    m = torch.ones(B, device=device)

    for b in range(B-1):
        current_seq = MSA[b].unsqueeze(0).unsqueeze(0)
        remaining = MSA[b+1:].unsqueeze(0)
        identical_positions = (current_seq == remaining)
        identity_scores = identical_positions.sum(dim=-1).squeeze(0)
        similar_seqs = (identity_scores >= identical_threshold)
        m[b] += similar_seqs.sum()
        indices = torch.where(similar_seqs)[0] + b + 1
        m.index_add_(0, indices, torch.ones(len(indices), device=device))

    w = 1.0 / m.float()
    Beff = w.sum()

    if tocpu:
        w = w.cpu()
        Beff = Beff.cpu()
    return w, Beff

@cuda.jit
def calc_grad_pll_kernel(J, h, MSA, r, pll, W, gradients, Jgradients):
    B, N = MSA.shape
    _, q = J.shape[0], J.shape[1]

    b = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if b < B:
        energies = numba.cuda.local.array(32, dtype=numba.float64)
        for l in range(q):
            sum_val = 0.0
            for i in range(r):
                k = MSA[b, i]
                sum_val += J[i, k, l]
            for i in range(r+1, N):
                k = MSA[b, i]
                sum_val += J[i-1, k, l]

            energies[l] = h[l] + sum_val
        max_energy = energies[0]
        for l in range(1, q):
            if energies[l] > max_energy:
                max_energy = energies[l]
        sum_exp = 0.0
        for l in range(q):
            sum_exp += numba.cuda.libdevice.exp(energies[l])

        lnorm = max_energy + numba.cuda.libdevice.log(sum_exp - max_energy)

        pll_contrib = -W[b] * (energies[MSA[b, r]] - lnorm)
        cuda.atomic.add(pll, 0, pll_contrib)

        vGrad = numba.cuda.local.array(32, dtype=numba.float64)
        for l in range(q):
            Ps = numba.cuda.libdevice.exp(energies[l] - lnorm)
            indicator = 1 if MSA[b, r] == l else 0
            vGrad[l] = W[b] * (indicator - Ps)

        for i in range(q):
            cuda.atomic.add(gradients, i, -vGrad[i])
        for i in range(r):
            s_ib = MSA[b, i]
            for l in range(q):
                cuda.atomic.add(Jgradients, (i, s_ib, l), -vGrad[l])
        for i in range(r+1, N):
            s_ib = MSA[b, i]
            for l in range(q):
                cuda.atomic.add(Jgradients, (i-1, s_ib, l), -vGrad[l])
@cuda.jit
def normalization_kernel(x, gradients, lambdaH, lambdaJ, nParams, q, reg):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    if i < nParams:
        if i < q:
            gradients[i] += 2.0 * lambdaH * x[i]
            cuda.atomic.add(reg, 0, x[i] ** 2)
        else:
            gradients[i] += lambdaJ * x[i]
            cuda.atomic.add(reg, 1, x[i] ** 2)

def compute_gradients(x, MSA, r, W, lambdaH, lambdaJ):
    B, N = MSA.shape
    pll = cp.zeros(1, dtype=data_type_gpu)
    gradients = cp.zeros_like(x)
    Jgradients = gradients[q:].reshape((N-1, q, q))

    threads_per_block = 128
    blocks_per_grid = math.ceil(B / threads_per_block)

    calc_grad_pll_kernel[blocks_per_grid, threads_per_block](
        cuda.as_cuda_array(x[q:].reshape((N-1, q, q))), cuda.as_cuda_array(x), MSA, r, pll, W,
        cuda.as_cuda_array(gradients), cuda.as_cuda_array(Jgradients)
    )
    reg = cp.zeros(2, dtype=data_type_gpu)
    blocks_for_norm = math.ceil(len(x) / threads_per_block)
    normalization_kernel[blocks_for_norm, threads_per_block](
        x, cuda.as_cuda_array(gradients), lambdaH, lambdaJ, len(x), q, reg
    )
    pll[0] += lambdaH * reg[0] + lambdaJ * reg[1] * 0.5
    
    return pll[0], gradients


def compute_score(J, q=21, gap_idx=20):
    # Average the J matrix out at main diagonal
    N = J.shape[1]
    J = J.T.reshape((N, N, q, q))
    J = (J + J.transpose(1, 0, 3, 2)) / 2

    # Gap exclude Frobenius norm
    mask = np.ones((q, q), dtype=bool)
    mask[gap_idx, :] = False
    mask[:, gap_idx] = False
    J_filtered = J[:, :, mask].reshape(N, N, -1)
    FN = np.sqrt(np.sum(J_filtered ** 2, axis=-1))

    # Average Product Correction of Frobenius norm
    row_mean = np.sum(FN, axis=1, keepdims=True) / (N - 1)
    col_mean = np.sum(FN, axis=0, keepdims=True) / (N - 1)
    total_mean = np.sum(FN) / (N * (N - 1))
    FN_APC = FN - (row_mean @ col_mean) / total_mean

    return FN_APC


def target_func(r, x0, opt, q, N, B, MSA, W, lambdaH, lambdaJ):
    result = minimize(
        partial(compute.perSitePllGradient, r=r, q=q, N=N, B=B, MSA=MSA, W=W, lambdaH=lambdaH, lambdaJ=lambdaJ, num_threads=1),
        x0,
        jac=True,
        method='L-BFGS-B',
        options={
            'ftol': opt["epsconv"],
            'gtol': opt["epsconv"],
            'maxiter': opt["maxit"],
            'disp': False
        }
    )
    return r, result
def minimize_pl_asym(opt, q, N, B, MSA, W, lambdaH, lambdaJ):
    nParamsh = q
    nParamsJ = (N - 1) * q * q
    nParams = nParamsh + nParamsJ
    x0 = np.zeros(nParams, dtype=np.float32)
    pll = np.zeros(N, dtype=np.float32)
    J = np.zeros((nParamsJ + q*q, N), dtype=np.float32)


    # for r in range(N):
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.starmap(target_func, [(r, x0, opt, q, N, B, MSA, W, lambdaH, lambdaJ) for r in range(N)])
        # minf = result.fun
        # minx = result.x
        # minJ = minx[q:]
        # compute.applyIsingGauge(minJ, q)
        # # insert zeros for speeding up tensor mapping
        # minJ = np.insert(minJ, r * q * q, np.zeros(q * q, dtype=minJ.dtype))
        # pll[r] = minf
        # J[:, r] = minJ
        total_nits = 0
        for r, result in results:
            print(f"Finished site {r+1}/{N} after {result.nit} iterations with pseudolikelihood {result.fun}")
            total_nits += result.nit
    print(f"Total iterations for all sites: {total_nits}")
    return J, pll
def minimize_pl_asym_threaded(opt, q, N, B, MSA, W, lambdaH, lambdaJ):
    """Use threading instead of multiprocessing to avoid data copying"""
    nParamsh = q
    nParamsJ = (N - 1) * q * q
    nParams = nParamsh + nParamsJ
    x0 = np.zeros(nParams, dtype=np.float32)
    pll = np.zeros(N, dtype=np.float32)
    J = np.zeros((nParamsJ + q*q, N), dtype=np.float32)
    
    # Build index map once
    index_map = compute.buildIndexMap(q, N)
    
    def optimize_site(r):
        """Optimize single site with C++ OpenMP threading"""
        result = minimize(
            partial(compute.perSitePllGradient,
                    r=r, q=q, N=N, B=B, MSA=MSA, W=W,
                    lambdaH=lambdaH, lambdaJ=lambdaJ,
                    num_threads=1),  # Each C++ call uses 1 thread
            x0,
            jac=True,
            method='L-BFGS-B',
            options={
                'ftol': opt["epsconv"],
                'gtol': opt["epsconv"],
                'maxiter': opt["maxit"]
            }
        )
        
        minf = result.fun
        minx = result.x
        minJ = minx[q:]
        compute.applyIsingGauge(minJ, q)
        minJ = np.insert(minJ, r * q * q, np.zeros(q * q, dtype=minJ.dtype))
        
        return r, minf, minJ, result.nit
    
    # Use thread pool (shared memory, no data copying)
    max_workers = mp.cpu_count()
    total_nits = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(optimize_site, r) for r in range(N)]
        
        for future in futures:
            r, minf, minJ, nit = future.result()
            pll[r] = minf
            J[:, r] = minJ
            total_nits += nit
            print(f"Finished site {r+1}/{N} after {nit} iterations with pseudolikelihood {minf}")
    
    print(f"Total iterations for all sites: {total_nits}")
    return J, pll
def minimize_pl_asym_new(opt, q, N, B, MSA, W, lambdaH, lambdaJ):
    nParamsh = q
    nParamsJ = (N - 1) * q * q
    nParams = nParamsh + nParamsJ
    print(f"Total number of parameters: {nParams}")
    x0 = np.zeros(nParams, dtype=np.float32)
    pll = np.zeros(N, dtype=np.float32)
    J = np.zeros((nParamsJ + q*q, N), dtype=np.float32)
    index_map = compute.buildIndexMap(q, N)
    
    for r in range(N):
        result = minimize(
            partial(compute.perSitePllGradientNew, r=r, q=q, N=N, B=B, MSA=MSA, W=W, lambdaH=lambdaH, lambdaJ=lambdaJ, index_map=index_map),
            x0,
            jac=True,
            method='L-BFGS-B',
            options={
                'ftol': opt["epsconv"],
                'gtol': opt["epsconv"],
                'maxiter': opt["maxit"],
                'disp': False
            }
        )

        minf = result.fun
        minx = result.x
        minJ = minx[q:]
        compute.applyIsingGauge(minJ, q)
        # insert zeros for speeding up tensor mapping
        minJ = np.insert(minJ, r * q * q, np.zeros(q * q, dtype=minJ.dtype))
        pll[r] = minf
        J[:, r] = minJ
        print(f"Finished site {r+1}/{N} after {result.nit} iterations with pseudolikelihood {minf}")

    return J, pll


if __name__ == "__main__":
    opt = {
        "method": "L-BFGS-B",
        "epsconv": 1.0e-6,
        "maxit": 1000
    }
    # align = TextAlignment('/Users/simon/research/EVcouplings/test_run/couplings/output/mycsep_00000025/align_1/mycsep_00000025.a2m')
    align = TextAlignment('/Users/simon/research/EVcouplings/MSA_subset_evaluation/PDXH_ECOLI_1-218_b0.5.a2m')
    align.Filtering_MSA_Invalid()
    align.Filtering_MSA_Gap(50, 50)
    align.Downsample_Randomly(26809)
    MSA = align.Get_Numpy_Array(mapped=True)
    print(MSA.shape)

    B, N = MSA.shape
    q = int(np.max(MSA)+1)
    lambdaJ = 0.01
    lambdaH = 0.01
    start = time.time()
    if cuda.is_available():
        W, Beff = reweight_sequence_cuda(MSA, x=0.8)
    else:
        W, Beff = reweight_sequence(MSA)
    W /= Beff
    end = time.time()
    print(f'Time taken for reweighting: {end - start} seconds')

    # Run a small test
    start_time = time.time()
    tracemalloc.start()

    # Run minimization
    J, pll = minimize_pl_asym_threaded(opt, q, N, B, MSA, W, lambdaH, lambdaJ)
    # np.save('/Users/simon/Dev/CoevoFlash/Jmat.npy', J, allow_pickle=True)
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")

    tracemalloc.stop()
    end_time = time.time()
    print(f"Optimization took {end_time - start_time} seconds")

    start_time = time.time()
    score = compute_score(J)
    end_time = time.time()
    print(f"Score calculation took {end_time - start_time} seconds")
    # np.save('/Users/simon/Dev/CoevoFlash/Jmat.npy', score, allow_pickle=True)
