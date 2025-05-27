import numpy as np
import multiprocessing as mp
from scipy.optimize import minimize
import time
import tracemalloc
from functools import cache, partial
import compute
from alignment import TextAlignment
import torch

num_gpus = torch.cuda.device_count()

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

def reweight_sequence_cuda(MSA, x=0.8, tocpu=True):
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

def reweight_sequence_cuda_memory_efficient(MSA, x=0.8, tocpu=True):
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

def distribute_upper_triangle_work(B, num_gpus):
    total_elements = (B * (B - 1)) // 2
    elements_per_gpu = total_elements / num_gpus
    gpu_assignments = []
    current_elements = 0
    start_row = 0
    
    for b in range(B-1):
        current_elements += (B - b - 1)
        if current_elements >= elements_per_gpu:
            gpu_assignments.append((start_row, b))
            start_row = b + 1
            current_elements = 0
    gpu_assignments.append((start_row, B-1))
    
    return gpu_assignments

def compute_sim_count_on_gpu(rank, MSA, x, start, end, return_dict):
    device = torch.device(f'cuda:{rank}')
    MSA = MSA.to(device)
    B, N = MSA.shape
    identical_threshold = x * N

    m = torch.ones(end-start+1, device=device)

    for b in range(start, end+1):
        current_seq = MSA[b].unsqueeze(0).unsqueeze(0)
        remaining = MSA[b+1:].unsqueeze(0)
        identical_positions = (current_seq == remaining)
        identity_scores = identical_positions.sum(dim=-1).squeeze(0)
        similar_seqs = (identity_scores >= identical_threshold)
        m[b - start] += similar_seqs.sum()
        indices = torch.where(similar_seqs)[0] + b + 1
        m.index_add_(0, indices-start, torch.ones(len(indices), device=device))

    return_dict[rank] = m.cpu()

def reweight_sequence_multi_gpu(MSA, x=0.8, num_gpus=num_gpus):
    if isinstance(MSA, np.ndarray):
        MSA = torch.from_numpy(MSA)
    MSA = MSA.pin_memory()

    B = MSA.shape[0]
    ctx = mp.get_context('spawn')

    job_ranges = distribute_upper_triangle_work(B, num_gpus)
    manager = ctx.Manager()
    return_dict = manager.dict()
    processes = []

    for rank in range(num_gpus):
        p = ctx.Process(target=compute_sim_count_on_gpu, args=(rank, MSA, x, job_ranges[rank][0], job_ranges[rank][1], return_dict))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    all_counts = torch.cat([return_dict[i] for i in range(num_gpus)], dim=0)
    w = 1.0 / all_counts.float()
    Beff = w.sum()
    return w, Beff

def compute_score(J, q=21, gap_idx=20):
    # Average the J matrix out at main diagonal
    N = J.shape[1]
    J = J.flatten(order='F').reshape((N, N, q, q), order='C').transpose(1, 0, 2, 3)
    J = (J + J.swapaxes(0, 1)) / 2

    # Gap exclude Frobenius norm
    mask = np.ones((q, q), dtype=bool)
    mask[gap_idx, :] = False
    mask[:, gap_idx] = False
    J_filtered = J[:, :, mask].reshape(N, N, -1)
    FN = np.sqrt(np.sum(J_filtered ** 2, axis=-1))

    # Average Product Correction of Frobenius norm
    row_mean = FN.mean(axis=1, keepdims=True)
    col_mean = FN.mean(axis=0, keepdims=True)
    total_mean = row_mean.mean()
    FN_APC = FN - (row_mean @ col_mean) / total_mean

    return FN_APC

def minimize_pl_asym(opt, q, N, B, MSA, W, lambdaH, lambdaJ):
    nParamsh = q
    nParamsJ = (N - 1) * q * q
    nParams = nParamsh + nParamsJ
    x0 = np.zeros(nParams, dtype=np.float32)
    pll = np.zeros(N, dtype=np.float32)
    J = np.zeros((nParamsJ + q*q, N), dtype=np.float32)

    # In Python we'll iterate through sites sequentially
    # For parallel processing, you could use concurrent.futures or joblib
    for r in range(N):
        result = minimize(
            partial(compute.perSitePllGradient, r=r, q=q, N=N, B=B, MSA=MSA, W=W, lambdaH=lambdaH, lambdaJ=lambdaJ),
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
        print(f"Finished site {r+1}/{N} with pseudolikelihood {minf}")
    np.save('/Users/simon/Dev/CoevoFlash/Jmat_withoutIsing.npy', J, allow_pickle=True)
    return J, pll


if __name__ == "__main__":
    opt = {
        "method": "L-BFGS-B",
        "epsconv": 1.0e-6,
        "maxit": 1000
    }
    # align = TextAlignment('/Users/simon/research/EVcouplings/test_run/couplings/output/mycsep_00000025/align_1/mycsep_00000025.a2m')
    align = TextAlignment('/Users/simon/research/EVcouplings/MSA_subset_evaluation/PDXH_ECOLI_1-218_b0.5.a2m')
    MSA = align.Get_Numpy_Array(mapped=True)
    print(MSA.shape)

    B, N = MSA.shape
    q = int(np.max(MSA)+1)
    lambdaJ = 0.01
    lambdaH = 0.01
    start = time.time()
    if torch.cuda.is_available():
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
    J, pll = minimize_pl_asym(opt, q, N, B, MSA, W, lambdaH, lambdaJ)
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
    np.save('/Users/simon/Dev/CoevoFlash/Jmat.npy', score, allow_pickle=True)