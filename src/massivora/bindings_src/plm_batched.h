#ifndef PLM_BATCHED_H
#define PLM_BATCHED_H

#include <cstddef>

/* Optimiser counters of one pair (plm::cuda::Stats, C-compatible). */
struct PlmBatchStats {
    int    slots;       /* sites optimised concurrently                         */
    long   rounds;      /* batched evaluations (one GEMM pair each)             */
    long   evals;       /* site evaluations, summed over rounds                 */
    long   iterations;  /* accepted L-BFGS steps, summed over sites             */
    int    converged;   /* stopped on ‖∇f‖∞ ≤ eps_g                             */
    int    stalled;     /* stopped by the flat-f safety net                     */
    int    max_iter;    /* stopped at maxeval                                   */
    int    ls_failed;   /* line search found no sufficient decrease             */
    double f_sum;       /* final objective, summed over sites                   */
};

/* One pair through a plmDCA Session (plm_dca_cuda.cuh) on the current device,
 * for plm_opt_site: the batched multi-site L-BFGS, up to `site_batch` sites in
 * lockstep, fitted to the device memory free now since nothing else here
 * admits pairs.
 *
 *   MSA    (B, N) int8 host, states 0 .. q-1
 *   W      (B,)   float64 host, already divided by Beff
 *   J_out  (N, N*q*q) float32 host: row r = site r's couplings J[i, a, c] at
 *          (i*q + a)*q + c, raw (no gauge), rounded to fp16 like the per-site
 *          path's output. Every site starts from zero.
 *
 * Returns 0 on success; otherwise 2 for out of device memory, 3 for a lost
 * CUDA context, 1 for anything else, with the message in err. */
extern "C" int plmCouplingsBatched(
    const signed char* MSA, const double* W,
    int B, int N, int q,
    float lambdaH, float lambdaJ, int maxeval, int site_batch,
    float* J_out, PlmBatchStats* stats, char* err, size_t err_len);

#endif
