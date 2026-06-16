#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <cmath>
#include <functional>
#include <nvtx3/nvToolsExt.h>
#include "optimizers.cuh"
#include "cuda_kernels.cuh"


extern "C" void cudaFillPllGradients(
    const __half*      MSA_pad_flat,  /* (B, N*q_pad)             fp16 device ptr */
    const __half*      x_pad,         /* (q_pad + N*q_pad*q_pad,) fp16 device ptr – x0_pad[r] */
    const signed char* MSA,           /* (B, N)                   int8 device ptr */
    const float*       W,             /* (B,)                     fp32 device ptr */
    int r, int B, int N, int q, int q_pad,
    float lambdaH, float lambdaJ,
    float* pll_out,                   /* (4,)                     fp32 – [0]=prev kept, [1:] zeroed */
    float* grad_pad,                  /* (q_pad + N*q_pad*q_pad,) fp32 – zeroed internally */
    __half* vgrad_pad,                /* (B, q_pad)               fp16 – scratch, zeroed internally */
    float*  energies_fp32,            /* (B, q_pad)               fp32 – scratch, caller-allocated */
    cublasHandle_t ext_handle,        /* pass nullptr to create internally */
    cudaStream_t stream = 0           /* CUDA stream (default: null stream) */
)
{
    const int Jr_pad_total_params = N * q_pad * q_pad;
    const int x_pad_total         = q_pad + Jr_pad_total_params;
    const int sumsquare_blocks    = (Jr_pad_total_params + 255) / 256;
    const int nb                  = 10;
    const dim3 threads_vgrad(q_pad, nb);
    const dim3 blocks_vgrad((B + nb - 1) / nb);

    /* ------------------------------------------------------------------
     * Split flat buffers into h and J views (no copy – pointer arithmetic)
     * ----------------------------------------------------------------*/
    const __half* hr_pad     = x_pad;           /* (q_pad,)          fp16 */
    const __half* Jr_pad     = x_pad + q_pad;   /* (N*q_pad, q_pad)  fp16 */
    float*        grad_hr_pad = grad_pad;        /* (q_pad,)          fp32 */
    float*        grad_Jr_pad = grad_pad + q_pad; /* (N*q_pad, q_pad) fp32 */

    /* ------------------------------------------------------------------
     * Zero:  pll_out[1:3],  grad_pad[:],  vgrad_pad[:]
     * Matches Python:  pll_out[1:]=0; grad_hr_pad[:]=0; grad_Jr_pad[:]=0; vgrad_pad[:]=0
     * ----------------------------------------------------------------*/
    cudaMemsetAsync(pll_out + 1,  0, 3 * sizeof(float),                 stream);
    cudaMemsetAsync(grad_pad,     0, (size_t)x_pad_total * sizeof(float), stream);
    cudaMemsetAsync(vgrad_pad,    0, (size_t)B * q_pad   * sizeof(__half), stream);

    /* ------------------------------------------------------------------
     * cuBLAS handle
     * ----------------------------------------------------------------*/
    cublasHandle_t handle;
    bool own_handle = (ext_handle == nullptr);
    if (own_handle)
        cublasCreate(&handle);
    else
        handle = ext_handle;
    cublasSetStream(handle, stream);

    const float  alpha32 = 1.0f;
    const float  beta32  = 0.0f;

    /* ==================================================================
     * GEMM 1: energies_fp32 (B, q_pad) = MSA_pad_flat (B, N*q_pad)
     *                                   @ Jr_pad      (N*q_pad, q_pad)
     *
     * Row-major C = A @ B  <==>  col-major C^T = B^T @ A^T
     *   m=q_pad, n=B, k=N*q_pad
     *   A_col = Jr_pad        (q_pad × N*q_pad,  lda=q_pad,    OP_N)
     *   B_col = MSA_pad_flat  (N*q_pad × B,      ldb=N*q_pad,  OP_N)
     *   C_col = energies_fp32 (q_pad × B,        ldc=q_pad)
     * ================================================================*/
     cublasGemmEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        q_pad, B, N * q_pad,                            // m, n, k
        &alpha32, Jr_pad, CUDA_R_16F, q_pad,            // A
        MSA_pad_flat, CUDA_R_16F, N * q_pad, &alpha32,  // B
        energies_fp32,  CUDA_R_32F, q_pad,              // C
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT);

    /* ==================================================================
     * fused_vgrad kernel
     * ================================================================*/
    fused_vgrad<<<blocks_vgrad, threads_vgrad, 0, stream>>>(
        energies_fp32, MSA, W,
        r, B, N, q, q_pad,
        pll_out + 1,    /* pll_out[1] */
        grad_hr_pad,
        vgrad_pad
    );

    /* ==================================================================
     * GEMM 2: grad_Jr_pad (N*q_pad, q_pad) fp32
     *       = MSA_pad_flat.T (N*q_pad, B) fp16
     *       @ vgrad_pad      (B, q_pad)   fp16
     *
     * Row-major C = A^T @ B  <==>  col-major C^T = B^T @ A
     *   m=q_pad, n=N*q_pad, k=B
     *   A_col = vgrad_pad    (q_pad  × B,       lda=q_pad,      OP_N)
     *   B_col = MSA_pad_flat (N*q_pad × B,      ldb=N*q_pad,    OP_T → B × N*q_pad)
     *   C_col = grad_Jr_pad  (q_pad × N*q_pad,  ldc=q_pad)   fp32
     * ================================================================*/
    cublasGemmEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        q_pad, N * q_pad, B,                            // m, n, k
        &alpha32, vgrad_pad, CUDA_R_16F, q_pad,         // A
        MSA_pad_flat, CUDA_R_16F, N * q_pad, &beta32,   // B
        grad_Jr_pad,  CUDA_R_32F, q_pad,                // C
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT);

    if (own_handle)
        cublasDestroy(handle);

    /* ==================================================================
     * grad_l2:  (N,) blocks × (q_pad, q_pad) threads
     * ================================================================*/
    grad_l2<<<N, dim3(q_pad, q_pad), 0, stream>>>(
        grad_hr_pad, grad_Jr_pad,
        hr_pad, Jr_pad,
        N, q, q_pad, lambdaH, lambdaJ
    );

    /* ==================================================================
     * sum_squares kernel
     * ================================================================*/
    sum_squares<<<sumsquare_blocks, 256, 0, stream>>>(
        hr_pad, Jr_pad, pll_out,
        q, q_pad, Jr_pad_total_params
    );

    /* ==================================================================
     * pll_l2:  pll_out[1] += lambdaH * pll_out[2] + 0.5f * lambdaJ * pll_out[3]
     * ================================================================*/
    pll_l2<<<1, 1, 0, stream>>>(pll_out, lambdaH, lambdaJ);
}


std::pair<int, float> cudaOptimizeSite(
    const __half*      MSA_pad,       /* (B, N, q_pad) fp16 – full one-hot */
    const signed char* MSA,           /* (B, N)        int8               */
    const float*       W,             /* (B,)         fp32               */
    int r, int B, int N, int q, int q_pad,
    float lambdaH, float lambdaJ,
    float eps_conv, int maxeval,
    __half* x_pad,                    /* (n_params,)  fp16 in/out        */
    const std::map<std::string, float>& hyperparams,
    cublasHandle_t ext_handle,
    cudaStream_t stream = 0           /* CUDA stream; non-zero enables multi-stream concurrency */
)
{
    auto get = [&](const char* k, float def) -> float {
        auto it = hyperparams.find(k);
        return (it != hyperparams.end()) ? it->second : def;
    };

    const int optimizer_type  = static_cast<int>(get("optimizer", 0.0f));
    const int n_params        = q_pad + N * q_pad * q_pad;
    const size_t msa_flat_bytes = (size_t)B * N * q_pad * sizeof(__half);

    /* ---- Allocate shared scratch buffers asynchronously (stream-ordered) ---- */
    __half* MSA_pad_flat;     /* (B, N*q_pad) fp16 – internal copy with col r zeroed */
    float*  pll_out;
    float*  grad_pad;
    __half* vgrad_pad;
    __half* energies_fp16;
    float*  energies_fp32;

    cudaMallocAsync(&MSA_pad_flat,  msa_flat_bytes,                           stream);
    cudaMallocAsync(&pll_out,       4 * sizeof(float),                        stream);
    cudaMallocAsync(&grad_pad,      (size_t)n_params * sizeof(float),         stream);
    cudaMallocAsync(&vgrad_pad,     (size_t)B * q_pad * sizeof(__half),       stream);
    cudaMallocAsync(&energies_fp16, (size_t)B * q_pad * sizeof(__half),       stream);
    cudaMallocAsync(&energies_fp32, (size_t)B * q_pad * sizeof(float),        stream);
    /* Copy full MSA_pad into MSA_pad_flat, then zero column r.
     * fp16(0.0) == 0x0000, so cudaMemset2DAsync works correctly. */
    cudaMemcpyAsync(MSA_pad_flat, MSA_pad, msa_flat_bytes, cudaMemcpyDeviceToDevice, stream);
    cudaMemset2DAsync(
        MSA_pad_flat + (size_t)r * q_pad,      /* ptr to col-r block in row 0 */
        (size_t)N * q_pad * sizeof(__half),    /* pitch (bytes per row)        */
        0,                                     /* value                        */
        (size_t)q_pad * sizeof(__half),        /* width (bytes to zero per row)*/
        (size_t)B,                             /* height (number of rows)      */
        stream
    );
    cudaMemsetAsync(pll_out, 0, 4 * sizeof(float), stream);

    /* ---- Pinned host buffer for fast convergence check (avoids page-fault stall) ---- */
    float* pll_host;
    cudaHostAlloc(&pll_host, sizeof(float), cudaHostAllocDefault);
    *pll_host = 0.0f;

    /* ---- Create cuBLAS handle; always bind to our stream ---- */
    cublasHandle_t handle;
    bool own_handle = (ext_handle == nullptr);
    if (own_handle) cublasCreate(&handle);
    else            handle = ext_handle;
    cublasSetStream(handle, stream);

    float pll_prev    = 1e10f;
    float pll_current = 0.0f;
    int   iter        = 0;

    /* ============================================================
    * First-order optimizer path (RMSprop / NAdam)
    * ============================================================*/
    // std::unique_ptr<Optimizer> opt;
    // if (optimizer_type == 1) {
    //     opt = std::make_unique<NAdam>(
    //         x_pad, n_params,
    //         get("lr",            0.005f),
    //         get("beta1",         0.9f),
    //         get("beta2",         0.999f),
    //         get("eps",           1e-8f),
    //         get("weight_decay",  0.0f),
    //         get("momentum_decay",4e-3f),
    //         stream
    //     );
    // } else {
    //     opt = std::make_unique<RMSprop>(
    //         x_pad, n_params,
    //         get("lr",           0.002f),
    //         get("alpha",        0.92f),
    //         get("eps",          1e-8f),
    //         get("weight_decay", 0.0f),
    //         get("momentum",     0.9f),
    //         static_cast<int>(get("centered", 0.0f)),
    //         stream
    //     );
    // }

    // for (iter = 0; iter < maxeval; ++iter) {
    //     broadcast_hr_to_energies<<<(B*q_pad+255)/256, 256, 0, stream>>>(x_pad, energies_fp32, q_pad, B*q_pad);
    //     cudaFillPllGradients(
    //         MSA_pad_flat, x_pad, MSA, W,
    //         r, B, N, q, q_pad, lambdaH, lambdaJ,
    //         pll_out, grad_pad, vgrad_pad,
    //         energies_fp32, handle, stream);
    //     opt->step(x_pad, grad_pad);
    //     cudaMemcpyAsync(pll_host, pll_out + 1, sizeof(float),
    //                     cudaMemcpyDeviceToHost, stream);
    //     cudaStreamSynchronize(stream);
    //     pll_current = *pll_host;
    //     if (fabsf(pll_current - pll_prev) < eps_conv) { ++iter; break; }
    //     pll_prev = pll_current;
    // }
    // /* unique_ptr destructor frees optimizer GPU buffers (stream-ordered) */
    // opt.reset();

    LBFGS lbfgs(
        x_pad, MSA_pad_flat, MSA, W,
        pll_out, vgrad_pad, energies_fp16, energies_fp32,
        handle,
        n_params, B, N, q, q_pad, r,
        lambdaH, lambdaJ,
        static_cast<int>(get("m_corr",   8.0f)),
        get("eps_f", 3e-4f),
        get("eps_g", 3e-4f),
        get("eps_x", 3e-4f),
        maxeval);

    auto result = lbfgs.optimize();
    iter        += result.first;
    pll_current = result.second;

    if (own_handle) cublasDestroy(handle);
    cudaFreeHost(pll_host);

    /* ---- Free shared scratch buffers (stream-ordered) ---- */
    cudaFreeAsync(MSA_pad_flat,  stream);
    cudaFreeAsync(pll_out,       stream);
    cudaFreeAsync(grad_pad,      stream);
    cudaFreeAsync(vgrad_pad,     stream);
    cudaFreeAsync(energies_fp16, stream);
    cudaFreeAsync(energies_fp32, stream);

    return {iter, pll_current};
}
