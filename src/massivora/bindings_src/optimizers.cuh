#ifndef OPTIMIZERS_CUH
#define OPTIMIZERS_CUH

#include <cuda_fp16.h>
#include <cublas_v2.h>

#include <utility>
#include <cstring>

#include "culbfgsb.h"
#include "cuda_kernels.cuh"
#include "bindings.cuh"

/* =========================================================================
 * Optimizer – abstract base class
 * =========================================================================*/
struct Optimizer {
    virtual void step(__half* x_pad, const float* grad_pad) = 0;
    virtual ~Optimizer() = default;
};

/* =========================================================================
 * RMSprop / NAdam
 *
 * Device-side optimizer state structs mirroring the Python RMSprop / NAdam
 * All GPU buffers are stream-ordered
 * (cudaMallocAsync / cudaFreeAsync on the null stream).
 * =========================================================================*/
struct RMSprop : Optimizer {
    float* x_fp32;
    float* square_avg;
    float* grad_avg;
    float* momentum_buf;
    int    n_params;
    int    param_blks;
    float  lr, alpha, eps, weight_decay, momentum;
    int    centered;
    cudaStream_t stream_;

    RMSprop(const __half* x_pad_init, int n,
                float lr_, float alpha_, float eps_,
                float wd_,  float mom_,  int cent_,
                cudaStream_t stream = 0)
        : n_params(n), param_blks((n + 255) / 256),
          lr(lr_), alpha(alpha_), eps(eps_),
          weight_decay(wd_), momentum(mom_), centered(cent_),
          stream_(stream)
    {
        cudaMallocAsync(&x_fp32,       (size_t)n * sizeof(float), stream_);
        cudaMallocAsync(&square_avg,   (size_t)n * sizeof(float), stream_);
        cudaMallocAsync(&grad_avg,     (size_t)n * sizeof(float), stream_);
        cudaMallocAsync(&momentum_buf, (size_t)n * sizeof(float), stream_);
        /* initialise x_fp32 from the fp16 working buffer */
        half2float_copy<<<param_blks, 256, 0, stream_>>>(x_pad_init, x_fp32, n);
        /* zero running statistics */
        cudaMemsetAsync(square_avg,   0, (size_t)n * sizeof(float), stream_);
        cudaMemsetAsync(grad_avg,     0, (size_t)n * sizeof(float), stream_);
        cudaMemsetAsync(momentum_buf, 0, (size_t)n * sizeof(float), stream_);
    }

    /* One optimizer step: update x_fp32 then write fp16 back to x_pad */
    void step(__half* x_pad, const float* grad_pad) override {
        RMSprop_step<<<param_blks, 256, 0, stream_>>>(
            x_fp32, grad_pad,
            square_avg, grad_avg, momentum_buf,
            n_params, lr, alpha, eps, weight_decay, momentum, centered
        );
        float2half_copy<<<param_blks, 256, 0, stream_>>>(x_fp32, x_pad, n_params);
    }

    ~RMSprop() {
        cudaFreeAsync(x_fp32,       stream_);
        cudaFreeAsync(square_avg,   stream_);
        cudaFreeAsync(grad_avg,     stream_);
        cudaFreeAsync(momentum_buf, stream_);
    }
};

struct NAdam : Optimizer {
    float* x_fp32;
    float* exp_avg;
    float* exp_avg_sq;
    float* mu_product;       /* scalar (1,) – product of all μ_t so far */
    int    n_params;
    int    param_blks;
    int    step_counter;
    float  lr, beta1, beta2, eps, weight_decay, momentum_decay;
    cudaStream_t stream_;

    NAdam(const __half* x_pad_init, int n,
              float lr_, float b1_, float b2_, float eps_,
              float wd_,  float md_,
              cudaStream_t stream = 0)
        : n_params(n), param_blks((n + 255) / 256),
          step_counter(0),
          lr(lr_), beta1(b1_), beta2(b2_), eps(eps_),
          weight_decay(wd_), momentum_decay(md_),
          stream_(stream)
    {
        cudaMallocAsync(&x_fp32,     (size_t)n * sizeof(float), stream_);
        cudaMallocAsync(&exp_avg,    (size_t)n * sizeof(float), stream_);
        cudaMallocAsync(&exp_avg_sq, (size_t)n * sizeof(float), stream_);
        cudaMallocAsync(&mu_product, sizeof(float),             stream_);
        half2float_copy<<<param_blks, 256, 0, stream_>>>(x_pad_init, x_fp32, n);
        cudaMemsetAsync(exp_avg,    0, (size_t)n * sizeof(float), stream_);
        cudaMemsetAsync(exp_avg_sq, 0, (size_t)n * sizeof(float), stream_);
        fill_scalar_float<<<1, 1, 0, stream_>>>(mu_product, 1.0f);  /* mu_product = 1 */
    }

    void step(__half* x_pad, const float* grad_pad) override {
        ++step_counter;
        NAdam_step<<<param_blks, 256, 0, stream_>>>(
            x_fp32, grad_pad,
            exp_avg, exp_avg_sq, mu_product,
            step_counter, n_params,
            lr, beta1, beta2, eps, weight_decay, momentum_decay
        );
        float2half_copy<<<param_blks, 256, 0, stream_>>>(x_fp32, x_pad, n_params);
    }

    ~NAdam() {
        cudaFreeAsync(x_fp32,     stream_);
        cudaFreeAsync(exp_avg,    stream_);
        cudaFreeAsync(exp_avg_sq, stream_);
        cudaFreeAsync(mu_product, stream_);
    }
};

/* =========================================================================
 * LBFGS – L-BFGS-B optimizer backed by the culbfgsb library.
 *
 * Unlike RMSprop / NAdam this optimizer controls its own iteration loop
 * (including line search) so it does NOT inherit from Optimizer.
 * Call optimize() once; it returns {num_iterations, final_pll}.
 * =========================================================================*/
struct LBFGS {
    /* ---------- problem geometry ---------- */
    int n_params, param_blks;
    int B, N, q, q_pad, r;
    float lambdaH, lambdaJ;

    /* ---------- borrowed device pointers (caller owns lifetime) ---------- */
    __half*             x_pad;          /* (n_params,)   fp16 in/out          */
    const __half*       MSA_pad_flat;   /* (B, N*q_pad)  fp16                 */
    const signed char*  MSA;            /* (B, N)        int8                 */
    const float*        W;              /* (B,)          fp32                 */
    float*              pll_out;        /* (4,)          fp32 scratch         */
    __half*             vgrad_pad;      /* (B, q_pad)    fp16 scratch         */
    __half*             energies_fp16;  /* (B, q_pad)    fp16 scratch         */
    float*              energies_fp32;  /* (B, q_pad)    fp32 scratch         */
    cublasHandle_t      handle;

    /* ---------- L-BFGS-B hyperparameters ---------- */
    int   m_corr;     /* number of limited-memory corrections (3–8)  */
    float eps_f, eps_g, eps_x;
    int   max_iteration;

    LBFGS(__half*             x_pad_,
          const __half*       MSA_pad_flat_,
          const signed char*  MSA_,
          const float*        W_,
          float* pll_out_, __half* vgrad_pad_,
          __half* energies_fp16_, float* energies_fp32_,
          cublasHandle_t      handle_,
          int n, int B_, int N_, int q_, int q_pad_, int r_,
          float lambdaH_, float lambdaJ_,
          int m_corr_, float eps_f_, float eps_g_, float eps_x_,
          int max_iter_)
        : n_params(n), param_blks((n + 255) / 256),
          B(B_), N(N_), q(q_), q_pad(q_pad_), r(r_),
          lambdaH(lambdaH_), lambdaJ(lambdaJ_),
          x_pad(x_pad_), MSA_pad_flat(MSA_pad_flat_),
          MSA(MSA_), W(W_),
          pll_out(pll_out_), vgrad_pad(vgrad_pad_),
          energies_fp16(energies_fp16_), energies_fp32(energies_fp32_),
          handle(handle_),
          m_corr(m_corr_), eps_f(eps_f_), eps_g(eps_g_), eps_x(eps_x_),
          max_iteration(max_iter_) {}

    std::pair<int, float> optimize()
    {
        /* ---- Allocate fp32 parameter copy ---- */
        float* x_fp32;
        cudaMalloc(&x_fp32, (size_t)n_params * sizeof(float));
        half2float_copy<<<param_blks, 256>>>(x_pad, x_fp32, n_params);

        /* ---- Bounds arrays (all unconstrained) ---- */
        int*   nbd;
        float* lb;
        float* ub;
        cudaMalloc(&nbd, (size_t)n_params * sizeof(int));
        cudaMalloc(&lb,  (size_t)n_params * sizeof(float));
        cudaMalloc(&ub,  (size_t)n_params * sizeof(float));
        cudaMemset(nbd, 0, (size_t)n_params * sizeof(int));
        cudaMemset(lb,  0, (size_t)n_params * sizeof(float));
        cudaMemset(ub,  0, (size_t)n_params * sizeof(float));

        /* ---- L-BFGS-B options ---- */
        LBFGSB_CUDA_OPTION<float> options;
        lbfgsbcuda::lbfgsbdefaultoption(options);
        options.mode           = LCM_CUDA;
        options.max_iteration  = max_iteration;
        options.eps_f          = eps_f;
        options.eps_g          = eps_g;
        options.eps_x          = eps_x;
        options.hessian_approximate_dimension = m_corr;

        /* ---- State + callback ---- */
        LBFGSB_CUDA_STATE<float> state;
        std::memset(&state, 0, sizeof(state));
        state.m_cublas_handle = handle;

        float final_f = 0.0f;

        /*  The callback evaluates f(x) and ∇f(x) on the GPU.
         *  x and g are DEVICE pointers maintained by the library.
         *  f is a HOST reference.                                           */
        state.m_funcgrad_callback =
            [this, &final_f](float* x, float& f, float* g,
                             const cudaStream_t& st, // The library DOES NOT support custom streams!!
                             const LBFGSB_CUDA_SUMMARY<float>& /*summary*/) -> int
        {
            cudaStream_t s = st;  /* may be NULL (default stream) */

            /* 1. fp32 x → fp16 x_pad for GEMM computations */
            float2half_copy<<<param_blks, 256, 0, s>>>(x, x_pad, n_params);

            /* 2. Broadcast h_r into initial energies (beta=1 in GEMM1) */
            broadcast_hr_to_energies<<<(B * q_pad + 255) / 256, 256, 0, s>>>(
                x_pad, energies_fp32, q_pad, B * q_pad);

            /* 3. Compute PLL objective + gradient.
             *    Pass the library's g as grad_pad so the gradient is
             *    written directly where L-BFGS-B expects it.              */
            cublasSetStream(handle, s);
            cudaFillPllGradients(
                MSA_pad_flat, x_pad, MSA, W,
                r, B, N, q, q_pad, lambdaH, lambdaJ,
                pll_out,
                g,              /* gradient output → library buffer */
                vgrad_pad,
                energies_fp32,
                handle, s);

            /* 4. Copy pll_out[1] → host f (synchronous) */
            cudaDeviceSynchronize();
            cudaMemcpy(&f, pll_out + 1, sizeof(float), cudaMemcpyDeviceToHost);
            final_f = f;
            return 0;
        };

        /* ---- Run L-BFGS-B minimisation ---- */
        LBFGSB_CUDA_SUMMARY<float> summary;
        std::memset(&summary, 0, sizeof(summary));

        lbfgsbcuda::lbfgsbminimize<float>(
            n_params, state, options, x_fp32, nbd, lb, ub, summary);

        /* ---- Convert final fp32 solution back to fp16 x_pad ---- */
        float2half_copy<<<param_blks, 256>>>(x_fp32, x_pad, n_params);
        cudaDeviceSynchronize();

        /* ---- Cleanup ---- */
        cudaFree(x_fp32);
        cudaFree(nbd);
        cudaFree(lb);
        cudaFree(ub);

        return {summary.num_iteration, final_f};
    }
};

#endif
