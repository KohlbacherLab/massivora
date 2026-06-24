#ifndef OPTIMIZERS_CUH
#define OPTIMIZERS_CUH

#include <cuda_fp16.h>
#include <cublas_v2.h>

#include <cmath>
#include <utility>

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
 * Scalar helper kernels for the unconstrained L-BFGS.
 *
 * The two-loop recursion runs with CUBLAS_POINTER_MODE_DEVICE so that every
 * cublasSdot / cublasSaxpy keeps its scalar operand on the device and never
 * forces a host synchronisation (the multi-stream concurrency killer called
 * out in the design notes). These single-thread kernels do the tiny scalar
 * arithmetic between BLAS calls, all stream-ordered, all on the device.
 * =========================================================================*/
__global__ void lbfgs_set_scalar(float* p, float v) {
    if (blockIdx.x == 0 && threadIdx.x == 0) *p = v;
}
__global__ void lbfgs_mul(float* out, const float* a, const float* b) {
    if (blockIdx.x == 0 && threadIdx.x == 0) *out = (*a) * (*b);
}
__global__ void lbfgs_neg(float* out, const float* a) {
    if (blockIdx.x == 0 && threadIdx.x == 0) *out = -(*a);
}
__global__ void lbfgs_sub(float* out, const float* a, const float* b) {
    if (blockIdx.x == 0 && threadIdx.x == 0) *out = (*a) - (*b);
}
/* out = |v[idx-1]| — read the max-magnitude element cublasIsamax pointed at
 * (Isamax returns a 1-based index). Used to form ‖v‖∞ on the device. */
__global__ void lbfgs_absval_at(float* out, const float* v, const int* idx) {
    if (blockIdx.x == 0 && threadIdx.x == 0) *out = fabsf(v[(*idx) - 1]);
}
/* rho = 1 / (y·s) with a positive-curvature guard. A non-positive or tiny
 * y·s (fp16 noise can produce one even though the PLL+L2 objective is convex)
 * yields rho = 0, which makes that history pair contribute nothing to the
 * two-loop recursion instead of corrupting the search direction. */
__global__ void lbfgs_rho(float* rho, const float* sy) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const float v = *sy;
        *rho = (v > 1e-12f) ? (1.0f / v) : 0.0f;
    }
}
/* gamma = (s·y)/(y·y) – the initial inverse-Hessian scaling. Clamped to 1
 * when the most-recent curvature is non-positive. */
__global__ void lbfgs_gamma(float* gamma, const float* sy, const float* yy) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const float s = *sy;
        const float y = *yy;
        *gamma = (s > 1e-12f && y > 1e-30f) ? (s / y) : 1.0f;
    }
}

/* =========================================================================
 * LBFGS_Unconstrained – limited-memory BFGS without bound constraints.
 *
 * Everything runs on the single CUDA stream handed in by the caller, so many
 * sites can optimise concurrently on distinct streams. The s/y correction
 * history lives device-resident in a fixed m_corr-slot ring buffer.
 * =========================================================================*/
struct LBFGS_Unconstrained {
    /* ---------- problem geometry ---------- */
    int n_params, param_blks;
    int B, N, q, q_pad, r;
    float lambdaH, lambdaJ;

    /* ---------- borrowed device pointers (caller owns lifetime) ----------  */
    __half*             x_pad;          /* (n_params,)   fp16 in/out          */
    const __half*       MSA_pad_flat;   /* (B, N*q_pad)  fp16                 */
    const signed char*  MSA;            /* (B, N)        int8                 */
    const float*        W;              /* (B,)          fp32                 */
    float*              pll_out;        /* (4,)          fp32 scratch         */
    float*              grad_pad;       /* (n_params,)   fp32 scratch (∇f)    */
    __half*             vgrad_pad;      /* (B, q_pad)    fp16 scratch         */
    float*              energies_fp32;  /* (B, q_pad)    fp32 scratch         */
    cublasHandle_t      handle;
    cudaStream_t        stream_;

    /* ---------- L-BFGS hyperparameters ----------
     * eps_g: PRIMARY convergence tolerance — stop when ‖∇f‖∞ ≤ eps_g. The
     *        infinity norm is size-invariant, so a single calibrated value
     *        transfers across protein pairs of any N (unlike ‖·‖₂, which grows
     *        with √n_params). This is the knob to tune.
     * eps_f: relative f-change tolerance for the STALL safety net only (see
     *        kStallPatience). Kept tight so it never pre-empts the eps_g test;
     *        it just stops a site that has genuinely flat-lined (e.g. at the
     *        fp16 noise floor) instead of spinning to max_iteration.
     * eps_x: line-search step-length floor. */
    int   m_corr;     /* number of limited-memory corrections (typ. 4–8) */
    float eps_f, eps_g, eps_x;
    int   max_iteration;

    /* ---------- line-search / convergence constants ---------- */
    static constexpr float kArmijoC1    = 1e-4f; /* sufficient-decrease coeff     */
    static constexpr float kBacktrack   = 0.5f;  /* step shrink factor            */
    static constexpr int   kMaxLineEval = 10;    /* max f-evals per step          */
    static constexpr int   kStallPatience = 5;   /* consecutive flat steps → stop */

    /* ---------- owned device buffers (allocated in optimize()) ----------   */
    float* x_fp32   = nullptr;   /* (n_params,)            current accepted x */
    float* x_trial  = nullptr;   /* (n_params,)            line-search trial  */
    float* g_cur    = nullptr;   /* (n_params,)            ∇f at accepted x   */
    float* dir      = nullptr;   /* (n_params,)            search direction   */
    float* s_buf    = nullptr;   /* (m_corr, n_params)     ring of x steps    */
    float* y_buf    = nullptr;   /* (m_corr, n_params)     ring of grad steps */
    float* d_ws     = nullptr;   /* (2*m_corr + WS_EXTRA,) device scalars     */
    int*   d_imax   = nullptr;   /* (1,) cublasIsamax index for ‖∇f‖∞         */

    /* device-scalar workspace layout: [0,m_corr)=rho, [m_corr,2m_corr)=alpha,
     * then the named temporaries below. */
    enum { WS_DOT = 0, WS_SY, WS_YY, WS_GAMMA, WS_BETA, WS_COEF,
           WS_NEG, WS_GTD, WS_GG, WS_GINF, WS_STEP, WS_NEG_ONE, WS_EXTRA };

    /* ring-buffer state (host-side indices only; data is on device) */
    int head = 0, count = 0;

    LBFGS_Unconstrained(__half*             x_pad_,
          const __half*       MSA_pad_flat_,
          const signed char*  MSA_,
          const float*        W_,
          float* pll_out_, float* grad_pad_,
          __half* vgrad_pad_, float* energies_fp32_,
          cublasHandle_t      handle_,
          int n, int B_, int N_, int q_, int q_pad_, int r_,
          float lambdaH_, float lambdaJ_,
          int m_corr_, float eps_f_, float eps_g_, float eps_x_,
          int max_iter_, cudaStream_t stream)
        : n_params(n), param_blks((n + 255) / 256),
          B(B_), N(N_), q(q_), q_pad(q_pad_), r(r_),
          lambdaH(lambdaH_), lambdaJ(lambdaJ_),
          x_pad(x_pad_), MSA_pad_flat(MSA_pad_flat_),
          MSA(MSA_), W(W_),
          pll_out(pll_out_), grad_pad(grad_pad_),
          vgrad_pad(vgrad_pad_), energies_fp32(energies_fp32_),
          handle(handle_), stream_(stream),
          m_corr(m_corr_ > 0 ? m_corr_ : 1),
          eps_f(eps_f_), eps_g(eps_g_), eps_x(eps_x_),
          max_iteration(max_iter_) {}

    /* ---- ring-buffer slot helpers (newest first / oldest first) ---- */
    int slot_newest(int j) const { return (head - 1 - j + 2 * m_corr) % m_corr; }
    int slot_oldest(int j) const { return (head - count + j + 2 * m_corr) % m_corr; }

    /* ---- device-scalar workspace accessors ----
     * Layout in d_ws: [0, m_corr) = rho, [m_corr, 2*m_corr) = alpha,
     * [2*m_corr, 2*m_corr + WS_EXTRA) = the named temporaries below. */
    float* RHO(int s)        { return d_ws + s; }
    float* ALPHA(int s)      { return d_ws + m_corr + s; }
    float* T(int which)      { return d_ws + 2 * m_corr + which; }

    /* ---- pull a single device scalar back to the host (one stream sync) --- */
    float read_scalar(const float* dptr) {
        float h;
        cudaMemcpyAsync(&h, dptr, sizeof(float), cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);
        return h;
    }

    /* ---- ‖g‖∞ via cublasIsamax (DEVICE pointer mode) + a 1-thread fetch.
     * Requires the handle to be in CUBLAS_POINTER_MODE_DEVICE (the optimizer's
     * default outside evaluate()). One stream sync to bring the scalar back. */
    float g_inf_norm(const float* g) {
        cublasIsamax(handle, n_params, g, 1, d_imax);     /* 1-based idx → device */
        lbfgs_absval_at<<<1, 1, 0, stream_>>>(T(WS_GINF), g, d_imax);
        return read_scalar(T(WS_GINF));
    }

    /* =====================================================================
     * evaluate – f(x_in) and ∇f(x_in).
     *   Leaves f in pll_out[1] and ∇f in grad_pad (both device-resident).
     *   x_in is an fp32 device buffer; it is rounded to the fp16 x_pad the
     *   GEMMs consume. cudaFillPllGradients needs host-pointer cuBLAS scalars,
     *   so we drop to HOST pointer mode for the call and restore DEVICE after.
     * ===================================================================*/
    void evaluate(const float* x_in) {
        float2half_copy<<<param_blks, 256, 0, stream_>>>(x_in, x_pad, n_params);
        broadcast_hr_to_energies<<<(B * q_pad + 255) / 256, 256, 0, stream_>>>(
            x_pad, energies_fp32, q_pad, B * q_pad);

        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
        cublasSetStream(handle, stream_);
        cudaFillPllGradients(
            MSA_pad_flat, x_pad, MSA, W,
            r, B, N, q, q_pad, lambdaH, lambdaJ,
            pll_out, grad_pad, vgrad_pad, energies_fp32,
            handle, stream_);
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
    }

    /* =====================================================================
     * two_loop_recursion – dir = -H·g_cur (the L-BFGS search direction).
     *   Runs entirely in DEVICE pointer mode: no per-dot host syncs.
     * ===================================================================*/
    void two_loop_recursion() {
        /* dir <- g_cur */
        cudaMemcpyAsync(dir, g_cur, (size_t)n_params * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream_);

        if (count == 0) {
            /* no history: steepest descent, dir = -g */
            cublasSscal(handle, n_params, T(WS_NEG_ONE), dir, 1);
            return;
        }

        /* first loop: newest -> oldest */
        for (int j = 0; j < count; ++j) {
            const int s = slot_newest(j);
            const float* sv = s_buf + (size_t)s * n_params;
            const float* yv = y_buf + (size_t)s * n_params;
            cublasSdot(handle, n_params, sv, 1, dir, 1, T(WS_DOT));      /* s·q   */
            lbfgs_mul<<<1, 1, 0, stream_>>>(ALPHA(s), RHO(s), T(WS_DOT));/* alpha */
            lbfgs_neg<<<1, 1, 0, stream_>>>(T(WS_NEG), ALPHA(s));
            cublasSaxpy(handle, n_params, T(WS_NEG), yv, 1, dir, 1);     /* q-=αy */
        }

        /* initial Hessian scaling from the most-recent pair: H0 = γ·I,
         * γ = (sᵀy)/(yᵀy). */
        {
            const int s = slot_newest(0);
            const float* sv = s_buf + (size_t)s * n_params;
            const float* yv = y_buf + (size_t)s * n_params;
            cublasSdot(handle, n_params, sv, 1, yv, 1, T(WS_SY));
            cublasSdot(handle, n_params, yv, 1, yv, 1, T(WS_YY));
            lbfgs_gamma<<<1, 1, 0, stream_>>>(T(WS_GAMMA), T(WS_SY), T(WS_YY));
            cublasSscal(handle, n_params, T(WS_GAMMA), dir, 1);
        }

        /* second loop: oldest -> newest */
        for (int j = 0; j < count; ++j) {
            const int s = slot_oldest(j);
            const float* sv = s_buf + (size_t)s * n_params;
            const float* yv = y_buf + (size_t)s * n_params;
            cublasSdot(handle, n_params, yv, 1, dir, 1, T(WS_DOT));      /* y·r  */
            lbfgs_mul<<<1, 1, 0, stream_>>>(T(WS_BETA), RHO(s), T(WS_DOT));/* beta */
            lbfgs_sub<<<1, 1, 0, stream_>>>(T(WS_COEF), ALPHA(s), T(WS_BETA));
            cublasSaxpy(handle, n_params, T(WS_COEF), sv, 1, dir, 1);    /* r+= */
        }

        /* dir = -r  (descent direction) */
        cublasSscal(handle, n_params, T(WS_NEG_ONE), dir, 1);
    }

    std::pair<int, float> optimize() {
        const size_t nbytes = (size_t)n_params * sizeof(float);

        cudaMallocAsync(&x_fp32,  nbytes,                     stream_);
        cudaMallocAsync(&x_trial, nbytes,                     stream_);
        cudaMallocAsync(&g_cur,   nbytes,                     stream_);
        cudaMallocAsync(&dir,     nbytes,                     stream_);
        cudaMallocAsync(&s_buf,   (size_t)m_corr * nbytes,    stream_);
        cudaMallocAsync(&y_buf,   (size_t)m_corr * nbytes,    stream_);
        cudaMallocAsync(&d_ws,    (size_t)(2 * m_corr + WS_EXTRA) * sizeof(float),
                        stream_);
        cudaMallocAsync(&d_imax,  sizeof(int),                 stream_);

        cublasSetStream(handle, stream_);
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
        lbfgs_set_scalar<<<1, 1, 0, stream_>>>(T(WS_NEG_ONE), -1.0f);

        /* x_fp32 <- x_pad (fp16 working buffer, the caller's initial guess) */
        half2float_copy<<<param_blks, 256, 0, stream_>>>(x_pad, x_fp32, n_params);

        /* initial evaluation: f0, g0 */
        evaluate(x_fp32);
        cudaMemcpyAsync(g_cur, grad_pad, nbytes,
                        cudaMemcpyDeviceToDevice, stream_);
        float f_cur = read_scalar(&pll_out[1]);

        head = 0; count = 0;
        int iter = 0;
        int stall = 0;   /* consecutive flat-f iterations (stall safety net) */
        for (; iter < max_iteration; ++iter) {
            /* PRIMARY convergence: ‖∇f‖∞ ≤ eps_g (size-invariant gtol) */
            const float ginf = g_inf_norm(g_cur);
            if (ginf <= eps_g) break;

            /* search direction dir = -H·g */
            two_loop_recursion();

            /* directional derivative gᵀd (must be < 0 for a descent step) */
            cublasSdot(handle, n_params, g_cur, 1, dir, 1, T(WS_GTD));
            float gTd = read_scalar(T(WS_GTD));
            if (!(gTd < 0.0f)) {
                /* L-BFGS direction not a descent dir (degenerate curvature):
                 * fall back to steepest descent, gᵀd = -‖g‖₂² (computed lazily). */
                cudaMemcpyAsync(dir, g_cur, nbytes,
                                cudaMemcpyDeviceToDevice, stream_);
                cublasSscal(handle, n_params, T(WS_NEG_ONE), dir, 1);
                cublasSdot(handle, n_params, g_cur, 1, g_cur, 1, T(WS_GG));
                gTd = -read_scalar(T(WS_GG));
            }

            /* backtracking (Armijo) line search; first step scaled by 1/‖g‖∞
             * on iteration 0 (steepest descent), unit step thereafter. */
            float step = (iter == 0) ? fminf(1.0f, 1.0f / (ginf + 1e-12f))
                                     : 1.0f;
            bool accepted = false;
            float f_trial = f_cur;
            for (int ls = 0; ls < kMaxLineEval; ++ls) {
                /* x_trial = x_fp32 + step * dir */
                cudaMemcpyAsync(x_trial, x_fp32, nbytes,
                                cudaMemcpyDeviceToDevice, stream_);
                lbfgs_set_scalar<<<1, 1, 0, stream_>>>(T(WS_STEP), step);
                cublasSaxpy(handle, n_params, T(WS_STEP), dir, 1, x_trial, 1);

                evaluate(x_trial);
                f_trial = read_scalar(&pll_out[1]);

                if (f_trial <= f_cur + kArmijoC1 * step * gTd) {
                    accepted = true;
                    break;
                }
                step *= kBacktrack;
                if (step < eps_x) break;   /* step underflow: give up search */
            }
            if (!accepted) break;          /* line search failed: keep last x */

            /* update history before overwriting x_fp32 / g_cur:
             *   s_k = x_trial - x_fp32 ,  y_k = ∇f(x_trial) - g_cur          */
            const int slot = head;
            float* sv = s_buf + (size_t)slot * n_params;
            float* yv = y_buf + (size_t)slot * n_params;
            cudaMemcpyAsync(sv, x_trial, nbytes,
                            cudaMemcpyDeviceToDevice, stream_);
            cublasSaxpy(handle, n_params, T(WS_NEG_ONE), x_fp32, 1, sv, 1);
            cudaMemcpyAsync(yv, grad_pad, nbytes,
                            cudaMemcpyDeviceToDevice, stream_);
            cublasSaxpy(handle, n_params, T(WS_NEG_ONE), g_cur, 1, yv, 1);
            cublasSdot(handle, n_params, yv, 1, sv, 1, T(WS_SY));
            lbfgs_rho<<<1, 1, 0, stream_>>>(RHO(slot), T(WS_SY));
            head = (head + 1) % m_corr;
            if (count < m_corr) ++count;

            /* commit accepted step */
            cudaMemcpyAsync(x_fp32, x_trial, nbytes,
                            cudaMemcpyDeviceToDevice, stream_);
            cudaMemcpyAsync(g_cur, grad_pad, nbytes,
                            cudaMemcpyDeviceToDevice, stream_);
            const float f_prev = f_cur;
            f_cur = f_trial;

            /* STALL safety net: not a convergence test — only catches a site
             * that has flat-lined (e.g. at the fp16 noise floor) so it doesn't
             * spin to max_iteration. Relative tolerance, must persist for
             * kStallPatience consecutive steps; eps_g remains the real stop. */
            if (fabsf(f_prev - f_cur) <= eps_f * (fabsf(f_cur) + 1.0f))
                ++stall;
            else
                stall = 0;
            if (stall >= kStallPatience) { ++iter; break; }
        }

        /* write the final fp32 solution back to the caller's fp16 buffer */
        float2half_copy<<<param_blks, 256, 0, stream_>>>(x_fp32, x_pad, n_params);
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);

        cudaFreeAsync(x_fp32,  stream_);
        cudaFreeAsync(x_trial, stream_);
        cudaFreeAsync(g_cur,   stream_);
        cudaFreeAsync(dir,     stream_);
        cudaFreeAsync(s_buf,   stream_);
        cudaFreeAsync(y_buf,   stream_);
        cudaFreeAsync(d_ws,    stream_);
        cudaFreeAsync(d_imax,  stream_);

        return {iter, f_cur};
    }
};

#endif
