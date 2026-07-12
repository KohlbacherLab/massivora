#ifndef OPTIMIZERS_CUH
#define OPTIMIZERS_CUH

#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cooperative_groups.h>

#include <cmath>
#include <utility>

#include "cuda_kernels.cuh"
#include "bindings.cuh"

namespace cg = cooperative_groups;

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
 * Fused helper kernels for the unconstrained L-BFGS.
 *
 * The two-loop recursion and the line-search vector ops are launch-latency
 * bound (many tiny ops per iteration). These kernels fuse the scalar-combine
 * with the element-wise update, so each inner step is ONE custom launch on top
 * of the cuBLAS dot — instead of dot + mul + neg/sub + axpy (4 launches). cuBLAS
 * dots run in CUBLAS_POINTER_MODE_DEVICE so their scalar results stay on the
 * device and feed these kernels with no host sync. All use grid-stride loops.
 * =========================================================================*/

/* out[i] = a*x[i] + b*y[i] with HOST scalars a, b — folds a memcpy+axpy (or a
 * scal) into one launch with no device scalar. Covers dir=-g (a=-1,b=0),
 * x_trial = x + step·dir, s = x_trial - x, y = g_new - g_old. Safe in place. */
__global__ void lbfgs_axpby(float* out, const float* x, const float* y,
                            float a, float b, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x)
        out[i] = a * x[i] + b * y[i];
}

/* First two-loop sweep, fused: alpha = rho·dot; *alpha_out = alpha; q -= alpha·y.
 * Replaces lbfgs_mul + lbfgs_neg + cublasSaxpy. */
__global__ void lbfgs_first_axpy(float* q, const float* y, float* alpha_out,
                                 const float* rho, const float* dot, int n) {
    const float alpha = (*rho) * (*dot);
    if (blockIdx.x == 0 && threadIdx.x == 0) *alpha_out = alpha;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x)
        q[i] -= alpha * y[i];
}

/* Second two-loop sweep, fused: coef = alpha - rho·dot; r += coef·s.
 * Replaces lbfgs_mul + lbfgs_sub + cublasSaxpy. */
__global__ void lbfgs_second_axpy(float* r, const float* s, const float* alpha,
                                  const float* rho, const float* dot, int n) {
    const float coef = (*alpha) - (*rho) * (*dot);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x)
        r[i] += coef * s[i];
}

/* Initial-Hessian scaling, fused: gamma = (s·y)/(y·y), clamped to 1 on
 * non-positive curvature; q *= gamma. Replaces lbfgs_gamma + cublasSscal. */
__global__ void lbfgs_scale_gamma(float* q, const float* sy, const float* yy,
                                  int n) {
    const float s = *sy, y = *yy;
    const float gamma = (s > 1e-12f && y > 1e-30f) ? (s / y) : 1.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x)
        q[i] *= gamma;
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

/* Block-wide sum of `val`, atomically accumulated into *gacc. The leading
 * __syncthreads makes sdata reusable across back-to-back calls. blockDim.x must
 * be a power of two. */
__device__ inline void lbfgs_block_reduce(float val, float* sdata, float* gacc) {
    const int tid = threadIdx.x;
    __syncthreads();
    sdata[tid] = val;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(gacc, sdata[0]);
}

/* =========================================================================
 * lbfgs_two_loop_coop – the ENTIRE two-loop recursion in one cooperative
 * launch. Replaces ~2·m_corr+3 separate dot/axpy launches: the dots become
 * grid-wide reductions (block reduce + atomicAdd into acc[], then grid.sync),
 * the axpys are grid-stride element-wise updates. Each thread always touches
 * the same indices of `dir`, so dir needs no cross-thread sync between an axpy
 * and the next dot — only the reduction accumulators do (one grid.sync each).
 *
 * MUST be launched with cudaLaunchCooperativeKernel and a grid that is fully
 * resident (cooperative requirement). Produces dir = -H·g (descent direction).
 * acc must hold >= 2*count+2 floats. count >= 1.
 * =========================================================================*/
__global__ void lbfgs_two_loop_coop(
    float* __restrict__ dir, const float* __restrict__ g,
    const float* __restrict__ s_buf, const float* __restrict__ y_buf,
    const float* __restrict__ rho, float* __restrict__ alpha,
    float* __restrict__ acc, int n, int m_corr, int head, int count)
{
    cg::grid_group grid = cg::this_grid();
    extern __shared__ float sdata[];
    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const int T    = gridDim.x * blockDim.x;

    /* dir = -g */
    for (int i = gtid; i < n; i += T) dir[i] = -g[i];

    /* zero the reduction accumulators acc[0 .. 2*count+1] */
    const int R = 2 * count + 2;
    for (int rr = gtid; rr < R; rr += T) acc[rr] = 0.0f;
    grid.sync();

    /* first loop: newest -> oldest */
    for (int j = 0; j < count; ++j) {
        const int slot = (head - 1 - j + 2 * m_corr) % m_corr;
        const float* sv = s_buf + (size_t)slot * n;
        const float* yv = y_buf + (size_t)slot * n;
        float local = 0.0f;
        for (int i = gtid; i < n; i += T) local += sv[i] * dir[i];   /* s·q */
        lbfgs_block_reduce(local, sdata, &acc[j]);
        grid.sync();
        const float a = rho[slot] * acc[j];
        if (gtid == 0) alpha[slot] = a;
        for (int i = gtid; i < n; i += T) dir[i] -= a * yv[i];        /* q -= αy */
    }

    /* initial Hessian scaling γ = (sᵀy)/(yᵀy) from the most-recent pair —
     * both reductions in one pass, one grid.sync */
    {
        const int slot = (head - 1 + 2 * m_corr) % m_corr;
        const float* sv = s_buf + (size_t)slot * n;
        const float* yv = y_buf + (size_t)slot * n;
        float lsy = 0.0f, lyy = 0.0f;
        for (int i = gtid; i < n; i += T) {
            const float sa = sv[i], yb = yv[i];
            lsy += sa * yb; lyy += yb * yb;
        }
        lbfgs_block_reduce(lsy, sdata, &acc[count]);
        lbfgs_block_reduce(lyy, sdata, &acc[count + 1]);
        grid.sync();
        const float sy = acc[count], yy = acc[count + 1];
        const float gamma = (sy > 1e-12f && yy > 1e-30f) ? (sy / yy) : 1.0f;
        for (int i = gtid; i < n; i += T) dir[i] *= gamma;
    }

    /* second loop: oldest -> newest */
    for (int j = 0; j < count; ++j) {
        const int slot = (head - count + j + 2 * m_corr) % m_corr;
        const float* sv = s_buf + (size_t)slot * n;
        const float* yv = y_buf + (size_t)slot * n;
        const int ai = count + 2 + j;
        float local = 0.0f;
        for (int i = gtid; i < n; i += T) local += yv[i] * dir[i];   /* y·r */
        lbfgs_block_reduce(local, sdata, &acc[ai]);
        grid.sync();
        const float coef = alpha[slot] - rho[slot] * acc[ai];
        for (int i = gtid; i < n; i += T) dir[i] += coef * sv[i];     /* r += (α-β)s */
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
    float* d_acc    = nullptr;   /* (2*m_corr+2,) cooperative two-loop reduce */

    /* cooperative single-kernel two-loop (set up in optimize()) */
    bool use_coop  = false;
    int  coop_grid = 0;

    /* device-scalar workspace layout: [0,m_corr)=rho, [m_corr,2m_corr)=alpha,
     * then the named temporaries below (only cuBLAS-dot results now — the rest
     * fold into the fused kernels as host scalars). */
    enum { WS_DOT = 0, WS_SY, WS_YY, WS_GTD, WS_GG, WS_GINF, WS_EXTRA };

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
    /* dir <- -H·g_cur (the descent direction). Dispatches to the cooperative
     * single-kernel two-loop when available, else the fused multi-launch path. */
    void two_loop_recursion() {
        if (count == 0) {
            /* no history: steepest descent, dir = -g */
            lbfgs_axpby<<<param_blks, 256, 0, stream_>>>(
                dir, g_cur, g_cur, -1.0f, 0.0f, n_params);
            return;
        }
        if (use_coop) two_loop_coop();
        else          two_loop_fused();
    }

    /* One cooperative launch for the whole two-loop (count >= 1). */
    void two_loop_coop() {
        const float* rho_ptr   = d_ws;            /* RHO   region */
        float*       alpha_ptr = d_ws + m_corr;   /* ALPHA region */
        void* args[] = {
            (void*)&dir, (void*)&g_cur, (void*)&s_buf, (void*)&y_buf,
            (void*)&rho_ptr, (void*)&alpha_ptr, (void*)&d_acc,
            (void*)&n_params, (void*)&m_corr, (void*)&head, (void*)&count
        };
        cudaLaunchCooperativeKernel((void*)lbfgs_two_loop_coop,
            dim3(coop_grid), dim3(256), args, 256 * sizeof(float), stream_);
    }

    /* Fused multi-launch two-loop: dir=-g, then per inner step a cuBLAS dot +
     * one fused element-wise kernel. The two-loop is linear in its input, so
     * seeding -g yields H·(-g) = -H·g directly (no final negate). count >= 1. */
    void two_loop_fused() {
        lbfgs_axpby<<<param_blks, 256, 0, stream_>>>(
            dir, g_cur, g_cur, -1.0f, 0.0f, n_params);

        /* first loop: newest -> oldest  (dot + fused {alpha, q-=αy}) */
        for (int j = 0; j < count; ++j) {
            const int s = slot_newest(j);
            const float* sv = s_buf + (size_t)s * n_params;
            const float* yv = y_buf + (size_t)s * n_params;
            cublasSdot(handle, n_params, sv, 1, dir, 1, T(WS_DOT));        /* s·q */
            lbfgs_first_axpy<<<param_blks, 256, 0, stream_>>>(
                dir, yv, ALPHA(s), RHO(s), T(WS_DOT), n_params);
        }

        /* initial Hessian scaling H0 = γ·I, γ = (sᵀy)/(yᵀy), from newest pair */
        {
            const int s = slot_newest(0);
            const float* sv = s_buf + (size_t)s * n_params;
            const float* yv = y_buf + (size_t)s * n_params;
            cublasSdot(handle, n_params, sv, 1, yv, 1, T(WS_SY));
            cublasSdot(handle, n_params, yv, 1, yv, 1, T(WS_YY));
            lbfgs_scale_gamma<<<param_blks, 256, 0, stream_>>>(
                dir, T(WS_SY), T(WS_YY), n_params);
        }

        /* second loop: oldest -> newest  (dot + fused {r += (α-β)s}) */
        for (int j = 0; j < count; ++j) {
            const int s = slot_oldest(j);
            const float* sv = s_buf + (size_t)s * n_params;
            const float* yv = y_buf + (size_t)s * n_params;
            cublasSdot(handle, n_params, yv, 1, dir, 1, T(WS_DOT));        /* y·r */
            lbfgs_second_axpy<<<param_blks, 256, 0, stream_>>>(
                dir, sv, ALPHA(s), RHO(s), T(WS_DOT), n_params);
        }
        /* dir already holds -H·g — no final negate. */
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

        /* Cooperative single-kernel two-loop: one launch instead of ~2·m_corr+3
         * dot/axpy launches. Enabled iff the device supports cooperative launch
         * and the kernel gets >=1 resident block/SM. The grid is sized to cover
         * n_params but capped to what stays co-resident (grid.sync requirement).
         * NB: a (near-)full-grid cooperative kernel holds the whole GPU during
         * grid.sync, so it cuts single-stream latency but does NOT overlap other
         * streams — best at n_streams=1. */
        {
            int dev = 0; cudaGetDevice(&dev);
            int coop = 0;
            cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, dev);
            if (coop) {
                int numSM = 0, blocksPerSM = 0;
                cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, dev);
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &blocksPerSM, (const void*)lbfgs_two_loop_coop, 256,
                    256 * sizeof(float));
                const int maxBlocks = blocksPerSM * numSM;
                const int needed    = (n_params + 255) / 256;
                coop_grid = (needed < maxBlocks) ? needed : maxBlocks;
                use_coop  = (blocksPerSM > 0 && coop_grid > 0);
            }
            if (use_coop)
                cudaMallocAsync(&d_acc, (size_t)(2 * m_corr + 2) * sizeof(float),
                                stream_);
        }

        cublasSetStream(handle, stream_);
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);

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
                lbfgs_axpby<<<param_blks, 256, 0, stream_>>>(
                    dir, g_cur, g_cur, -1.0f, 0.0f, n_params);
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
                /* x_trial = x_fp32 + step * dir  (one fused launch, host step) */
                lbfgs_axpby<<<param_blks, 256, 0, stream_>>>(
                    x_trial, x_fp32, dir, 1.0f, step, n_params);

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

            /* update history before overwriting x_fp32 / g_cur (each a single
             * fused diff): s_k = x_trial - x_fp32 ,  y_k = ∇f(x_trial) - g_cur */
            const int slot = head;
            float* sv = s_buf + (size_t)slot * n_params;
            float* yv = y_buf + (size_t)slot * n_params;
            lbfgs_axpby<<<param_blks, 256, 0, stream_>>>(
                sv, x_trial, x_fp32, 1.0f, -1.0f, n_params);
            lbfgs_axpby<<<param_blks, 256, 0, stream_>>>(
                yv, grad_pad, g_cur, 1.0f, -1.0f, n_params);
            cublasSdot(handle, n_params, yv, 1, sv, 1, T(WS_SY));
            lbfgs_rho<<<1, 1, 0, stream_>>>(RHO(slot), T(WS_SY));
            head = (head + 1) % m_corr;
            if (count < m_corr) ++count;

            /* commit accepted step: x_trial becomes the new x_fp32 by pointer
             * swap (no copy); g_cur <- ∇f at the accepted point. */
            std::swap(x_fp32, x_trial);
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
        if (d_acc) cudaFreeAsync(d_acc, stream_);

        return {iter, f_cur};
    }
};

#endif
