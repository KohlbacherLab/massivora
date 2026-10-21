// plm_dca_cuda.cuh -- the batched multi-site plmDCA on CUDA, as a Session.
//
// A Session holds what is worth keeping between pairs -- the CUDA context, a
// stream, a cuBLAS handle and pinned host staging -- and takes every buffer a
// pair needs from a stream-ordered memory pool, which returns it to the driver
// as soon as the pair is done. So a Session sits on no device memory between
// pairs that another process could use, nothing in it synchronises the whole
// device, and many Sessions can share a GPU under the caller's admission
// control (Session::device_bytes).
//
// The optimiser:
//
// Per site, the PLL objective is two GEMMs against the (B, N*q) one-hot MSA,
// but the other operand is only one site's q columns wide. Each GEMM therefore
// streams the whole one-hot matrix to do ~q FLOPs per byte, an order of
// magnitude below what a tensor-core GPU needs to be compute-bound, and the
// per-site path (cudaOptimizeSite) is bandwidth- and launch-latency-bound with
// the tensor cores mostly idle.
//
// Here up to S sites advance in lockstep. Every round, each active site has
// exactly one pending trial point; the trial couplings of all S sites are
// packed side by side into one (N*q, q*S) matrix, so a single pair of GEMMs,
// q*S columns wide, evaluates all of them. Each pass over the one-hot MSA is
// now shared by S sites. The L-BFGS bookkeeping of every site (two-loop
// recursion, Armijo backtracking, history, stopping rules) is that of
// LBFGS_Unconstrained; only its vector ops are batched across sites, so each
// kernel launch covers all sites that need it.
//
// Layouts (row-major unless noted):
//   X     (B, K)   fp16 one-hot, X[b, i*q + a] = [MSA[b, i] == a];
//                  K = N*q rounded up to a multiple of 8, padding zero.
//   JT    (M, K)   fp16, row s*q + c holds slot s's J[i, a, c] over k = i*q + a;
//                  M = q*S rounded up to a multiple of 8.
//   E     (B, M)   fp32 energies,  E = X @ JT^T       (h added in k_vgrad)
//   V     (B, M)   fp16 PLL residuals feeding GEMM 2
//   G     (K, M)   fp32,           G = X^T @ V
//   Per-slot optimizer vectors are compact: n = q + N*q*q real entries,
//   [h (q)] then J[i, a, c] at q + (i*q + a)*q + c. The q_pad padding of the
//   per-site path is always zero there, so dropping it is exact and cuts the
//   L-BFGS memory traffic by ~(q/q_pad)^2.
//   J     (N, N*q*q) fp16, row r = site r's couplings J[i, a, c] at
//                  (i*q + a)*q + c: the layout the per-site path publishes.
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace plm {
namespace cuda {

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

// The device could not hold a pair. Its memory has been handed back and the
// context is intact, so the next pair can run.
struct OutOfMemory : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// The CUDA context is gone. An illegal address, a failed launch and the like
// are sticky: every later call in the process fails the same way, whichever
// Session makes it, so only a new process can use the device again.
struct DeviceLost : std::runtime_error {
    using std::runtime_error::runtime_error;
};

inline void cuda_check(cudaError_t e, const char* what, int line) {
    if (e == cudaSuccess) return;
    const std::string msg = std::string("plm_dca_cuda: ") + what + ": " +
                            cudaGetErrorString(e) + " (line " + std::to_string(line) + ")";
    if (e == cudaErrorMemoryAllocation) throw OutOfMemory(msg);
    throw std::runtime_error(msg);
}
inline void cublas_check(cublasStatus_t s, const char* what, int line) {
    if (s == CUBLAS_STATUS_SUCCESS) return;
    const std::string msg = std::string("plm_dca_cuda: cuBLAS ") + what +
                            " failed with status " + std::to_string((int)s) +
                            " (line " + std::to_string(line) + ")";
    if (s == CUBLAS_STATUS_ALLOC_FAILED) throw OutOfMemory(msg);
    throw std::runtime_error(msg);
}
#define PLM_CUDA(x)   ::plm::cuda::cuda_check((x), #x, __LINE__)
#define PLM_CUBLAS(x) ::plm::cuda::cublas_check((x), #x, __LINE__)

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

namespace detail {

constexpr int kMaxSlots = 64;   /* slot table size, passed by value */
constexpr int kThreads  = 256;
constexpr int kVgradWarps = 8;

/* L-BFGS constants and defaults: identical to cudaOptimizeSite's L-BFGS path
 * (m_corr, eps_g, eps_f, eps_x) and LBFGS_Unconstrained's line search. */
constexpr int   kMCorr         = 8;
constexpr float kEpsG          = 1e-3f;
constexpr float kEpsF          = 1e-9f;
constexpr float kEpsX          = 1e-9f;
constexpr float kArmijoC1      = 1e-4f;
constexpr float kBacktrack     = 0.5f;
constexpr int   kMaxLineEval   = 10;
constexpr int   kStallPatience = 5;

/* Slot table for one launch, passed by value: entry e (= blockIdx.y) works on
 * slot slot[e], which currently holds site site[e]. */
struct SlotArgs {
    int   n;
    int   slot [kMaxSlots];
    int   site [kMaxSlots];
    int   count[kMaxSlots];   /* L-BFGS history length       */
    int   head [kMaxSlots];   /* ring position of next write */
    float step [kMaxSlots];   /* line-search step            */
    int   bank [kMaxSlots];   /* which of the two x/g arrays holds x, g */
};

/* x and x_trial (and g and g_trial) live in two arrays, P0/P1 and Q0/Q1; a
 * slot's bank bit says which holds the current point. Accepting a step, or
 * committing a site's first evaluation, flips the bit instead of copying the
 * trial vectors over the current ones. */
#define PLM_X(e)  ((a.bank[e] ? P1 : P0) + (size_t)a.slot[e] * ns)
#define PLM_XT(e) ((a.bank[e] ? P0 : P1) + (size_t)a.slot[e] * ns)
#define PLM_G(e)  ((a.bank[e] ? Q1 : Q0) + (size_t)a.slot[e] * ns)
#define PLM_GT(e) ((a.bank[e] ? Q0 : Q1) + (size_t)a.slot[e] * ns)

/* =========================================================================
 * Reductions
 * =========================================================================*/
__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

/* Sum / max over a 1-D block whose size is a multiple of 32; the result is
 * valid in thread 0. Every thread must call it. Safe to call back to back. */
__device__ float block_sum(float v) {
    __shared__ float sh[32];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    v = (threadIdx.x < (blockDim.x >> 5)) ? sh[threadIdx.x] : 0.0f;
    return warp_sum(v);
}

__device__ float block_max(float v) {   /* v >= 0 */
    __shared__ float sh[32];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = warp_max(v);
    __syncthreads();
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    v = (threadIdx.x < (blockDim.x >> 5)) ? sh[threadIdx.x] : 0.0f;
    return warp_max(v);
}

#define PLM_GRID_STRIDE(i, n) \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += gridDim.x * blockDim.x)

/* The per-slot optimizer vectors (x, xt, g, gt, dir, s, y) are laid out with
 * stride ns = n rounded up to a multiple of 4, padding held at zero, so the
 * element-wise kernels below move float4s: 16-byte accesses keep more bytes
 * in flight than scalar loads and reach ~1.2x the bandwidth on an H100 for
 * these streams. Every op below is element-wise or a sum over elements, so a
 * zero pad stays zero and adds nothing to a dot product. */
__device__ __forceinline__ float4 ld4(const float* p, int i) {
    return reinterpret_cast<const float4*>(p)[i];
}
__device__ __forceinline__ void st4(float* p, int i, float4 v) {
    reinterpret_cast<float4*>(p)[i] = v;
}
__device__ __forceinline__ float dot4(float4 a, float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

/* =========================================================================
 * Setup / slot lifecycle
 * =========================================================================*/
__global__ void k_onehot(const signed char* __restrict__ MSA, __half* __restrict__ X,
                         int B, int N, int q, int K) {
    const long long total = (long long)B * K;
    const int Nq = N * q;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (long long)gridDim.x * blockDim.x) {
        const int b = (int)(idx / K), k = (int)(idx - (long long)b * K);
        float v = 0.0f;
        if (k < Nq) {
            const int i = k / q;
            v = (MSA[(size_t)b * N + i] == k - i * q) ? 1.0f : 0.0f;
        }
        X[idx] = __float2half(v);
    }
}

/* x = x_trial = 0: every site starts from zero, and its first trial point is
 * that initial guess. */
__global__ void k_load(SlotArgs a, float* __restrict__ x, float* __restrict__ xt, int ns) {
    const size_t base = (size_t)a.slot[blockIdx.y] * ns;
    const float4 z = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    PLM_GRID_STRIDE(i, ns / 4) {
        st4(x + base, i, z);
        st4(xt + base, i, z);
    }
}

__device__ __forceinline__ float round_fp16(float v) { return __half2float(__float2half(v)); }

/* Row `site` of J = the site's couplings in fp16, the precision the per-site
 * path keeps its solution in. h is not kept. */
__global__ void k_store(SlotArgs a, const float* __restrict__ P0, const float* __restrict__ P1,
                        __half* __restrict__ J, int n, int q, int ns) {
    const float* xs = PLM_X(blockIdx.y) + q;
    __half* out = J + (size_t)a.site[blockIdx.y] * (n - q);
    PLM_GRID_STRIDE(i, n - q) out[i] = __float2half(xs[i]);
}

/* =========================================================================
 * Batched objective: pack -> GEMM 1 -> k_vgrad -> GEMM 2 -> k_unpack
 * =========================================================================*/

/* Round each listed slot's trial point to fp16 into its rows of JT (h into
 * h_eval), and accumulate Σh² and ΣJ² of the rounded values -- the same
 * quantities sum_squares reads from the fp16 x_pad on the per-site path.
 * parts[3s + 0..2] = {PLL, Σh², ΣJ²}.
 *
 * JT is K-major (row s*q + c holds J[:, :, c] over all k = i*q + a), which is
 * the layout the GEMM 1 kernels run fastest on. Compact x has c fastest, so
 * 32 coupling rows at a time are staged through shared memory and written
 * out transposed, keeping both the reads and the writes coalesced. 1-D block
 * of kThreads. */
__global__ void k_pack(SlotArgs a, const float* __restrict__ P0, const float* __restrict__ P1,
                       __half* __restrict__ JT,
                       float* __restrict__ h_eval, float* __restrict__ parts,
                       int n, int q, int K, int ns) {
    __shared__ float tile[32][33];
    const int s = a.slot[blockIdx.y];
    const float* x = PLM_XT(blockIdx.y);
    const int Nq = (n - q) / q;
    const int tx = threadIdx.x & 31, ty = threadIdx.x >> 5, ny = blockDim.x >> 5;
    float hh = 0.0f, jj = 0.0f;
    for (int k0 = blockIdx.x * 32; k0 < Nq; k0 += gridDim.x * 32) {
        const int rows = min(32, Nq - k0);
        const float* src = x + q + (size_t)k0 * q;
        for (int t = threadIdx.x; t < rows * q; t += blockDim.x) {
            const float v = round_fp16(src[t]);
            tile[t / q][t % q] = v;
            jj += v * v;
        }
        __syncthreads();
        if (tx < rows)
            for (int c = ty; c < q; c += ny)
                JT[(size_t)(s * q + c) * K + k0 + tx] = __float2half(tile[tx][c]);
        __syncthreads();
    }
    if (blockIdx.x == 0 && threadIdx.x < q) {
        const float v = round_fp16(x[threadIdx.x]);
        h_eval[s * q + threadIdx.x] = v;
        hh = v * v;
    }
    hh = block_sum(hh);
    jj = block_sum(jj);
    if (threadIdx.x == 0) {
        atomicAdd(&parts[3 * s + 1], hh);
        atomicAdd(&parts[3 * s + 2], jj);
    }
}

/* fused_vgrad for many slots: one warp per (sequence b, slot), lane = state.
 * Softmax over the q states of site r, PLL residual into V, and per-slot
 * reductions of the PLL and of ∇h (= -Σ_b residual). Block = (32, 8). */
__global__ void k_vgrad(SlotArgs a, const float* __restrict__ E, const float* __restrict__ h_eval,
                        const signed char* __restrict__ MSA, const float* __restrict__ W,
                        __half* __restrict__ V, float* __restrict__ parts, float* __restrict__ gh,
                        int B, int N, int q, int M) {
    const int s = a.slot[blockIdx.y], r = a.site[blockIdx.y];
    const int lane = threadIdx.x, wy = threadIdx.y;
    const bool live = lane < q;
    const int col = s * q + lane;
    const float hc = live ? h_eval[col] : 0.0f;

    float gacc = 0.0f, pacc = 0.0f;
    for (int b = blockIdx.x * blockDim.y + wy; b < B; b += gridDim.x * blockDim.y) {
        const float e    = live ? E[(size_t)b * M + col] + hc : -INFINITY;
        const float mx   = warp_max(e);
        const float se   = warp_sum(live ? expf(e - mx) : 0.0f);
        const float logZ = mx + logf(se);
        const int   sr   = MSA[(size_t)b * N + r];
        const float wb   = W[b];
        const float p    = live ? expf(e - logZ) : 0.0f;
        const float vg   = (lane == sr) ? wb * (1.0f - p) : -wb * p;
        if (live) V[(size_t)b * M + col] = __float2half(vg);
        gacc -= vg;
        pacc -= wb * (__shfl_sync(0xffffffffu, e, sr) - logZ);
    }

    __shared__ float sg[kVgradWarps][32];
    __shared__ float sp[kVgradWarps];
    sg[wy][lane] = gacc;
    if (lane == 0) sp[wy] = pacc;
    __syncthreads();
    if (wy == 0) {
        float t = 0.0f;
        for (int w = 0; w < blockDim.y; ++w) t += sg[w][lane];
        if (live) atomicAdd(&gh[col], t);
        if (lane == 0) {
            float u = 0.0f;
            for (int w = 0; w < blockDim.y; ++w) u += sp[w];
            atomicAdd(&parts[3 * s], u);
        }
    }
}

/* ∇f into each slot's g_trial: ∇J = λJ·J - G, ∇h = gh + 2λh·h, with the
 * self-coupling block i == r held at zero (as on the per-site path). J and h
 * are the fp16-rounded values the GEMMs saw, matching grad_l2. */
__global__ void k_unpack(SlotArgs a, const float* __restrict__ G,
                         const float* __restrict__ P0, const float* __restrict__ P1,
                         const float* __restrict__ h_eval, const float* __restrict__ gh,
                         float* __restrict__ Q0, float* __restrict__ Q1, int n, int q, int M,
                         float lambdaH, float lambdaJ, int ns) {
    const int s = a.slot[blockIdx.y], r = a.site[blockIdx.y];
    const float* xt = PLM_XT(blockIdx.y);
    float* gt = PLM_GT(blockIdx.y);
    PLM_GRID_STRIDE(i, n) {
        if (i < q) {
            gt[i] = gh[s * q + i] + 2.0f * lambdaH * h_eval[s * q + i];
        } else {
            const int j = i - q, k = j / q;
            const size_t off = (size_t)k * M + s * q + (j - k * q);
            gt[i] = (k / q == r) ? 0.0f
                                 : lambdaJ * round_fp16(xt[i]) - G[off];
        }
    }
}

/* =========================================================================
 * Batched L-BFGS vector ops
 * =========================================================================*/

/* The steepest-descent fallback dir = -g. */
__global__ void k_neg_g(SlotArgs a, float* __restrict__ dir, const float* __restrict__ Q0,
                        const float* __restrict__ Q1, int ns) {
    const float* g = PLM_G(blockIdx.y);
    float* d = dir + (size_t)a.slot[blockIdx.y] * ns;
    PLM_GRID_STRIDE(i, ns / 4) {
        const float4 v = ld4(g, i);
        st4(d, i, make_float4(-v.x, -v.y, -v.z, -v.w));
    }
}

/* x_trial = x + step * dir */
__global__ void k_trial(SlotArgs a, float* __restrict__ P0, float* __restrict__ P1,
                        const float* __restrict__ dir, int ns) {
    const float* x = PLM_X(blockIdx.y);
    float* xt = PLM_XT(blockIdx.y);
    const float* d0 = dir + (size_t)a.slot[blockIdx.y] * ns;
    const float step = a.step[blockIdx.y];
    PLM_GRID_STRIDE(i, ns / 4) {
        const float4 u = ld4(x, i), d = ld4(d0, i);
        st4(xt, i, make_float4(u.x + step * d.x, u.y + step * d.y,
                               u.z + step * d.z, u.w + step * d.w));
    }
}

/* Accepted step, one pass: s_k = x_trial - x, y_k = g_trial - g into ring slot
 * head and accumulate s·y, y·y into acc[2s], acc[2s + 1]. The caller commits
 * x <- x_trial and g <- g_trial by flipping the slot's bank bit. */
__global__ void k_accept(SlotArgs a, const float* __restrict__ P0, const float* __restrict__ P1,
                         const float* __restrict__ Q0, const float* __restrict__ Q1,
                         float* __restrict__ sbuf, float* __restrict__ ybuf,
                         float* __restrict__ acc, int ns, int m) {
    const int e = blockIdx.y, s = a.slot[e];
    const float *x = PLM_X(e), *xt = PLM_XT(e), *g = PLM_G(e), *gt = PLM_GT(e);
    float* sv = sbuf + ((size_t)s * m + a.head[e]) * ns;
    float* yv = ybuf + ((size_t)s * m + a.head[e]) * ns;
    float lsy = 0.0f, lyy = 0.0f;
    PLM_GRID_STRIDE(i, ns / 4) {
        const float4 xn = ld4(xt, i), gn = ld4(gt, i);
        const float4 xo = ld4(x, i),  go = ld4(g, i);
        const float4 sd = make_float4(xn.x - xo.x, xn.y - xo.y, xn.z - xo.z, xn.w - xo.w);
        const float4 yd = make_float4(gn.x - go.x, gn.y - go.y, gn.z - go.z, gn.w - go.w);
        st4(sv, i, sd);
        st4(yv, i, yd);
        lsy += dot4(sd, yd);
        lyy += dot4(yd, yd);
    }
    lsy = block_sum(lsy);
    lyy = block_sum(lyy);
    if (threadIdx.x == 0) {
        atomicAdd(&acc[2 * s], lsy);
        atomicAdd(&acc[2 * s + 1], lyy);
    }
}

/* rho = 1/(y·s) with the positive-curvature guard of lbfgs_rho, and the
 * initial-Hessian scale γ = (s·y)/(y·y) of the newest pair as in the two-loop. */
__global__ void k_curvature(SlotArgs a, const float* __restrict__ acc,
                            float* __restrict__ rho_hist, float* __restrict__ gamma, int m) {
    const int e = threadIdx.x;
    if (e >= a.n) return;
    const int s = a.slot[e];
    const float sy = acc[2 * s], yy = acc[2 * s + 1];
    rho_hist[s * m + a.head[e]] = (sy > 1e-12f) ? (1.0f / sy) : 0.0f;
    gamma[s] = (sy > 1e-12f && yy > 1e-30f) ? (sy / yy) : 1.0f;
}

/* One step t of the two-loop recursion for every listed slot, dir <- -H·g.
 * A slot with history length c runs steps t = 0 .. 2c; each step fuses the
 * element-wise update that consumes the previous dot with the next dot, so
 * the whole recursion is 2c+1 passes and needs no host round trip:
 *   t = 0        q = -g;                    a1[0] = s_σ0·q   (+ g·g, ‖g‖∞)
 *   0 < t < c    q -= α_{t-1} y_σ(t-1);     a1[t] = s_σt·q
 *   t = c        r = γ(q - α_{c-1} y_σ(c-1)); a2[0] = y_τ0·r
 *   c < t < 2c   r += (α - β) s_τ(j-1);     a2[j] = y_τj·r   (j = t - c)
 *   t = 2c       r += (α - β) s_τ(c-1);     gᵀd = g·r
 * σ_j is the j-th newest pair, τ_j = σ_{c-1-j} the j-th oldest. With c = 0 the
 * single step t = 0 leaves dir = -g. Per-slot scratch sc (2m + 3 floats):
 * [a1 (m), a2 (m), g·g, ‖g‖∞, gᵀd], zeroed before t = 0. */
__global__ void k_twoloop(SlotArgs a, int t, const float* __restrict__ Q0,
                          const float* __restrict__ Q1, float* __restrict__ P0,
                          float* __restrict__ P1, float* __restrict__ dir,
                          const float* __restrict__ sbuf, const float* __restrict__ ybuf,
                          const float* __restrict__ rho_hist, const float* __restrict__ gamma,
                          float* __restrict__ scr, int ns, int m) {
    const int e = blockIdx.y, s = a.slot[e], c = a.count[e], h = a.head[e];
    if (t > 2 * c) return;
    const int n4 = ns / 4;

    const size_t base = (size_t)s * ns;
    const float* gs = PLM_G(e);
    float*       d  = dir + base;
    float*       sc = scr + (size_t)s * (2 * m + 3);
    const float* S0 = sbuf + (size_t)s * m * ns;
    const float* Y0 = ybuf + (size_t)s * m * ns;
    const float* rh = rho_hist + (size_t)s * m;
    auto sigma = [&](int j) { return (h - 1 - j + 2 * m) % m; };

    if (t == 0) {
        const float* sv = (c > 0) ? S0 + (size_t)sigma(0) * ns : nullptr;
        float l_dot = 0.0f, l_gg = 0.0f, l_max = 0.0f;
        PLM_GRID_STRIDE(i, n4) {
            const float4 gi = ld4(gs, i);
            st4(d, i, make_float4(-gi.x, -gi.y, -gi.z, -gi.w));
            l_gg += dot4(gi, gi);
            l_max = fmaxf(l_max, fmaxf(fmaxf(fabsf(gi.x), fabsf(gi.y)),
                                       fmaxf(fabsf(gi.z), fabsf(gi.w))));
            if (c > 0) l_dot -= dot4(ld4(sv, i), gi);
        }
        l_dot = block_sum(l_dot);
        l_gg  = block_sum(l_gg);
        l_max = block_max(l_max);
        if (threadIdx.x == 0) {
            if (c > 0) atomicAdd(&sc[0], l_dot);
            atomicAdd(&sc[2 * m], l_gg);
            atomicMax(reinterpret_cast<int*>(&sc[2 * m + 1]), __float_as_int(l_max));
        }
        return;
    }

    float l = 0.0f;
    float* out;
    if (t < c) {
        const int sp = sigma(t - 1);
        const float alpha = rh[sp] * sc[t - 1];
        const float* yp = Y0 + (size_t)sp * ns;
        const float* sv = S0 + (size_t)sigma(t) * ns;
        PLM_GRID_STRIDE(i, n4) {
            const float4 dv = ld4(d, i), y = ld4(yp, i);
            const float4 qv = make_float4(dv.x - alpha * y.x, dv.y - alpha * y.y,
                                          dv.z - alpha * y.z, dv.w - alpha * y.w);
            st4(d, i, qv);
            l += dot4(ld4(sv, i), qv);
        }
        out = &sc[t];
    } else if (t == c) {
        const int sp = sigma(c - 1);                 /* newest-last == oldest */
        const float alpha = rh[sp] * sc[c - 1];
        const float gm = gamma[s];
        const float* yp = Y0 + (size_t)sp * ns;
        PLM_GRID_STRIDE(i, n4) {
            const float4 dv = ld4(d, i), y = ld4(yp, i);
            const float4 rv = make_float4(gm * (dv.x - alpha * y.x), gm * (dv.y - alpha * y.y),
                                          gm * (dv.z - alpha * y.z), gm * (dv.w - alpha * y.w));
            st4(d, i, rv);
            l += dot4(y, rv);
        }
        out = &sc[m];
    } else {
        const int j  = t - c;                        /* 1 .. c */
        const int tp = sigma(c - j);                 /* τ_{j-1} */
        const float coef = rh[tp] * sc[c - j] - rh[tp] * sc[m + j - 1];
        const float* sp = S0 + (size_t)tp * ns;
        const float* dv = (j < c) ? Y0 + (size_t)sigma(c - 1 - j) * ns : gs;
        /* The last pass also writes the trial point x + dir: with history the
         * step is 1 (z.step is 1 once iter > 0, and c > 0 implies iter > 0).
         * A slot that then needs the -g fallback gets k_trial again. */
        const bool fuse = (j == c);
        const float* xs = PLM_X(e);
        float* xts = PLM_XT(e);
        PLM_GRID_STRIDE(i, n4) {
            const float4 d0 = ld4(d, i), sx = ld4(sp, i);
            const float4 rv = make_float4(d0.x + coef * sx.x, d0.y + coef * sx.y,
                                          d0.z + coef * sx.z, d0.w + coef * sx.w);
            st4(d, i, rv);
            l += dot4(ld4(dv, i), rv);
            if (fuse) {
                const float4 u = ld4(xs, i);
                st4(xts, i, make_float4(u.x + rv.x, u.y + rv.y, u.z + rv.z, u.w + rv.w));
            }
        }
        out = (j < c) ? &sc[m + j] : &sc[2 * m + 2];
    }
    l = block_sum(l);
    if (threadIdx.x == 0) atomicAdd(out, l);
}

/* =========================================================================
 * Sequence reweighting neighbour counts -- GaussDCA's k_similarity (see
 * gauss_dca_cuda.cuh and cuda_kernels.cuh's pairwise_similarity_tiled), so a
 * pair stored without weights gets the very counts reweightNeighbourCounts
 * returns: m[b] is how many sequences, b itself included, agree with sequence
 * b at >= threshold of the N positions.
 * =========================================================================*/
constexpr int kRwTile  = 32;                 /* rows per tile                  */
constexpr int kRwChunk = 32;                 /* uint32 words staged per pass   */
constexpr int kRwTy    = 8;                  /* threadIdx.y extent             */
constexpr int kRwPt    = kRwTile / kRwTy;

__global__ void k_rw_fill_ones(int* __restrict__ m, int B) {
    for (long t = blockIdx.x * (long)blockDim.x + threadIdx.x; t < B;
         t += (long)gridDim.x * blockDim.x)
        m[t] = 1;                            /* every sequence matches itself */
}

/* (B, Nw) uint32 rows, zero-padded on the right by `pad` residues; upper
 * triangle of tiles; m[] must start at 1. */
__global__ void k_rw_similarity(const unsigned int* __restrict__ msa32, int* __restrict__ m,
                                int B, int Nw, int pad, int threshold) {
    const int tI = blockIdx.x, tJ = blockIdx.y;
    if (tJ < tI) return;
    const int i0 = tI * kRwTile, j0 = tJ * kRwTile;
    if (i0 >= B || j0 >= B) return;

    __shared__ unsigned int sA[kRwChunk][kRwTile + 1];
    __shared__ unsigned int sB[kRwChunk][kRwTile + 1];
    __shared__ int cntA[kRwTile], cntB[kRwTile];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = tx + ty * kRwTile;
    const int nthread = kRwTile * kRwTy;
    if (tid < kRwTile) { cntA[tid] = 0; cntB[tid] = 0; }

    int acc[kRwPt];
    #pragma unroll
    for (int k = 0; k < kRwPt; ++k) acc[k] = 0;

    for (int c0 = 0; c0 < Nw; c0 += kRwChunk) {
        __syncthreads();
        for (int idx = tid; idx < kRwTile * kRwChunk; idx += nthread) {
            const int r = idx / kRwChunk, w = idx % kRwChunk;
            const int c = c0 + w;
            const int ga = i0 + r, gb = j0 + r;
            sA[w][r] = (ga < B && c < Nw) ? msa32[(size_t)ga * Nw + c] : 0u;
            sB[w][r] = (gb < B && c < Nw) ? msa32[(size_t)gb * Nw + c] : 0u;
        }
        __syncthreads();
        const int wmax = (Nw - c0) < kRwChunk ? (Nw - c0) : kRwChunk;
        for (int w = 0; w < wmax; ++w) {
            const unsigned int bw = sB[w][tx];
            #pragma unroll
            for (int k = 0; k < kRwPt; ++k) {
                const unsigned int x = sA[w][ty + k * kRwTy] ^ bw;
                /* Zero bytes of x are the residues that matched. */
                const unsigned int nz = ((x & 0x7F7F7F7Fu) + 0x7F7F7F7Fu) | x | 0x7F7F7F7Fu;
                acc[k] += __popc(~nz);
            }
        }
    }
    __syncthreads();

    const int j = j0 + tx;
    #pragma unroll
    for (int k = 0; k < kRwPt; ++k) {
        const int i = i0 + ty + k * kRwTy;
        if (i < B && j < B && i < j && acc[k] - pad >= threshold) {
            atomicAdd(&cntA[ty + k * kRwTy], 1);
            atomicAdd(&cntB[tx], 1);
        }
    }
    __syncthreads();
    if (tid < kRwTile) {
        if (i0 + tid < B && cntA[tid]) atomicAdd(&m[i0 + tid], cntA[tid]);
        if (j0 + tid < B && cntB[tid]) atomicAdd(&m[j0 + tid], cntB[tid]);
    }
}

/* =========================================================================
 * Score: Frobenius norm of the gauged, symmetrised couplings
 * =========================================================================*/
constexpr int kFnWarps = 8;

/* FN[r, i] = FN[i, r] for every i > r: the Frobenius norm, gap state `gap`
 * excluded, of the Ising-gauged symmetrised block (J[r, i] + J[i, r]^T) / 2.
 * That is compute_score's FN of the couplings after applyIsingGauge: the gauge
 * is a double centring, linear and commuting with the transpose, so applying
 * it after the symmetrisation gives the same block. One warp per (r, i),
 * blockIdx.y = r; the diagonal is left as the caller set it (zero). */
__global__ void k_frobenius(const __half* __restrict__ J, float* __restrict__ FN,
                            int N, int q, int gap) {
    __shared__ float t[kFnWarps][32][33];
    __shared__ float rm[kFnWarps][32], cm[kFnWarps][32];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int r = blockIdx.y, i = blockIdx.x * kFnWarps + w;
    if (i <= r || i >= N) return;               /* whole warps only; no block barrier below */

    const int qq = q * q;
    const size_t Nqq = (size_t)N * qq;
    const __half* A  = J + (size_t)r * Nqq + (size_t)i * qq;   /* J[r, i, a, c] */
    const __half* Bt = J + (size_t)i * Nqq + (size_t)r * qq;   /* J[i, r, c, a] */
    for (int idx = lane; idx < qq; idx += 32) {
        const int a = idx / q, c = idx - a * q;
        t[w][a][c] = 0.5f * (__half2float(A[idx]) + __half2float(Bt[c * q + a]));
    }
    __syncwarp();
    float rsum = 0.0f, csum = 0.0f;
    if (lane < q) {
        for (int c = 0; c < q; ++c) rsum += t[w][lane][c];
        for (int a = 0; a < q; ++a) csum += t[w][a][lane];
    }
    rm[w][lane] = rsum / q;
    cm[w][lane] = csum / q;
    const float mean = warp_sum(rsum) / qq;
    __syncwarp();
    float acc = 0.0f;
    for (int idx = lane; idx < qq; idx += 32) {
        const int a = idx / q, c = idx - a * q;
        if (a == gap || c == gap) continue;
        const float v = t[w][a][c] - rm[w][a] - cm[w][c] + mean;
        acc += v * v;
    }
    acc = warp_sum(acc);
    if (lane == 0) {
        const float f = sqrtf(acc);
        FN[(size_t)r * N + i] = f;
        FN[(size_t)i * N + r] = f;
    }
}

__global__ void k_half_to_float(const __half* __restrict__ src, float* __restrict__ dst,
                                size_t n) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        dst[i] = __half2float(src[i]);
}

inline int round_up(int v, int k) { return (v + k - 1) / k * k; }

}  // namespace detail

// ---------------------------------------------------------------------------
// Results, settings, device properties
// ---------------------------------------------------------------------------

struct Stats {
    int    slots = 0;       // sites optimised concurrently
    long   rounds = 0;      // batched evaluations (one GEMM pair each)
    long   evals = 0;       // site evaluations, summed over rounds
    long   iterations = 0;  // accepted L-BFGS steps, summed over sites
    int    converged = 0;   // stopped on ‖∇f‖∞ ≤ eps_g
    int    stalled = 0;     // stopped by the flat-f safety net
    int    max_iter = 0;    // stopped at maxeval
    int    ls_failed = 0;   // line search found no sufficient decrease
    double f_sum = 0.0;     // final objective, summed over sites
};

struct Config {
    float lambdaH = 0.01f;
    float lambdaJ = 0.01f;
    int   maxeval = 100;    // L-BFGS iterations per site
    int   site_batch = 24;  // upper bound on sites in lockstep (see Session::slots)
};

struct DeviceProps {
    int    sm = 1;
    size_t max_persist = 0;   // persisting-L2 set-aside the device allows, bytes
    size_t max_window = 0;    // largest access-policy window, bytes
};

inline DeviceProps device_props(int device) {
    DeviceProps p;
    int v = 0;
    if (cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, device) == cudaSuccess) p.sm = v;
    if (cudaDeviceGetAttribute(&v, cudaDevAttrMaxPersistingL2CacheSize, device) == cudaSuccess)
        p.max_persist = (size_t)v;
    if (cudaDeviceGetAttribute(&v, cudaDevAttrMaxAccessPolicyWindowSize, device) == cudaSuccess)
        p.max_window = (size_t)v;
    cudaGetLastError();
    return p;
}

// The persisting-L2 set-aside: how much of the L2 the search directions may
// hold, as a share of what the device allows. More is not better -- the rest
// of the L2 is what the GEMMs and the streaming kernels have left -- and 80%
// measured best on an H100 (31 MB allowed of 50 MB).
constexpr double kL2SetAside = 0.8;

inline size_t l2_set_aside(const DeviceProps& p) {
    return (size_t)(kL2SetAside * (double)p.max_persist);
}

// What the Sessions of a process on one device share, set up by the first to
// arrive and handed back by the last to leave:
//
//   * the persisting-L2 set-aside, a property of the context. Changing it
//     synchronises the whole device, so it is set once; each Session pins no
//     more than its share of it (l2_share).
//   * a memory pool of their own, with no release threshold: what a pair
//     hands back goes straight back to the driver (measured free of cost), so
//     a process idle between pairs, or busy with small ones, does not sit on
//     device memory another process -- or a later, larger pair -- needs. The
//     device's default pool is left alone for whatever else the process runs.
class DeviceShared {
public:
    static cudaMemPool_t join(int device, size_t set_aside) {
        std::lock_guard<std::mutex> lock(mutex());
        Entry& e = entries()[device];
        if (e.users++ == 0) {
            if (set_aside > 0 &&
                cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, set_aside) != cudaSuccess)
                cudaGetLastError();         // best effort: unpinned is only slower
            int pools = 0;
            cudaDeviceGetAttribute(&pools, cudaDevAttrMemoryPoolsSupported, device);
            if (pools) {
                cudaMemPoolProps props{};
                props.allocType     = cudaMemAllocationTypePinned;
                props.handleTypes   = cudaMemHandleTypeNone;
                props.location.type = cudaMemLocationTypeDevice;
                props.location.id   = device;
                uint64_t keep = 0;
                if (cudaMemPoolCreate(&e.pool, &props) != cudaSuccess ||
                    cudaMemPoolSetAttribute(e.pool, cudaMemPoolAttrReleaseThreshold, &keep) != cudaSuccess) {
                    cudaGetLastError();     // fall back to cudaMalloc / cudaFree
                    e.pool = nullptr;
                }
            }
            cudaGetLastError();
        }
        return e.pool;
    }
    static void leave(int device) {
        std::lock_guard<std::mutex> lock(mutex());
        Entry& e = entries()[device];
        if (--e.users == 0) {
            if (e.pool) cudaMemPoolDestroy(e.pool);
            e.pool = nullptr;
            cudaCtxResetPersistingL2Cache();
            cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 0);
            cudaGetLastError();
        }
    }
private:
    struct Entry { int users = 0; cudaMemPool_t pool = nullptr; };
    static std::mutex& mutex() { static std::mutex m; return m; }
    static std::map<int, Entry>& entries() { static std::map<int, Entry> e; return e; }
};

// ---------------------------------------------------------------------------
// Per-pair device memory and host staging
//
// Every buffer a pair needs comes from the Sessions' stream-ordered pool and
// goes back to it when the pair is done (see DeviceShared). Allocating and
// freeing through a pool does not synchronise the device, so concurrent
// Sessions do not stall each other the way cudaMalloc/cudaFree would.
// ---------------------------------------------------------------------------

class PairArena {
public:
    PairArena(cudaStream_t stream, cudaMemPool_t pool) : stream_(stream), pool_(pool) {}
    ~PairArena() { release(); }
    PairArena(const PairArena&) = delete;
    PairArena& operator=(const PairArena&) = delete;

    template <class T> T* get(size_t count) {
        const size_t bytes = std::max<size_t>(count, 1) * sizeof(T);
        void* p = nullptr;
        cudaError_t e = alloc(&p, bytes);
        if (e == cudaErrorMemoryAllocation && pool_) {
            // Whatever the pool still holds back may be what is missing.
            cudaGetLastError();
            cudaMemPoolTrimTo(pool_, 0);
            e = alloc(&p, bytes);
        }
        if (e != cudaSuccess) {
            const std::string msg = "plm_dca_cuda: allocating " + std::to_string(bytes >> 20) +
                                    " MiB on the device: " + cudaGetErrorString(e);
            if (e == cudaErrorMemoryAllocation) throw OutOfMemory(msg);
            throw std::runtime_error(msg);
        }
        ptrs_.push_back(p);
        bytes_ += bytes;
        return static_cast<T*>(p);
    }

    // Free everything, ordered after the work queued on the stream. Errors are
    // dropped: after a device fault there is nothing left to free.
    void release() noexcept {
        for (void* p : ptrs_) {
            if (pool_) cudaFreeAsync(p, stream_);
            else       cudaFree(p);
        }
        ptrs_.clear();
    }

    size_t bytes() const { return bytes_; }

private:
    cudaError_t alloc(void** p, size_t bytes) {
        return pool_ ? cudaMallocFromPoolAsync(p, bytes, pool_, stream_) : cudaMalloc(p, bytes);
    }

    cudaStream_t stream_;
    cudaMemPool_t pool_;           // null where the device has no memory pools
    std::vector<void*> ptrs_;
    size_t bytes_ = 0;
};

// Pinned host staging. It grows to the largest pair a Session has seen and is
// kept: it is host memory, and bounded by that one pair.
class HostBuf {
public:
    HostBuf() = default;
    ~HostBuf() { if (p_) cudaFreeHost(p_); }
    HostBuf(const HostBuf&) = delete;
    HostBuf& operator=(const HostBuf&) = delete;
    void* reserve(size_t bytes) {
        if (bytes > cap_) {
            if (p_) PLM_CUDA(cudaFreeHost(p_));
            p_ = nullptr;
            cap_ = 0;              // a failed alloc below must not leave a stale size
            PLM_CUDA(cudaHostAlloc(&p_, bytes, cudaHostAllocDefault));
            cap_ = bytes;
        }
        return p_;
    }
private:
    void* p_ = nullptr;
    size_t cap_ = 0;
};

namespace detail {

struct SlotState {
    int   site  = -1;      /* -1: free */
    bool  fresh = false;   /* pending evaluation is at the initial guess */
    int   iter = 0, ls = 0, stall = 0, head = 0, count = 0;
    int   bank = 0;        /* see SlotArgs::bank */
    float step = 0.0f, f_cur = 0.0f, gtd = 0.0f;
};

/* One pair's batched L-BFGS, over buffers its Session carved from the pair's
 * arena. Throws on any CUDA error; the Session sorts out what it means. */
struct Solver {
    /* borrowed from the Session */
    cudaStream_t   stream = nullptr;
    cublasHandle_t handle = nullptr;
    float*         h_scr = nullptr;        /* pinned, S * (2m + 3) floats */
    int            num_sm = 1;

    /* problem */
    const signed char* MSA = nullptr;      /* (B, N) device */
    const float*       W = nullptr;        /* (B,)   device */
    int   B = 0, N = 0, q = 0, maxeval = 0;
    float lambdaH = 0.0f, lambdaJ = 0.0f;

    /* geometry */
    int m = kMCorr, S = 0, n = 0, K = 0, M = 0;
    int ns = 0;                            /* slot stride of the optimizer vectors: n rounded up to 4 */

    /* device buffers */
    __half *X = nullptr, *JT = nullptr, *V = nullptr, *J = nullptr;
    float  *E = nullptr, *G = nullptr, *h_eval = nullptr;
    float  *x = nullptr, *xt = nullptr, *g = nullptr, *gt = nullptr, *dir = nullptr;
    float  *sbuf = nullptr, *ybuf = nullptr, *rho = nullptr, *gamma = nullptr;
    float  *scratch = nullptr, *parts = nullptr, *gh = nullptr, *acc = nullptr, *scr = nullptr;

    std::vector<SlotState> st;
    Stats tally;

    int scr_width() const { return 2 * m + 3; }
    size_t scratch_floats() const { return (size_t)(3 * S + S * q + 2 * S); }

    int vec_blocks(int entries) const {
        const int want = (ns + kThreads * 4 - 1) / (kThreads * 4);
        const int cap  = std::max(1, (num_sm * 8) / std::max(1, entries));
        return std::max(1, std::min(want, cap));
    }
    dim3 vec_grid(const SlotArgs& a) const { return dim3(vec_blocks(a.n), a.n); }

    void add(SlotArgs& a, int s) const {
        const SlotState& z = st[s];
        const int e = a.n++;
        a.slot[e] = s; a.site[e] = z.site; a.count[e] = z.count;
        a.head[e] = z.head; a.step[e] = z.step; a.bank[e] = z.bank;
    }

    void evaluate(int n_active);
    void move_slot(int src, int dst);
    void optimize();
};

inline void Solver::evaluate(int n_active) {
    SlotArgs ev{};
    for (int s = 0; s < n_active; ++s) add(ev, s);
    const int Me = round_up(q * n_active, 8);
    const float one = 1.0f, zero = 0.0f;

    PLM_CUDA(cudaMemsetAsync(scratch, 0, scratch_floats() * sizeof(float), stream));
    const int pack_x = std::max(1, std::min((N * q + 31) / 32, (num_sm * 8) / std::max(1, ev.n)));
    k_pack<<<dim3(pack_x, ev.n), kThreads, 0, stream>>>(ev, x, xt, JT, h_eval, parts, n, q, K, ns);

    /* GEMM 1 (col-major view): E^T (Me × B) = JT^T (Me × K) · X^T (K × B).
     * Both operands K-major ("TN"). */
    PLM_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, Me, B, K,
                            &one, JT, CUDA_R_16F, K, X, CUDA_R_16F, K,
                            &zero, E, CUDA_R_32F, M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

    const int vgx = std::max(1, std::min((B + kVgradWarps - 1) / kVgradWarps,
                                         (num_sm * 16) / std::max(1, ev.n)));
    k_vgrad<<<dim3(vgx, ev.n), dim3(32, kVgradWarps), 0, stream>>>(
        ev, E, h_eval, MSA, W, V, parts, gh, B, N, q, M);

    /* GEMM 2 (col-major view): G^T (Me × K) = V^T (Me × B) · X (B × K) */
    PLM_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, Me, K, B,
                            &one, V, CUDA_R_16F, M, X, CUDA_R_16F, K,
                            &zero, G, CUDA_R_32F, M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

    k_unpack<<<vec_grid(ev), kThreads, 0, stream>>>(
        ev, G, x, xt, h_eval, gh, g, gt, n, q, M, lambdaH, lambdaJ, ns);
}

/* Move an active slot's optimizer state into a free slot so active slots stay
 * contiguous and the GEMMs only span live columns once the queue drains.
 * g_trial is not moved: the next evaluation overwrites it. */
inline void Solver::move_slot(int src, int dst) {
    const size_t v = (size_t)ns * sizeof(float), h = (size_t)m * v;
    float* vecs[] = {x, xt, g, dir};
    for (float* p : vecs)
        PLM_CUDA(cudaMemcpyAsync(p + (size_t)dst * ns, p + (size_t)src * ns, v,
                                 cudaMemcpyDeviceToDevice, stream));
    PLM_CUDA(cudaMemcpyAsync(sbuf + (size_t)dst * m * ns, sbuf + (size_t)src * m * ns, h,
                             cudaMemcpyDeviceToDevice, stream));
    PLM_CUDA(cudaMemcpyAsync(ybuf + (size_t)dst * m * ns, ybuf + (size_t)src * m * ns, h,
                             cudaMemcpyDeviceToDevice, stream));
    PLM_CUDA(cudaMemcpyAsync(rho + dst * m, rho + src * m, m * sizeof(float),
                             cudaMemcpyDeviceToDevice, stream));
    PLM_CUDA(cudaMemcpyAsync(gamma + dst, gamma + src, sizeof(float),
                             cudaMemcpyDeviceToDevice, stream));
    st[dst] = st[src];
    st[src] = SlotState();
}

/* All N sites, S at a time; site r's couplings end up in row r of J. */
inline void Solver::optimize() {
    tally = Stats{};
    tally.slots = S;
    const int sw = scr_width();
    {
        const long long total = (long long)B * K;
        const int blocks = (int)std::min<long long>((total + 255) / 256, 65535LL * 16);
        k_onehot<<<blocks, 256, 0, stream>>>(MSA, X, B, N, q, K);
    }

    /* ---- Fill the slots with the first sites ---- */
    st.assign(S, SlotState());
    int next_site = 0, n_active = 0;
    {
        SlotArgs la{};
        for (int s = 0; s < S && next_site < N; ++s, ++n_active) {
            st[s].site = next_site++;
            st[s].fresh = true;
            add(la, s);
        }
        k_load<<<vec_grid(la), kThreads, 0, stream>>>(la, x, xt, ns);
    }

    std::vector<int> finished;
    while (n_active > 0) {
        /* ---- 1. f and ∇f at the pending trial point of every active slot ---- */
        evaluate(n_active);
        PLM_CUDA(cudaMemcpyAsync(h_scr, parts, (size_t)3 * n_active * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream));
        PLM_CUDA(cudaGetLastError());
        PLM_CUDA(cudaStreamSynchronize(stream));
        ++tally.rounds;
        tally.evals += n_active;

        /* ---- 2. Armijo test / first-evaluation bookkeeping, per site ---- */
        SlotArgs accept{}, need_dir{}, trial{};
        finished.clear();
        for (int s = 0; s < n_active; ++s) {
            SlotState& z = st[s];
            const float* p = h_scr + 3 * s;
            const float f = p[0] + (lambdaH * p[1] + 0.5f * lambdaJ * p[2]);
            if (z.fresh) {
                z.fresh = false;
                z.f_cur = f;
                z.bank ^= 1;                    /* commit g <- g_trial (x == x_trial) */
                if (maxeval <= 0) { finished.push_back(s); ++tally.max_iter; }
                else              add(need_dir, s);
                continue;
            }
            if (f <= z.f_cur + kArmijoC1 * z.step * z.gtd) {
                add(accept, s);                 /* writes ring slot z.head */
                z.bank ^= 1;                    /* commit x <- x_trial, g <- g_trial */
                z.head = (z.head + 1) % m;
                if (z.count < m) ++z.count;
                const float f_prev = z.f_cur;
                z.f_cur = f;
                ++z.iter;
                if (fabsf(f_prev - z.f_cur) <= kEpsF * (fabsf(z.f_cur) + 1.0f)) ++z.stall;
                else                                                            z.stall = 0;
                if (z.stall >= kStallPatience) { finished.push_back(s); ++tally.stalled; }
                else if (z.iter >= maxeval)    { finished.push_back(s); ++tally.max_iter; }
                else                           add(need_dir, s);
            } else {
                z.step *= kBacktrack;
                ++z.ls;
                if (z.step < kEpsX || z.ls >= kMaxLineEval) { finished.push_back(s); ++tally.ls_failed; }
                else                                        add(trial, s);
            }
        }

        if (accept.n) {
            k_accept<<<vec_grid(accept), kThreads, 0, stream>>>(
                accept, x, xt, g, gt, sbuf, ybuf, acc, ns, m);  /* old bank */
            k_curvature<<<1, kMaxSlots, 0, stream>>>(accept, acc, rho, gamma, m);
        }

        /* ---- 3. New search direction for sites starting an iteration ---- */
        if (need_dir.n) {
            PLM_CUDA(cudaMemsetAsync(scr, 0, (size_t)S * sw * sizeof(float), stream));
            int cmax = 0;
            for (int e = 0; e < need_dir.n; ++e) cmax = std::max(cmax, need_dir.count[e]);
            for (int t = 0; t <= 2 * cmax; ++t)
                k_twoloop<<<vec_grid(need_dir), kThreads, 0, stream>>>(
                    need_dir, t, g, gt, x, xt, dir, sbuf, ybuf, rho, gamma, scr, ns, m);
            PLM_CUDA(cudaMemcpyAsync(h_scr, scr, (size_t)n_active * sw * sizeof(float),
                                     cudaMemcpyDeviceToHost, stream));
            PLM_CUDA(cudaGetLastError());
            PLM_CUDA(cudaStreamSynchronize(stream));

            SlotArgs fallback{};
            for (int e = 0; e < need_dir.n; ++e) {
                const int s = need_dir.slot[e];
                SlotState& z = st[s];
                const float* sc = h_scr + (size_t)s * sw;
                const float gg = sc[2 * m], ginf = sc[2 * m + 1];
                if (ginf <= kEpsG) { finished.push_back(s); ++tally.converged; continue; }
                float gtd = (z.count > 0) ? sc[2 * m + 2] : -gg;
                const bool fb = !(gtd < 0.0f);
                if (fb) { add(fallback, s); gtd = -gg; }
                z.gtd  = gtd;
                z.step = (z.iter == 0) ? fminf(1.0f, 1.0f / (ginf + 1e-12f)) : 1.0f;
                z.ls   = 0;
                /* with history the last two-loop pass already wrote x + dir */
                if (z.count == 0 || fb) add(trial, s);
            }
            if (fallback.n)
                k_neg_g<<<vec_grid(fallback), kThreads, 0, stream>>>(fallback, dir, g, gt, ns);
        }
        if (trial.n)
            k_trial<<<vec_grid(trial), kThreads, 0, stream>>>(trial, x, xt, dir, ns);

        /* ---- 4. Retire finished sites, refill from the queue, compact ---- */
        if (!finished.empty()) {
            std::sort(finished.begin(), finished.end());
            SlotArgs done{}, load{};
            for (int s : finished) {
                add(done, s);
                tally.iterations += st[s].iter;
                tally.f_sum += st[s].f_cur;
            }
            k_store<<<vec_grid(done), kThreads, 0, stream>>>(done, x, xt, J, n, q, ns);
            for (int s : finished) {
                st[s] = SlotState();
                if (next_site < N) {
                    st[s].site = next_site++;
                    st[s].fresh = true;
                    add(load, s);
                }
            }
            if (load.n)
                k_load<<<vec_grid(load), kThreads, 0, stream>>>(load, x, xt, ns);

            int lo = 0, hi = n_active - 1;
            for (;;) {
                while (lo <= hi && st[lo].site >= 0) ++lo;
                while (hi > lo && st[hi].site < 0) --hi;
                if (lo >= hi) break;
                move_slot(hi, lo);
            }
            n_active = 0;
            while (n_active < S && st[n_active].site >= 0) ++n_active;
        }
    }
    PLM_CUDA(cudaGetLastError());
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Session: holds the CUDA context, a stream, the cuBLAS handle and the host
// staging. Device memory is taken per pair from the device's pool (PairArena).
// Not thread-safe: give each worker thread its own. Sessions in one process
// share the device's primary context, so a pool of them costs handles and
// staging buffers, not contexts.
// ---------------------------------------------------------------------------

class Session {
public:
    // l2_share: the part of the device's persisting L2 this Session may pin;
    // 1 / (Sessions on the device) keeps them from evicting each other's.
    explicit Session(int device = 0, double l2_share = 1.0)
        : device_(device), l2_share_(std::min(1.0, std::max(0.0, l2_share))) {
        PLM_CUDA(cudaSetDevice(device));
        PLM_CUDA(cudaFree(nullptr));                    // force context creation
        PLM_CUBLAS(cublasCreate(&handle_));
        PLM_CUDA(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        PLM_CUBLAS(cublasSetStream(handle_, stream_));
        props_ = device_props(device);
        cudaDeviceProp prop{};
        PLM_CUDA(cudaGetDeviceProperties(&prop, device));
        name_ = prop.name;

        pool_ = DeviceShared::join(device_, l2_set_aside(props_));
        joined_ = true;
    }
    ~Session() {
        cudaSetDevice(device_);
        if (stream_) cudaStreamSynchronize(stream_);
        if (joined_) DeviceShared::leave(device_);
        if (handle_) cublasDestroy(handle_);
        if (stream_) cudaStreamDestroy(stream_);
    }
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;

    const std::string& device_name() const { return name_; }
    int device() const { return device_; }
    double l2_share() const { return l2_share_; }
    const Stats& last_stats() const { return stats_; }
    // Device memory the last successful pair took, in bytes.
    size_t last_pair_bytes() const { return last_pair_bytes_; }

    // Sites a pair optimises in lockstep: site_batch, capped by N, by the slot
    // table, and by how many sites' search directions fit in this Session's
    // share of the persisting L2 set-aside -- past that the two-loop recursion
    // runs from DRAM and extra sites mostly add traffic -- but never below 8
    // sites, where the GEMMs get narrow.
    //
    // Deep alignments are the exception: each round streams the (B, N*q)
    // one-hot X through both GEMMs once for all S sites, so for B >= kDeepB
    // the GEMMs gain more from width than the two-loop loses to the L2, and
    // site_batch is kept. Measured on an H100 (helpy55, site_batch 24): 1.12x
    // at B = 7.5k up to 1.32x at B = 10-24k; 0.89x at B = 2.4k.
    static constexpr int kDeepB = 5000;

    static int slots(int device, int B, int N, int q, int site_batch, double l2_share) {
        int S = std::max(1, std::min({site_batch, N, detail::kMaxSlots}));
        const size_t set_aside = l2_set_aside(device_props(device));
        if (set_aside > 0 && B < kDeepB) {
            const double share = std::min(1.0, std::max(0.0, l2_share));
            const size_t n = (size_t)q + (size_t)N * q * q;
            const int fit = (int)(share * (double)set_aside / (double)(n * sizeof(float)));
            S = std::min(S, std::max(8, fit));
        }
        return S;
    }

    // Device memory one (B, N) pair takes, in bytes: what the caller reserves
    // for it before dispatch. Every buffer run_pair allocates is counted; the
    // slack covers the pool's rounding. last_pair_bytes() reports what a pair
    // really took.
    static size_t device_bytes(int device, int B, int N, int q, int site_batch, double l2_share) {
        const int S = slots(device, B, N, q, site_batch, l2_share);
        const int m = detail::kMCorr;
        const size_t n = (size_t)q + (size_t)N * q * q;
        const size_t K = (size_t)detail::round_up(N * q, 8);
        const size_t M = (size_t)detail::round_up(q * S, 8);
        const size_t Nqq = (size_t)N * q * q;
        return (size_t)B * N + (size_t)B * sizeof(float)                // alignment, weights
             + (size_t)B * K * sizeof(__half)                           // X
             + K * M * (sizeof(__half) + sizeof(float))                 // JT, G
             + (size_t)B * M * (sizeof(__half) + sizeof(float))         // V, E
             + (size_t)S * ((n + 3) & ~(size_t)3) * sizeof(float) * (5 + 2 * m)  // x, xt, g, gt, dir, s, y (padded)
             + (size_t)S * (2 * q + 3 * m + 9) * sizeof(float)          // h, rho, gamma, scratch
             + (size_t)N * Nqq * sizeof(__half)                         // J
             + std::min(Nqq * N, kChunk) * sizeof(float)                // couplings_into staging
             + (size_t)N * N * sizeof(float)                            // FN
             + (size_t)24 * (2u << 20)                                  // pool rounding
             + (64u << 20);
    }

    // (B, N) int8 alignment (states 0 .. q-1) and (B,) weights, already divided
    // by Beff -> (N, N) float32 score: the APC-corrected Frobenius norm of the
    // Ising-gauged couplings, gap state 0 excluded, exactly what
    // compute_score(applyIsingGauge(J)) returns for the same couplings.
    void score_into(const int8_t* MSA, const double* W, int B, int N, int q,
                    const Config& cfg, float* score_out) {
        run(MSA, W, B, N, q, cfg, score_out, nullptr);
    }

    // The raw couplings instead: (N, N*q*q) float32, row r = site r's J[i, a, c]
    // at (i*q + a)*q + c, as plm_opt_site --use-gpu publishes them.
    void couplings_into(const int8_t* MSA, const double* W, int B, int N, int q,
                        const Config& cfg, float* J_out) {
        run(MSA, W, B, N, q, cfg, nullptr, J_out);
    }

    // The score of couplings J -- (N, N*q*q) as couplings_into returns them,
    // rounded to fp16 on the way in as the optimiser keeps them -- by the same
    // device path score_into takes after the optimisation.
    void score_couplings_into(const float* J, int N, int q, float* score_out);

    // Sequence reweighting neighbour counts, (B, N) alignment -> (B,) counts:
    // what reweightNeighbourCounts returns for the same matrix and
    // identical_threshold (x * N); the weights are 1 / counts. Takes no more
    // device memory than device_bytes, so it can run inside the reservation a
    // pair is admitted with, before the pair itself.
    void neighbour_counts_into(const int8_t* MSA, int B, int N, double identical_threshold,
                               int32_t* counts_out);

    // Hand back to the driver whatever device memory the pool still holds.
    // Each call does that as it finishes already; this is for after a failure.
    void trim() {
        PLM_CUDA(cudaSetDevice(device_));
        PLM_CUDA(cudaStreamSynchronize(stream_));
        if (pool_) PLM_CUDA(cudaMemPoolTrimTo(pool_, 0));
    }

private:
    static constexpr size_t kChunk = (size_t)16 << 20;   // floats per staging copy (64 MiB)

    template <class F> void with_arena(F&& body);
    void run(const int8_t* MSA, const double* W, int B, int N, int q, const Config& cfg,
             float* score_out, float* J_out);
    void run_pair(PairArena& arena, const int8_t* MSA, const double* W, int B, int N, int q,
                  const Config& cfg, float* score_out, float* J_out);
    void finish_score(PairArena& arena, const __half* J, int N, int q, float* score_out);
    void set_l2_window(void* base, size_t bytes);
    void clear_l2_window() noexcept;
    bool settle() noexcept;

    int device_ = 0;
    double l2_share_ = 1.0;
    cublasHandle_t handle_ = nullptr;
    cudaStream_t stream_ = nullptr;
    cudaMemPool_t pool_ = nullptr;
    DeviceProps props_;
    std::string name_;
    bool joined_ = false;
    Stats stats_;
    size_t last_pair_bytes_ = 0;
    HostBuf hIn_, hScr_, hOut_;
};

// After a failed pair: wait out whatever it left queued and drop a non-sticky
// error, so it does not resurface at the next pair's launch check. Returns
// false when the context itself is gone.
inline bool Session::settle() noexcept {
    cudaSetDevice(device_);
    const bool ok = cudaStreamSynchronize(stream_) == cudaSuccess;
    cudaGetLastError();
    return ok;
}

// Pin [base, base + bytes) in this Session's share of the persisting L2 set-
// aside for the kernels on its stream: the search directions, which every
// two-loop step reads and rewrites while streaming one history vector past
// them. slots() sized the pair so they fit. Best effort: without set-aside L2
// the pair just runs unpinned.
inline void Session::set_l2_window(void* base, size_t bytes) {
    if (props_.max_persist == 0 || props_.max_window == 0 || l2_share_ <= 0.0) return;
    const size_t share = (size_t)(l2_share_ * (double)l2_set_aside(props_));
    cudaStreamAttrValue attr{};
    attr.accessPolicyWindow.base_ptr  = base;
    attr.accessPolicyWindow.num_bytes = std::min({bytes, share, props_.max_window});
    attr.accessPolicyWindow.hitRatio  = 1.0f;
    attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
    attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
    if (cudaStreamSetAttribute(stream_, cudaStreamAttributeAccessPolicyWindow, &attr) != cudaSuccess)
        cudaGetLastError();
}

inline void Session::clear_l2_window() noexcept {
    if (props_.max_persist == 0 || props_.max_window == 0) return;
    cudaStreamAttrValue attr{};
    attr.accessPolicyWindow.num_bytes = 0;
    attr.accessPolicyWindow.hitRatio  = 0.0f;
    attr.accessPolicyWindow.hitProp   = cudaAccessPropertyNormal;
    attr.accessPolicyWindow.missProp  = cudaAccessPropertyNormal;
    if (cudaStreamSetAttribute(stream_, cudaStreamAttributeAccessPolicyWindow, &attr) != cudaSuccess)
        cudaGetLastError();
}

// Run `body` with device memory for one call, all of it handed back after.
template <class F>
void Session::with_arena(F&& body) {
    PairArena arena(stream_, pool_);
    try {
        body(arena);
    } catch (const std::exception& ex) {
        clear_l2_window();
        // Free only once the stream that may still use the buffers is idle.
        if (!settle())
            throw DeviceLost(std::string(ex.what()) +
                             " -- the CUDA context is lost; restart the process");
        arena.release();
        if (pool_ && dynamic_cast<const OutOfMemory*>(&ex)) {
            cudaStreamSynchronize(stream_);
            cudaMemPoolTrimTo(pool_, 0);        // a library allocation may need it
        }
        throw;
    }
    last_pair_bytes_ = arena.bytes();
    // The pool returns freed memory to the driver at a synchronisation; make
    // that now, not whenever this Session next runs a pair.
    arena.release();
    cudaStreamSynchronize(stream_);
}

inline void Session::run(const int8_t* MSA, const double* W, int B, int N, int q,
                         const Config& cfg, float* score_out, float* J_out) {
    with_arena([&](PairArena& arena) {
        run_pair(arena, MSA, W, B, N, q, cfg, score_out, J_out);
    });
}

// (N, N) score of the fp16 couplings J on the device: gauged, symmetrised
// Frobenius norms there, then the average product correction here, as
// compute_score applies it.
inline void Session::finish_score(PairArena& arena, const __half* J, int N, int q,
                                  float* score_out) {
    using namespace detail;
    float* FN = arena.get<float>((size_t)N * N);
    PLM_CUDA(cudaMemsetAsync(FN, 0, (size_t)N * N * sizeof(float), stream_));
    k_frobenius<<<dim3((N + kFnWarps - 1) / kFnWarps, N), 32 * kFnWarps, 0, stream_>>>(
        J, FN, N, q, /*gap=*/0);
    float* hFN = static_cast<float*>(hOut_.reserve((size_t)N * N * sizeof(float)));
    PLM_CUDA(cudaMemcpyAsync(hFN, FN, (size_t)N * N * sizeof(float),
                             cudaMemcpyDeviceToHost, stream_));
    PLM_CUDA(cudaGetLastError());
    PLM_CUDA(cudaStreamSynchronize(stream_));
    std::vector<double> mean((size_t)N);
    double total = 0.0;
    for (int r = 0; r < N; ++r) {
        double s = 0.0;
        for (int i = 0; i < N; ++i) s += hFN[(size_t)r * N + i];
        mean[r] = s / (N - 1);
        total += s;
    }
    total /= (double)N * (N - 1);
    for (int r = 0; r < N; ++r)
        for (int i = 0; i < N; ++i)
            score_out[(size_t)r * N + i] =
                (float)(hFN[(size_t)r * N + i] - mean[r] * mean[i] / total);
}

inline void Session::score_couplings_into(const float* J, int N, int q, float* score_out) {
    if (q < 1 || q > 32)
        throw std::invalid_argument("plm_dca_cuda: alphabet size q must satisfy 1 <= q <= 32");
    if (N < 1)
        throw std::invalid_argument("plm_dca_cuda: need N >= 1");
    const size_t total = (size_t)N * N * q * q;
    with_arena([&](PairArena& arena) {
        PLM_CUDA(cudaSetDevice(device_));       // see run_pair
        __half* hJ = static_cast<__half*>(hIn_.reserve(total * sizeof(__half)));
        for (size_t k = 0; k < total; ++k) hJ[k] = __float2half(J[k]);
        __half* dJ = arena.get<__half>(total);
        PLM_CUDA(cudaMemcpyAsync(dJ, hJ, total * sizeof(__half), cudaMemcpyHostToDevice, stream_));
        finish_score(arena, dJ, N, q, score_out);
    });
}

inline void Session::neighbour_counts_into(const int8_t* MSA, int B, int N,
                                           double identical_threshold, int32_t* counts_out) {
    using namespace detail;
    if (N < 1 || B < 1)
        throw std::invalid_argument("plm_dca_cuda: need N >= 1 and B >= 1");
    // An integer threshold the kernel can compare exactly; clamped to what a
    // match count can reach, so it fits an int.
    const double t = std::ceil(identical_threshold);
    const int threshold = t <= 0.0 ? 0 : (t > N ? N + 1 : (int)t);
    const int Nw = (N + 3) / 4;
    const size_t row = 4 * (size_t)Nw;

    with_arena([&](PairArena& arena) {
        PLM_CUDA(cudaSetDevice(device_));       // see run_pair
        // Rows padded to whole words with zeros, which match in every pair and
        // are taken back off by the kernel.
        uint8_t* hSeq = static_cast<uint8_t*>(hIn_.reserve((size_t)B * row));
        for (int b = 0; b < B; ++b) {
            std::memcpy(hSeq + (size_t)b * row, MSA + (size_t)b * N, (size_t)N);
            std::memset(hSeq + (size_t)b * row + N, 0, row - N);
        }
        uint8_t* dSeq = arena.get<uint8_t>((size_t)B * row);
        int* dM = arena.get<int>((size_t)B);
        PLM_CUDA(cudaMemcpyAsync(dSeq, hSeq, (size_t)B * row, cudaMemcpyHostToDevice, stream_));
        k_rw_fill_ones<<<(B + 255) / 256, 256, 0, stream_>>>(dM, B);
        const int ntile = (B + kRwTile - 1) / kRwTile;
        k_rw_similarity<<<dim3(ntile, ntile), dim3(kRwTile, kRwTy), 0, stream_>>>(
            reinterpret_cast<const unsigned int*>(dSeq), dM, B, Nw, (int)(row - N), threshold);
        PLM_CUDA(cudaGetLastError());
        PLM_CUDA(cudaMemcpyAsync(counts_out, dM, (size_t)B * sizeof(int),
                                 cudaMemcpyDeviceToHost, stream_));
        PLM_CUDA(cudaStreamSynchronize(stream_));
    });
}

inline void Session::run_pair(PairArena& arena, const int8_t* MSA, const double* W,
                              int B, int N, int q, const Config& cfg,
                              float* score_out, float* J_out) {
    using namespace detail;
    if (q < 1 || q > 32)
        throw std::invalid_argument("plm_dca_cuda: alphabet size q must satisfy 1 <= q <= 32");
    if (N < 1 || B < 1)
        throw std::invalid_argument("plm_dca_cuda: need N >= 1 and B >= 1");
    if ((size_t)N * q * q + q > (size_t)INT32_MAX / 2)
        throw std::invalid_argument("plm_dca_cuda: N*q*q exceeds the kernels' 32-bit index range");

    // CUDA's current device is per *thread*, and a Session may be handed
    // between threads.
    PLM_CUDA(cudaSetDevice(device_));

    Solver sv;
    sv.stream  = stream_;
    sv.handle  = handle_;
    sv.num_sm  = props_.sm;
    sv.B = B; sv.N = N; sv.q = q;
    sv.maxeval = cfg.maxeval;
    sv.lambdaH = cfg.lambdaH;
    sv.lambdaJ = cfg.lambdaJ;
    sv.S = slots(device_, B, N, q, cfg.site_batch, l2_share_);
    sv.n = q + N * q * q;
    sv.ns = (sv.n + 3) & ~3;
    sv.K = round_up(N * q, 8);
    sv.M = round_up(q * sv.S, 8);
    const size_t Nqq = (size_t)N * q * q;

    // ---- Stage the alignment and the weights. A state outside 0 .. q-1
    // would index past a site's energies on the device -- an illegal address,
    // which takes the context down for every Session -- so it is caught here.
    const size_t msa_bytes = (size_t)B * N;
    const size_t w_off = (msa_bytes + 15) & ~(size_t)15;
    char* hin = static_cast<char*>(hIn_.reserve(w_off + (size_t)B * sizeof(float)));
    std::memcpy(hin, MSA, msa_bytes);
    {
        uint8_t bad = 0;
        const uint8_t lim = (uint8_t)q;
        for (size_t k = 0; k < msa_bytes; ++k) bad |= (uint8_t)((uint8_t)hin[k] >= lim);
        if (bad)
            throw std::invalid_argument("plm_dca_cuda: the alignment holds a state outside 0 .. q-1");
    }
    float* hw = reinterpret_cast<float*>(hin + w_off);
    for (int b = 0; b < B; ++b) hw[b] = static_cast<float>(W[b]);

    signed char* dMSA = arena.get<signed char>(msa_bytes);
    float* dW = arena.get<float>((size_t)B);
    PLM_CUDA(cudaMemcpyAsync(dMSA, hin, msa_bytes, cudaMemcpyHostToDevice, stream_));
    PLM_CUDA(cudaMemcpyAsync(dW, hw, (size_t)B * sizeof(float), cudaMemcpyHostToDevice, stream_));
    sv.MSA = dMSA;
    sv.W   = dW;

    // ---- Solver buffers ----
    const size_t Sn = (size_t)sv.S * sv.ns;
    sv.X      = arena.get<__half>((size_t)B * sv.K);
    sv.JT     = arena.get<__half>((size_t)sv.K * sv.M);
    sv.V      = arena.get<__half>((size_t)B * sv.M);
    sv.E      = arena.get<float>((size_t)B * sv.M);
    sv.G      = arena.get<float>((size_t)sv.K * sv.M);
    sv.h_eval = arena.get<float>((size_t)sv.S * q);
    sv.x      = arena.get<float>(Sn);
    sv.xt     = arena.get<float>(Sn);
    sv.g      = arena.get<float>(Sn);
    sv.gt     = arena.get<float>(Sn);
    sv.dir    = arena.get<float>(Sn);
    sv.sbuf   = arena.get<float>(Sn * sv.m);
    sv.ybuf   = arena.get<float>(Sn * sv.m);
    sv.rho    = arena.get<float>((size_t)sv.S * sv.m);
    sv.gamma  = arena.get<float>((size_t)sv.S);
    sv.scratch = arena.get<float>(sv.scratch_floats());
    sv.scr    = arena.get<float>((size_t)sv.S * sv.scr_width());
    sv.J      = arena.get<__half>((size_t)N * Nqq);
    sv.parts  = sv.scratch;
    sv.gh     = sv.parts + 3 * sv.S;
    sv.acc    = sv.gh + sv.S * q;
    // Pool memory comes back as the last pair left it. The GEMMs read all of
    // JT and V, including padding and the columns of idle slots, so those
    // must at least be finite; everything else is written before it is read.
    PLM_CUDA(cudaMemsetAsync(sv.JT, 0, (size_t)sv.K * sv.M * sizeof(__half), stream_));
    PLM_CUDA(cudaMemsetAsync(sv.V, 0, (size_t)B * sv.M * sizeof(__half), stream_));
    // The float4 kernels also run over each slot's zero pad [n, ns): k_unpack
    // writes only the first n of g_trial, so the pads of g, g_trial and dir
    // start at zero and every element-wise op keeps them there.
    for (float* p : {sv.g, sv.gt, sv.dir})
        PLM_CUDA(cudaMemsetAsync(p, 0, Sn * sizeof(float), stream_));
    sv.h_scr = static_cast<float*>(hScr_.reserve((size_t)sv.S * sv.scr_width() * sizeof(float)));

    set_l2_window(sv.dir, Sn * sizeof(float));
    sv.optimize();
    clear_l2_window();

    if (score_out) finish_score(arena, sv.J, N, q, score_out);
    if (J_out) {
        const size_t total = (size_t)N * Nqq;
        const size_t chunk = std::min(total, kChunk);
        float* dChunk = arena.get<float>(chunk);
        float* hChunk = static_cast<float*>(hOut_.reserve(chunk * sizeof(float)));
        for (size_t off = 0; off < total; off += chunk) {
            const size_t len = std::min(chunk, total - off);
            const int blocks = (int)std::min<size_t>((len + 255) / 256, 65535u * 16);
            k_half_to_float<<<blocks, 256, 0, stream_>>>(sv.J + off, dChunk, len);
            PLM_CUDA(cudaMemcpyAsync(hChunk, dChunk, len * sizeof(float),
                                     cudaMemcpyDeviceToHost, stream_));
            PLM_CUDA(cudaGetLastError());
            PLM_CUDA(cudaStreamSynchronize(stream_));
            std::memcpy(J_out + off, hChunk, len * sizeof(float));
        }
    }
    stats_ = sv.tally;
}

}  // namespace cuda
}  // namespace plm
