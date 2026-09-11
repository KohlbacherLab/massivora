// gauss_dca_cuda.cuh
//
// CUDA port of the GaussDCA covariance-inverse core (see gauss_dca.hpp for the
// notation and the reference CPU implementation). The whole pipeline runs on
// the device: the only host traffic is the (N x B) alignment in and the
// (N x N) score out.
//
//   MSA -> recode -> fi -> pair histograms -> C (lower)      k_recode/k_fi/k_cov_*
//   C -> L                                                 cusolverDnXpotrf
//   L -> R = L^-1                                          trtri (this file)
//   R -> A = R^T R = C^-1                                  lauum (this file)
//   A -> per-block Frobenius norm -> APC -> score          k_score/k_apc
//
// ---------------------------------------------------------------------------
// Why the triangular halves are re-implemented here
//
// cusolverDnXtrtri and cusolverDnSlauum reach only 2.9 and 4.3 TFLOP/s at
// NQ = 10140 on a P100 whose SGEMM measures 8.4, which is 200 ms of the 256 ms
// cusolverDnSpotri spends on an explicit inverse -- the dominant stage.
// The versions below are level-by-level bottom-up formulations that write into
// a *second* buffer, so every level is a set of independent, identically
// shaped GEMMs: the small levels go through one strided-batched call each and
// the large ones through a single big TRMM/SYRK. Out-of-place is what buys the
// freedom to reorder -- the classic in-place algorithms have to alternate
// between the two halves of the matrix and cannot be batched.
//
// The extra NQ x NQ buffer is the price. Device memory is 2 x NQ^2 x
// sizeof(CovScalar), which for the widest alignment in a typical proteome run
// (N ~ 500, NQ ~ 10k) is 0.8 GB of the card's 16.

#pragma once

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace gaussdca {
namespace cuda {

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

inline void cuda_check(cudaError_t e, const char* what, int line) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("gauss_dca_cuda: ") + what + ": " +
                                 cudaGetErrorString(e) + " (line " +
                                 std::to_string(line) + ")");
}
inline void cublas_check(cublasStatus_t s, const char* what, int line) {
    if (s != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string("gauss_dca_cuda: cuBLAS ") + what +
                                 " failed with status " + std::to_string((int)s) +
                                 " (line " + std::to_string(line) + ")");
}
inline void cusolver_check(cusolverStatus_t s, const char* what, int line) {
    if (s != CUSOLVER_STATUS_SUCCESS)
        throw std::runtime_error(std::string("gauss_dca_cuda: cuSOLVER ") + what +
                                 " failed with status " + std::to_string((int)s) +
                                 " (line " + std::to_string(line) + ")");
}
#define GD_CUDA(x)     ::gaussdca::cuda::cuda_check((x), #x, __LINE__)
#define GD_CUBLAS(x)   ::gaussdca::cuda::cublas_check((x), #x, __LINE__)
#define GD_CUSOLVER(x) ::gaussdca::cuda::cusolver_check((x), #x, __LINE__)

// ---------------------------------------------------------------------------
// Tuning constants
// ---------------------------------------------------------------------------

// Diagonal blocks of this size are inverted / squared by one warp each.
static constexpr int GD_BASE = 32;
// Below this width the trtri/lauum levels are uniform and go through
// strided-batched GEMM; above it they are single large TRMM/SYRK calls. The
// factorisation width is padded to a multiple of it.
static constexpr int GD_BOT = 128;
// Warps per block in the pair-frequency kernel: one (i, j) pair each. Four is
// the measured optimum on sm_60 -- past that the extra pairs per block buy less
// than the loss in blocks resident per SM.
#ifndef GD_PAIR_WARPS_OVERRIDE
#define GD_PAIR_WARPS_OVERRIDE 4
#endif
static constexpr int GD_PAIR_WARPS = GD_PAIR_WARPS_OVERRIDE;
// Histogram copies per warp, indexed by lane parity, to halve the same-address
// atomic conflict a conserved column pair produces. Must be a power of two.
#ifndef GD_HIST_COPIES_OVERRIDE
#define GD_HIST_COPIES_OVERRIDE 2
#endif
static constexpr int GD_HIST_COPIES = GD_HIST_COPIES_OVERRIDE;
// Fold repeated (a, b) bins within a lane's eight sequences into one atomic.
// Worth ~4x on sm_60, where shared atomics are slow; re-measure on newer parts,
// where the atomic units are fast enough that the folding can cost more than
// the conflicts it removes.
#ifndef GD_COV_DEDUP
#define GD_COV_DEDUP 1
#endif
// Warps per block in the reduction kernels, which are latency- not atomic-bound.
static constexpr int GD_RED_WARPS = 8;
// Warps per block in the diagonal-block kernels. Each holds one or two
// GD_BASE^2 tiles in shared memory, so two warps is what fits in double.
static constexpr int GD_DIAG_WARPS = 2;

// Panel width for the blocked Cholesky. 1024 measured best at NQ = 11264:
// narrower makes the trailing GEMMs too thin, wider lets the serial
// factorisation of the diagonal block dominate.
static constexpr int GD_POTRF_NB = 1024;

// Width the covariance is factorised in on the device. float32 by default, as
// on the CPU: it halves the two NQ x NQ buffers and its error is far below what
// the float16 scores it feeds into can resolve.
#ifdef GAUSSDCA_CUDA_DOUBLE_PRECISION
using CovScalar = double;
#else
using CovScalar = float;
#endif

// This applies to the float32 score path only. couplings_into() factorises in
// float64 precisely so its result can be compared against the Julia reference
// at 1e-8, and a float32 histogram would put ~2e-7 into it, so that path always
// accumulates in double -- see HistFor.
#ifdef GAUSSDCA_CUDA_DOUBLE_HIST
using HistScalar = double;
#else
using HistScalar = float;
#endif

// Histogram width that goes with a given covariance width.
template <class S> struct HistFor         { using type = HistScalar; };
template <>        struct HistFor<double> { using type = double; };

// ---------------------------------------------------------------------------
// Typed cuBLAS / cuSOLVER wrappers
// ---------------------------------------------------------------------------

inline cublasStatus_t xgemm(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb,
                            int m, int n, int k, float a, const float* A, int lda,
                            const float* B, int ldb, float b, float* C, int ldc) {
    return cublasSgemm(h, ta, tb, m, n, k, &a, A, lda, B, ldb, &b, C, ldc);
}
inline cublasStatus_t xgemm(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb,
                            int m, int n, int k, double a, const double* A, int lda,
                            const double* B, int ldb, double b, double* C, int ldc) {
    return cublasDgemm(h, ta, tb, m, n, k, &a, A, lda, B, ldb, &b, C, ldc);
}
inline cublasStatus_t xgemm_batched(cublasHandle_t h, cublasOperation_t ta,
                                    cublasOperation_t tb, int m, int n, int k, float a,
                                    const float* A, int lda, long long sa,
                                    const float* B, int ldb, long long sb, float b,
                                    float* C, int ldc, long long sc, int batch) {
    return cublasSgemmStridedBatched(h, ta, tb, m, n, k, &a, A, lda, sa, B, ldb, sb,
                                     &b, C, ldc, sc, batch);
}
inline cublasStatus_t xgemm_batched(cublasHandle_t h, cublasOperation_t ta,
                                    cublasOperation_t tb, int m, int n, int k, double a,
                                    const double* A, int lda, long long sa,
                                    const double* B, int ldb, long long sb, double b,
                                    double* C, int ldc, long long sc, int batch) {
    return cublasDgemmStridedBatched(h, ta, tb, m, n, k, &a, A, lda, sa, B, ldb, sb,
                                     &b, C, ldc, sc, batch);
}
inline cublasStatus_t xtrmm(cublasHandle_t h, cublasSideMode_t side, cublasOperation_t op,
                            int m, int n, float a, const float* A, int lda,
                            const float* B, int ldb, float* C, int ldc) {
    return cublasStrmm(h, side, CUBLAS_FILL_MODE_LOWER, op, CUBLAS_DIAG_NON_UNIT,
                       m, n, &a, A, lda, B, ldb, C, ldc);
}
inline cublasStatus_t xtrmm(cublasHandle_t h, cublasSideMode_t side, cublasOperation_t op,
                            int m, int n, double a, const double* A, int lda,
                            const double* B, int ldb, double* C, int ldc) {
    return cublasDtrmm(h, side, CUBLAS_FILL_MODE_LOWER, op, CUBLAS_DIAG_NON_UNIT,
                       m, n, &a, A, lda, B, ldb, C, ldc);
}
inline cublasStatus_t xsyrk(cublasHandle_t h, int n, int k, float a, const float* A,
                            int lda, float b, float* C, int ldc) {
    return cublasSsyrk(h, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, n, k, &a, A, lda, &b, C, ldc);
}
inline cublasStatus_t xsyrk(cublasHandle_t h, int n, int k, double a, const double* A,
                            int lda, double b, double* C, int ldc) {
    return cublasDsyrk(h, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, n, k, &a, A, lda, &b, C, ldc);
}
// fp32 operands, TF32 tensor-core multiply, fp32 accumulator. On sm_90 this is
// 261 TFLOP/s against 36 for plain SGEMM, and unlike the fp16 modes it eats the
// fp32 buffers directly -- no conversion, no shadow copies, and fp32's exponent
// range. TRMM and SYRK do *not* honour it (measured: unchanged at 32 TFLOP/s
// under CUBLAS_TF32_TENSOR_OP_MATH), which is why the triangular stages have to
// be re-expressed as GEMM to use it at all.
inline cublasStatus_t xgemm_tf32(cublasHandle_t h, cublasOperation_t ta,
                                 cublasOperation_t tb, int m, int n, int k,
                                 float alpha, const float* A, int lda,
                                 const float* B, int ldb, float beta,
                                 float* C, int ldc) {
    return cublasGemmEx(h, ta, tb, m, n, k, &alpha, A, CUDA_R_32F, lda,
                        B, CUDA_R_32F, ldb, &beta, C, CUDA_R_32F, ldc,
                        CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT);
}

// Half precision is only reachable through GEMM: cuBLAS has no half TRMM or
// SYRK, which is what makes the blocked lauum below the only stage that can use
// it without tripling the flop count.
inline cublasStatus_t hgemm_16acc(cublasHandle_t h, cublasOperation_t ta,
                                  cublasOperation_t tb, int m, int n, int k,
                                  const __half* A, int lda, const __half* B, int ldb,
                                  __half* C, int ldc) {
    const __half one = __float2half(1.f), zero = __float2half(0.f);
    return cublasHgemm(h, ta, tb, m, n, k, &one, A, lda, B, ldb, &zero, C, ldc);
}
inline cusolverStatus_t xpotrf_buffer(cusolverDnHandle_t h, int n, float* A, int lda, int* lw) {
    return cusolverDnSpotrf_bufferSize(h, CUBLAS_FILL_MODE_LOWER, n, A, lda, lw);
}
inline cusolverStatus_t xpotrf_buffer(cusolverDnHandle_t h, int n, double* A, int lda, int* lw) {
    return cusolverDnDpotrf_bufferSize(h, CUBLAS_FILL_MODE_LOWER, n, A, lda, lw);
}
inline cusolverStatus_t xpotrf(cusolverDnHandle_t h, int n, float* A, int lda,
                               float* work, int lw, int* info) {
    return cusolverDnSpotrf(h, CUBLAS_FILL_MODE_LOWER, n, A, lda, work, lw, info);
}
inline cusolverStatus_t xpotrf(cusolverDnHandle_t h, int n, double* A, int lda,
                               double* work, int lw, int* info) {
    return cusolverDnDpotrf(h, CUBLAS_FILL_MODE_LOWER, n, A, lda, work, lw, info);
}

// ---------------------------------------------------------------------------
// Stage 1 -- alignment recode and single-site frequencies
// ---------------------------------------------------------------------------

// [0, Q-1] for the real symbols, Q for anything gap-like; the padded tail of
// each row is filled with Q so the vectorised inner loop can ignore it.
__global__ void k_recode(const int8_t* __restrict__ MSA, int N, int B, int Bp, int Q,
                         uint8_t* __restrict__ MSAc) {
    const long total = (long)N * Bp;
    for (long t = blockIdx.x * (long)blockDim.x + threadIdx.x; t < total;
         t += (long)gridDim.x * blockDim.x) {
        const int k = (int)(t % Bp);
        const long i = t / Bp;
        int v = Q;
        if (k < B) {
            const int a = MSA[i * B + k];
            v = (a >= 1 && a <= Q) ? a - 1 : Q;
        }
        MSAc[t] = (uint8_t)v;
    }
}

// Weight vector in both widths, zero-padded to Bp.
__global__ void k_weights(const double* __restrict__ W, int B, int Bp,
                          double* __restrict__ Wd, float* __restrict__ Wf) {
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < Bp;
         k += gridDim.x * blockDim.x) {
        const double w = (k < B) ? W[k] : 0.0;
        Wd[k] = w;
        Wf[k] = (float)w;
    }
}

// fi(i, a) = sum_k W[k] [MSAc(i, k) == a] / Beff, then the pseudocount mix.
// One block per site; per-warp private bins keep the atomics off one address.
__global__ void k_fi(const uint8_t* __restrict__ MSAc, int Bp, int Q,
                     const double* __restrict__ Wd, double invBeff,
                     double one_minus_lambda, double unif_i,
                     double* __restrict__ fi) {
    extern __shared__ double sfi[];                 // nwarps * (Q + 1)
    const int QP = Q + 1;
    const int nwarps = blockDim.x >> 5;
    const int warp = threadIdx.x >> 5;
    const int i = blockIdx.x;
    for (int t = threadIdx.x; t < nwarps * QP; t += blockDim.x) sfi[t] = 0.0;
    __syncthreads();
    double* acc = sfi + warp * QP;
    const uint8_t* zc = MSAc + (size_t)i * Bp;
    for (int k = threadIdx.x; k < Bp; k += blockDim.x)
        atomicAdd(&acc[zc[k]], Wd[k]);
    __syncthreads();
    for (int a = threadIdx.x; a < Q; a += blockDim.x) {
        double s = 0.0;
        for (int w = 0; w < nwarps; ++w) s += sfi[w * QP + a];
        fi[(size_t)i * Q + a] = one_minus_lambda * (s * invBeff) + unif_i;
    }
}

// ---------------------------------------------------------------------------
// Stage 2 -- pair frequencies fused with the covariance store
// ---------------------------------------------------------------------------

// One warp per (i, j) pair, GD_PAIR_WARPS pairs per block sharing the same i so
// row i is read once into cache and reused.
//
// This stage is pinned to the P100's shared-memory atomic rate, and on real
// alignments the binding constraint is *same-address* conflict, not bandwidth:
// a conserved column pair puts 38% of the sequences in one (a, b) bin at the
// median and 79% at the 90th percentile, so a naive warp serialises a dozen or
// more lanes onto one counter. Two things fix that, and between them they take
// the kernel from 368 ms to 69 ms on a 561 x 31000 pair:
//
//   * each lane reads eight sequences at a time and folds repeats among them
//     into one update before touching shared memory -- neighbouring rows of a
//     real MSA are often near-duplicates, so this alone removes ~4x;
//   * the histogram is replicated per lane parity, halving what is left.
//
// Full ballot-based warp aggregation was tried and is slower: it costs a
// shuffle reduction per distinct bin, which outweighs the conflicts it removes.
// Staging W and row i in shared memory (2/3 less load traffic) does not move
// the number either -- with the atomics replaced by plain adds the same loop
// runs 5x faster, so traffic is not what this kernel is waiting on.
//
// Gap-involving pairs are dropped before the atomic, which is why only Q^2 and
// not (Q+1)^2 bins are needed.
template <class T> struct Vec4;
template <> struct Vec4<float>  { using type = float4; };
template <> struct Vec4<double> { using type = double4; };

template <class HIST, class S>
__global__ void k_cov_pairs(const uint8_t* __restrict__ MSAc, int N, int Bp, int Q,
                            const HIST* __restrict__ Wh,
                            const int2* __restrict__ groups,
                            const double* __restrict__ fi, long np,
                            double one_minus_lambda, double invBeff, double unif_ij,
                            S* __restrict__ C) {
    using WVec = typename Vec4<HIST>::type;
    extern __shared__ char smem_raw[];
    HIST* hist = reinterpret_cast<HIST*>(smem_raw);
    const int QQ = Q * Q;
    const int nwarps = blockDim.x >> 5;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    for (int t = threadIdx.x; t < nwarps * GD_HIST_COPIES * QQ; t += blockDim.x)
        hist[t] = (HIST)0;
    __syncthreads();

    const int i = groups[blockIdx.x].x;
    const int j = groups[blockIdx.x].y + warp;
    HIST* mine = hist + (warp * GD_HIST_COPIES + (lane & (GD_HIST_COPIES - 1))) * QQ;

    if (j < N) {
        // Bp is a multiple of 16 and the tail is padded with the gap symbol, so
        // eight symbols and eight weights can be pulled per lane per step.
        const uint2* zi = reinterpret_cast<const uint2*>(MSAc + (size_t)i * Bp);
        const uint2* zj = reinterpret_cast<const uint2*>(MSAc + (size_t)j * Bp);
        const int n8 = Bp >> 3;
        for (int k = lane; k < n8; k += 32) {
            const uint2 A = zi[k];
            const uint2 B = zj[k];
            const WVec* w = reinterpret_cast<const WVec*>(Wh + ((size_t)k << 3));
            const WVec w0 = w[0], w1 = w[1];
            int idx[8];
            HIST wv[8];
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                const int a = (c < 4) ? ((A.x >> (8 * c)) & 255)
                                      : ((A.y >> (8 * (c - 4))) & 255);
                const int b = (c < 4) ? ((B.x >> (8 * c)) & 255)
                                      : ((B.y >> (8 * (c - 4))) & 255);
                idx[c] = (a < Q && b < Q) ? (a * Q + b) : -1;
            }
            wv[0] = w0.x; wv[1] = w0.y; wv[2] = w0.z; wv[3] = w0.w;
            wv[4] = w1.x; wv[5] = w1.y; wv[6] = w1.z; wv[7] = w1.w;
#if GD_COV_DEDUP
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                if (idx[c] < 0) continue;
                HIST acc = wv[c];
#pragma unroll
                for (int d = c + 1; d < 8; ++d)
                    if (idx[d] == idx[c]) { acc += wv[d]; idx[d] = -1; }
                atomicAdd(&mine[idx[c]], acc);
            }
#else
#pragma unroll
            for (int c = 0; c < 8; ++c)
                if (idx[c] >= 0) atomicAdd(&mine[idx[c]], wv[c]);
#endif
        }
    }
    __syncthreads();

    if (j < N) {
        // Lower-triangle home of block (i < j): rows j*Q + b, columns i*Q + a.
        const HIST* base_h = hist + (size_t)warp * GD_HIST_COPIES * QQ;
        const double* fi_i = fi + (size_t)i * Q;
        const double* fi_j = fi + (size_t)j * Q;
        S* base = C + (size_t)i * Q * np + (size_t)j * Q;
        for (int t = lane; t < QQ; t += 32) {
            double n = 0.0;
            for (int r = 0; r < GD_HIST_COPIES; ++r) n += (double)base_h[r * QQ + t];
            const int a = t / Q, b = t - a * Q;
            const double v = one_minus_lambda * (n * invBeff) + unif_ij
                             - fi_j[b] * fi_i[a];
            base[(size_t)a * np + b] = (S)v;
        }
    }
}

// Diagonal blocks: zero off-diagonal, fi on the diagonal, minus fi*fi^T.
template <class S>
__global__ void k_cov_diag(int N, int Q, const double* __restrict__ fi, long np,
                           S* __restrict__ C) {
    const int i = blockIdx.x;
    const double* f = fi + (size_t)i * Q;
    S* base = C + (size_t)i * Q * np + (size_t)i * Q;
    for (int t = threadIdx.x; t < Q * Q; t += blockDim.x) {
        const int a = t / Q, b = t - a * Q;
        if (b < a) continue;                       // lower triangle of the block
        base[(size_t)a * np + b] = (S)((a == b ? f[a] : 0.0) - f[b] * f[a]);
    }
}

// Identity on the padding rows/columns so the factorisation ignores them.
template <class S>
__global__ void k_pad_identity(long NQ, long np, S* __restrict__ C) {
    for (long c = blockIdx.x; c < np; c += gridDim.x) {
        const long r0 = (c < NQ) ? NQ : c;
        for (long r = r0 + threadIdx.x; r < np; r += blockDim.x)
            C[c * np + r] = (S)((r == c) ? 1 : 0);
    }
}

// ---------------------------------------------------------------------------
// Stage 3 -- triangular inverse (R = L^-1) and its square (A = R^T R)
// ---------------------------------------------------------------------------

// R[b] = inv(L[b]) for every aligned GD_BASE x GD_BASE diagonal block, one warp
// each. Thread j owns column j, so the forward substitution it runs reads only
// its own column of the partial result -- no cross-lane dependency.
template <class S>
__global__ void k_trtri_diag(const S* __restrict__ L, long np, long nblk,
                             S* __restrict__ R) {
    __shared__ S sl[GD_DIAG_WARPS][GD_BASE * (GD_BASE + 1)];
    __shared__ S sr[GD_DIAG_WARPS][GD_BASE * (GD_BASE + 1)];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const long blk = (long)blockIdx.x * GD_DIAG_WARPS + warp;
    if (blk >= nblk) return;
    const long off = blk * GD_BASE;
    S* l = sl[warp];
    S* r = sr[warp];
    const int ld = GD_BASE + 1;
    // column-major tile with a padded leading dimension
    for (int c = 0; c < GD_BASE; ++c) l[c * ld + lane] = L[(off + c) * np + off + lane];
    __syncwarp();
    const int j = lane;
    r[j * ld + j] = (S)1 / l[j * ld + j];
    for (int i = j + 1; i < GD_BASE; ++i) {
        S s = (S)0;
        for (int k = j; k < i; ++k) s += l[k * ld + i] * r[j * ld + k];
        r[j * ld + i] = -s / l[i * ld + i];
    }
    __syncwarp();
    for (int c = 0; c < GD_BASE; ++c)
        if (lane >= c) R[(off + c) * np + off + lane] = r[c * ld + lane];
}

// A[b] = R[b]^T R[b] (lower triangle) for every aligned diagonal block.
template <class S>
__global__ void k_lauum_diag(const S* __restrict__ R, long np, long nblk,
                             S* __restrict__ A) {
    __shared__ S sr[GD_DIAG_WARPS][GD_BASE * (GD_BASE + 1)];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const long blk = (long)blockIdx.x * GD_DIAG_WARPS + warp;
    if (blk >= nblk) return;
    const long off = blk * GD_BASE;
    S* r = sr[warp];
    const int ld = GD_BASE + 1;
    for (int c = 0; c < GD_BASE; ++c)
        r[c * ld + lane] = (lane >= c) ? R[(off + c) * np + off + lane] : (S)0;
    __syncwarp();
    const int j = lane;
    for (int i = j; i < GD_BASE; ++i) {
        S s = (S)0;
        for (int k = i; k < GD_BASE; ++k) s += r[i * ld + k] * r[j * ld + k];
        A[(off + j) * np + off + i] = s;
    }
}

// ---------------------------------------------------------------------------
// Half-precision support for the blocked lauum
// ---------------------------------------------------------------------------

template <class S>
__global__ void k_to_half(const S* __restrict__ src, __half* __restrict__ dst,
                          long n, float scale) {
    for (long t = blockIdx.x * (long)blockDim.x + threadIdx.x; t < n;
         t += (long)gridDim.x * blockDim.x)
        dst[t] = __float2half((float)src[t] * scale);
}

// max_j ||R(:, j)||^2, which is the largest diagonal entry A can take. Used to
// pick a scale that keeps the fp16 accumulator away from its 65504 ceiling: a
// badly conditioned pair can otherwise overflow it silently.
template <class S>
__global__ void k_max_colnorm2(const S* __restrict__ R, long n, long ld,
                               float* __restrict__ out) {
    __shared__ float red[256];
    for (long c = blockIdx.x; c < n; c += gridDim.x) {
        double s = 0.0;
        for (long r = c + threadIdx.x; r < n; r += blockDim.x) {
            const double v = (double)R[c * ld + r];
            s += v * v;
        }
        red[threadIdx.x] = (float)s;
        __syncthreads();
        for (int o = blockDim.x >> 1; o; o >>= 1) {
            if (threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o];
            __syncthreads();
        }
        if (threadIdx.x == 0) atomicMax((int*)out, __float_as_int(red[0]));
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Stage 4 -- Frobenius norm of the gauge-projected blocks, then APC
// ---------------------------------------------------------------------------

// One warp per (i, j) pair, i > j; lane b holds column b of the Q x Q block in
// registers, so the row means come out of a shuffle reduction and the block is
// read exactly once. The register array is walked with a constant trip count so
// it stays in registers instead of spilling to local memory; Q <= 32 is checked
// by the caller.
template <class S>
__global__ void k_score(const S* __restrict__ A, int N, int Q, long np,
                        double* __restrict__ FN) {
    const int i = blockIdx.x;
    const int nwarps = blockDim.x >> 5;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const double invQ = 1.0 / Q;
    const unsigned mask = 0xffffffffu;
    const bool live = (lane < Q);

    for (int j = warp; j < i; j += nwarps) {
        double col[32];
        double cb = 0.0;
        // block(a, b) = A[i*Q + a, j*Q + b]; lower triangle, column-major, so
        // the a index runs contiguously down column j*Q + b.
        const S* c = A + ((size_t)j * Q + lane) * np + (size_t)i * Q;
#pragma unroll
        for (int a = 0; a < 32; ++a) {
            const double v = (live && a < Q) ? (double)c[a] : 0.0;
            col[a] = v;
            cb += v;
        }
        double grand = cb;
        for (int o = 16; o; o >>= 1) grand += __shfl_down_sync(mask, grand, o);
        grand = __shfl_sync(mask, grand, 0) * invQ * invQ;
        cb *= invQ;                      // mean over a, for this b

        double ss = 0.0;
#pragma unroll
        for (int a = 0; a < 32; ++a) {
            double ra = col[a];
            for (int o = 16; o; o >>= 1) ra += __shfl_down_sync(mask, ra, o);
            ra = __shfl_sync(mask, ra, 0) * invQ;   // mean over b
            if (live && a < Q) {
                const double e = col[a] - ra - cb + grand;
                ss += e * e;
            }
        }
        for (int o = 16; o; o >>= 1) ss += __shfl_down_sync(mask, ss, o);
        if (lane == 0) {
            const double fn = sqrt(ss);
            FN[(size_t)i * N + j] = fn;
            FN[(size_t)j * N + i] = fn;
        }
    }
}

__global__ void k_rowsums(const double* __restrict__ FN, int N,
                          double* __restrict__ Si) {
    __shared__ double red[256];
    const int i = blockIdx.x;
    double s = 0.0;
    for (int j = threadIdx.x; j < N; j += blockDim.x) s += FN[(size_t)i * N + j];
    red[threadIdx.x] = s;
    __syncthreads();
    for (int o = blockDim.x >> 1; o; o >>= 1) {
        if (threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o];
        __syncthreads();
    }
    if (threadIdx.x == 0) Si[i] = red[0];
}

__global__ void k_total(const double* __restrict__ Si, int N, double* __restrict__ out) {
    __shared__ double red[256];
    double s = 0.0;
    for (int i = threadIdx.x; i < N; i += blockDim.x) s += Si[i];
    red[threadIdx.x] = s;
    __syncthreads();
    for (int o = blockDim.x >> 1; o; o >>= 1) {
        if (threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[0] = red[0] * (1.0 - 1.0 / (double)N);
}

__global__ void k_apc(const double* __restrict__ FN, const double* __restrict__ Si,
                      const double* __restrict__ Sa, int N, float* __restrict__ out) {
    const double sa = Sa[0];
    const long total = (long)N * N;
    for (long t = blockIdx.x * (long)blockDim.x + threadIdx.x; t < total;
         t += (long)gridDim.x * blockDim.x) {
        const int i = (int)(t / N), j = (int)(t - (long)i * N);
        out[t] = (float)(FN[t] - (Si[i] * Si[j]) / sa);
    }
}

// Mirror the lower triangle into the upper (couplings entry point only).
template <class S>
__global__ void k_mirror(S* A, long n, long ld) {
    for (long c = blockIdx.x; c < n; c += gridDim.x)
        for (long r = c + 1 + threadIdx.x; r < n; r += blockDim.x)
            A[r * ld + c] = A[c * ld + r];
}

}  // namespace cuda
}  // namespace gaussdca

namespace gaussdca {
namespace cuda {

// ---------------------------------------------------------------------------
// Blocked Cholesky with the trailing update on tensor cores
//
// Right-looking: factor the diagonal block, scale the panel under it, then push
// the panel's outer product into the trailing submatrix. Only that last step is
// large enough to matter, and it is the only one expressible as GEMM -- TRSM
// does not honour TF32 any more than TRMM does.
//
// The trailing update is issued as strip GEMMs down the block columns rather
// than one square GEMM, which keeps it at n^3/6; a square GEMM would compute
// the symmetric half twice and hand back most of the speed-up.
// ---------------------------------------------------------------------------

inline void potrf_blocked_tf32(cublasHandle_t hb, cusolverDnHandle_t hs, float* A,
                               long n, long lda, float* work, int lwork, int* info) {
    for (long k = 0; k < n; k += GD_POTRF_NB) {
        const int kb = (int)std::min((long)GD_POTRF_NB, n - k);
        const int m = (int)(n - k - kb);
        float* Akk = A + k * lda + k;
        GD_CUSOLVER(cusolverDnSpotrf(hs, CUBLAS_FILL_MODE_LOWER, kb, Akk, (int)lda,
                                     work, lwork, info));
        if (m <= 0) continue;
        float* A21 = A + k * lda + k + kb;             // (m x kb), below the block
        const float one = 1.f;
        GD_CUBLAS(cublasStrsm(hb, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                              CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, m, kb, &one,
                              Akk, (int)lda, A21, (int)lda));
        for (int c = 0; c < m; c += GD_POTRF_NB) {
            const int cb = std::min(GD_POTRF_NB, m - c);
            const int rows = m - c;
            GD_CUBLAS(xgemm_tf32(hb, CUBLAS_OP_N, CUBLAS_OP_T, rows, cb, kb, -1.f,
                                 A21 + c, (int)lda, A21 + c, (int)lda, 1.f,
                                 A + (k + kb + c) * lda + (k + kb + c), (int)lda));
        }
    }
}

// ---------------------------------------------------------------------------
// Merge tree shared by trtri and lauum
//
// Both algorithms reduce to the same binary split of [0, np) at multiples of
// GD_BOT. A node can run once both of its children have, so grouping the nodes
// by height above the leaves gives levels whose members are mutually
// independent -- and, because a node's height is strictly greater than its
// descendants', members of one level never touch overlapping memory. The lower
// levels hold dozens of small nodes that individually leave the device idle, so
// each level is dealt round-robin over a few streams and joined at the end.
// ---------------------------------------------------------------------------

struct MergeNode { long p, n1, n2; int level; };

// The TF32 merge needs an n2 x n1 scratch per node, and a level's nodes are
// dealt round-robin over the fan, so any node can land on any slot. That forces
// uniform slices, hence (widest node) x (streams) rather than the smaller peak
// aggregate demand -- sizing by the latter underflows the slice the single
// top-level node needs.
inline size_t merge_scratch_elems(const std::vector<MergeNode>& nodes,
                                  int nstreams) {
    size_t widest = 0;
    for (const MergeNode& nd : nodes)
        widest = std::max(widest, (size_t)nd.n1 * nd.n2);
    return widest * (size_t)nstreams;
}

inline int build_merge_tree(std::vector<MergeNode>& out, long p, long n) {
    if (n <= GD_BOT) return 0;
    const long n1 = ((n / 2 + GD_BOT - 1) / GD_BOT) * GD_BOT;
    const long n2 = n - n1;
    const int l1 = build_merge_tree(out, p, n1);
    const int l2 = build_merge_tree(out, p + n1, n2);
    const int lv = (l1 > l2 ? l1 : l2) + 1;
    out.push_back(MergeNode{p, n1, n2, lv});
    return lv;
}

// A small fan-out of streams plus the events needed to join them.
class StreamFan {
public:
    explicit StreamFan(int n = 4) : s_(n), e_(n) {
        for (int i = 0; i < n; ++i) {
            GD_CUDA(cudaStreamCreate(&s_[i]));
            GD_CUDA(cudaEventCreateWithFlags(&e_[i], cudaEventDisableTiming));
        }
        GD_CUDA(cudaEventCreateWithFlags(&join_, cudaEventDisableTiming));
    }
    ~StreamFan() {
        for (auto& x : s_) cudaStreamDestroy(x);
        for (auto& x : e_) cudaEventDestroy(x);
        cudaEventDestroy(join_);
    }
    StreamFan(const StreamFan&) = delete;
    StreamFan& operator=(const StreamFan&) = delete;

    int size() const { return (int)s_.size(); }
    cudaStream_t at(int i) const { return s_[i]; }

    // Every fan stream waits for whatever `main` has queued so far.
    void fork(cudaStream_t main) {
        GD_CUDA(cudaEventRecord(join_, main));
        for (auto& x : s_) GD_CUDA(cudaStreamWaitEvent(x, join_, 0));
    }
    // `main` waits for the first `used` fan streams.
    void join(cudaStream_t main, int used) {
        for (int i = 0; i < used && i < (int)s_.size(); ++i) {
            GD_CUDA(cudaEventRecord(e_[i], s_[i]));
            GD_CUDA(cudaStreamWaitEvent(main, e_[i], 0));
        }
    }
private:
    std::vector<cudaStream_t> s_;
    std::vector<cudaEvent_t> e_;
    cudaEvent_t join_ = nullptr;
};

// Run `emit(node, stream)` over the tree, one level at a time, spreading each
// level over the fan. `independent` skips the per-level join for passes whose
// nodes never overlap at all.
template <class F>
inline void run_levels(cublasHandle_t hb, cudaStream_t main, StreamFan& fan,
                       const std::vector<MergeNode>& nodes, int maxlevel,
                       bool independent, F&& emit) {
    if (nodes.empty()) return;
    const int K = fan.size();
    fan.fork(main);
    int slot = 0;
    for (int lv = 1; lv <= maxlevel; ++lv) {
        int here = 0;
        for (const MergeNode& nd : nodes) {
            if (nd.level != lv) continue;
            GD_CUBLAS(cublasSetStream(hb, fan.at(slot)));
            emit(nd);
            slot = (slot + 1) % K;
            ++here;
        }
        if (!independent && here) {
            // Serialise level lv+1 behind lv: join into main, then fork again.
            fan.join(main, K);
            fan.fork(main);
        }
    }
    fan.join(main, K);
    GD_CUBLAS(cublasSetStream(hb, main));
}

// ---------------------------------------------------------------------------
// Triangular inverse: R = L^-1, out of place
//
// Bottom-up over the block size. The aligned GD_BASE diagonal blocks are
// inverted by one warp each; every level after that merges pairs of already
// inverted blocks with
//
//     R21 = -R22 * L21 * R11
//
// which reads L (untouched) and the previous level's R, so all the merges of a
// level are independent. Below GD_BOT they are uniform and go through two
// strided-batched GEMMs; above it the merge tree issues one TRMM pair per node,
// which is where ~75% of the flops land.
// ---------------------------------------------------------------------------

template <class S>
inline void trtri(cublasHandle_t hb, cudaStream_t main, StreamFan& fan,
                  const S* L, S* R, long np, S* tmp,
                  const std::vector<MergeNode>& nodes, int maxlevel,
                  bool use_tf32 = false, float* scratch = nullptr,
                  size_t scratch_elems = 0) {
    const long nblk = np / GD_BASE;
    const int blocks = (int)((nblk + GD_DIAG_WARPS - 1) / GD_DIAG_WARPS);
    k_trtri_diag<S><<<blocks, 32 * GD_DIAG_WARPS, 0, main>>>(L, np, nblk, R);
    GD_CUDA(cudaGetLastError());

    for (long s = GD_BASE; s < GD_BOT; s *= 2) {
        const int batch = (int)(np / (2 * s));
        const long long stride = 2LL * s * (np + 1);
        // tmp = -R22 * L21
        GD_CUBLAS(xgemm_batched(hb, CUBLAS_OP_N, CUBLAS_OP_N, (int)s, (int)s, (int)s,
                                (S)-1, R + s * np + s, (int)np, stride,
                                L + s, (int)np, stride, (S)0,
                                tmp, (int)s, (long long)s * s, batch));
        // R21 = tmp * R11
        GD_CUBLAS(xgemm_batched(hb, CUBLAS_OP_N, CUBLAS_OP_N, (int)s, (int)s, (int)s,
                                (S)1, tmp, (int)s, (long long)s * s,
                                R, (int)np, stride, (S)0,
                                R + s, (int)np, stride, batch));
    }

    const bool tf32 = use_tf32 && std::is_same<S, float>::value && scratch != nullptr;
    // Hand each node in a level its own slice of the scratch pool; run_levels
    // deals them round-robin over the same number of streams, so slice i is
    // only ever in use by one node at a time.
    size_t slice = 0;
    int slot = 0;
    const int K = fan.size();
    if (tf32) {
        slice = scratch_elems / (size_t)K;
    }
    run_levels(hb, main, fan, nodes, maxlevel, /*independent=*/false,
               [&](const MergeNode& nd) {
        S* R11 = R + nd.p * np + nd.p;
        S* R21 = R + nd.p * np + nd.p + nd.n1;
        S* R22 = R + (nd.p + nd.n1) * np + nd.p + nd.n1;
        const S* L21 = L + nd.p * np + nd.p + nd.n1;
        if (tf32) {
            // R21 = -R22 * L21 * R11, as two GEMMs through a scratch tile.
            // R11/R22 are read as full blocks, which is correct because the
            // strictly upper triangle of R is zero by construction.
            float* T = scratch + (size_t)slot * slice;
            slot = (slot + 1) % K;
            GD_CUBLAS(xgemm_tf32(hb, CUBLAS_OP_N, CUBLAS_OP_N, (int)nd.n2, (int)nd.n1,
                                 (int)nd.n2, -1.f, (const float*)R22, (int)np,
                                 (const float*)L21, (int)np, 0.f, T, (int)nd.n2));
            GD_CUBLAS(xgemm_tf32(hb, CUBLAS_OP_N, CUBLAS_OP_N, (int)nd.n2, (int)nd.n1,
                                 (int)nd.n1, 1.f, T, (int)nd.n2,
                                 (const float*)R11, (int)np, 0.f,
                                 (float*)R21, (int)np));
        } else {
            GD_CUBLAS(xtrmm(hb, CUBLAS_SIDE_LEFT, CUBLAS_OP_N, (int)nd.n2, (int)nd.n1,
                            (S)-1, R22, (int)np, L21, (int)np, R21, (int)np));
            GD_CUBLAS(xtrmm(hb, CUBLAS_SIDE_RIGHT, CUBLAS_OP_N, (int)nd.n2, (int)nd.n1,
                            (S)1, R11, (int)np, R21, (int)np, R21, (int)np));
        }
    });
}

// ---------------------------------------------------------------------------
// Triangular square: A = R^T R (lower), out of place
//
// A(I, J) = sum_{K >= max(I, J)} R(K, I)^T R(K, J). Splitting a block [p, p+n)
// into halves H1 and H2 splits that sum into four disjoint parts:
//
//   I, J, K in H1                    -> recurse on H1
//   I, J in H1, K in H2              -> A11 += R21^T R21      (accumulating)
//   I in H2, J in H1, K in H2        -> A21  = R22^T R21      (overwriting)
//   I, J, K in H2                    -> recurse on H2
//
// The A21 region of a node lies *inside* the A11 region of every ancestor, so
// the two kinds cannot be interleaved: every overwriting product has to be
// issued before any accumulating one. Hence the two passes below. Pass 1 is
// wholly independent (disjoint targets); pass 2 only has to respect the level
// order, because overlapping A11 regions always sit at different heights.
// A must be zeroed on entry.
// ---------------------------------------------------------------------------

template <class S>
inline void lauum(cublasHandle_t hb, cudaStream_t main, StreamFan& fan,
                  const S* R, S* A, long np,
                  const std::vector<MergeNode>& nodes, int maxlevel) {
    const long nblk = np / GD_BASE;
    const int blocks = (int)((nblk + GD_DIAG_WARPS - 1) / GD_DIAG_WARPS);

    // ---- pass 1: A21 = R22^T R21 (overwriting) ----
    k_lauum_diag<S><<<blocks, 32 * GD_DIAG_WARPS, 0, main>>>(R, np, nblk, A);
    GD_CUDA(cudaGetLastError());
    for (long s = GD_BASE; s < GD_BOT; s *= 2) {
        const int batch = (int)(np / (2 * s));
        const long long stride = 2LL * s * (np + 1);
        GD_CUBLAS(xgemm_batched(hb, CUBLAS_OP_T, CUBLAS_OP_N, (int)s, (int)s, (int)s,
                                (S)1, R + s * np + s, (int)np, stride,
                                R + s, (int)np, stride, (S)0,
                                A + s, (int)np, stride, batch));
    }
    run_levels(hb, main, fan, nodes, maxlevel, /*independent=*/true,
               [&](const MergeNode& nd) {
        const S* R21 = R + nd.p * np + nd.p + nd.n1;
        const S* R22 = R + (nd.p + nd.n1) * np + nd.p + nd.n1;
        S* A21 = A + nd.p * np + nd.p + nd.n1;
        GD_CUBLAS(xtrmm(hb, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, (int)nd.n2, (int)nd.n1,
                        (S)1, R22, (int)np, R21, (int)np, A21, (int)np));
    });

    // ---- pass 2: A11 += R21^T R21 (accumulating) ----
    for (long s = GD_BASE; s < GD_BOT; s *= 2) {
        const int batch = (int)(np / (2 * s));
        const long long stride = 2LL * s * (np + 1);
        GD_CUBLAS(xgemm_batched(hb, CUBLAS_OP_T, CUBLAS_OP_N, (int)s, (int)s, (int)s,
                                (S)1, R + s, (int)np, stride,
                                R + s, (int)np, stride, (S)1,
                                A, (int)np, stride, batch));
    }
    run_levels(hb, main, fan, nodes, maxlevel, /*independent=*/false,
               [&](const MergeNode& nd) {
        const S* R21 = R + nd.p * np + nd.p + nd.n1;
        S* A11 = A + nd.p * np + nd.p;
        GD_CUBLAS(xsyrk(hb, (int)nd.n1, (int)nd.n2, (S)1, R21, (int)np, (S)1,
                        A11, (int)np));
    });
}

// ---------------------------------------------------------------------------
// Blocked lauum: A = R^T R (lower) as one GEMM per block row
//
//     A(I, 0:I+1) = R(I:m, I)^T * R(I:m, 0:I+1)
//
// Every (I, J, K) with K >= I >= J is covered exactly once, so this is the same
// n^3/6 as the recursive form above -- the only waste is the strictly upper
// half of each nb x nb diagonal tile, O(n*nb^2). It needs no base kernel, no
// merge tree and no streams. A single wide GEMM beats a tree of TRMM/SYRK calls 
// that get small near the leaves. It relies on the strictly upper triangle of R 
// being zero, which the pipeline guarantees (the buffer is memset and only the 
// lower half written).
//
// It is also the only stage that can use fp16: cuBLAS has no half TRMM or SYRK,
// so any half-precision version of the recursive form would have to treat the
// triangles as full and do three times the work, which on sm_60 exactly cancels
// the 2.1x that fp16 arithmetic buys.
// ---------------------------------------------------------------------------

// Block row height. 512 measured best from np = 5120 to 11264; wider tiles lose
// more to the diagonal waste, narrower ones make the GEMM too thin.
static constexpr long GD_LAUUM_NB = 512;

template <class S>
inline void lauum_blocked(cublasHandle_t hb, const S* R, S* A, long np,
                          bool tf32 = false) {
    for (long p = 0; p < np; p += GD_LAUUM_NB) {
        const int rows = (int)std::min(GD_LAUUM_NB, np - p);
        const int cols = (int)(p + rows);
        const int k = (int)(np - p);
        if (tf32 && std::is_same<S, float>::value) {
            GD_CUBLAS(xgemm_tf32(hb, CUBLAS_OP_T, CUBLAS_OP_N, rows, cols, k, 1.f,
                                 (const float*)(R + p * np + p), (int)np,
                                 (const float*)(R + p), (int)np, 0.f,
                                 (float*)(A + p), (int)np));
        } else {
            GD_CUBLAS(xgemm(hb, CUBLAS_OP_T, CUBLAS_OP_N, rows, cols, k, (S)1,
                            R + p * np + p, (int)np,   // R(p:np, p:p+rows)
                            R + p, (int)np,            // R(p:np, 0:p+rows)
                            (S)0, A + p, (int)np));    // A(p:p+rows, 0:p+rows)
        }
    }
}

// fp16 variant of the same loop. `hR` holds R in half (np*np); `hA` is only one
// block row wide (GD_LAUUM_NB * np), because each tile is converted back to S
// right after its GEMM -- that also keeps the untouched upper triangle of A out
// of the conversion.
//
// R is rescaled so the largest diagonal entry of A lands near 2^10. Half tops
// out at 65504 and the diagonal accumulator is monotone (A(i,i) is a sum of
// squares), so without this a pair with a small enough eigenvalue would
// saturate silently rather than fail.
inline void lauum_blocked_half(cublasHandle_t hb, cudaStream_t stream, const float* R,
                               float* A, long np, __half* hR, float* scratch) {
    const float zero = 0.f;
    GD_CUDA(cudaMemcpyAsync(scratch, &zero, sizeof(float), cudaMemcpyHostToDevice,
                            stream));
    k_max_colnorm2<float><<<256, 256, 0, stream>>>(R, np, np, scratch);
    float maxdiag = 0.f;
    GD_CUDA(cudaMemcpyAsync(&maxdiag, scratch, sizeof(float), cudaMemcpyDeviceToHost,
                            stream));
    GD_CUDA(cudaStreamSynchronize(stream));
    const float alpha = (maxdiag > 0.f) ? std::sqrt(1024.f / maxdiag) : 1.f;
    const float inv = 1.f / (alpha * alpha);

    k_to_half<float><<<1024, 256, 0, stream>>>(R, hR, np * np, alpha);
    GD_CUDA(cudaGetLastError());

    for (long p = 0; p < np; p += GD_LAUUM_NB) {
        const int rows = (int)std::min(GD_LAUUM_NB, np - p);
        const int cols = (int)(p + rows);
        const int k = (int)(np - p);
        // The fp32 accumulator takes a float alpha, so the rescale is free.
        const float one = inv, beta = 0.f;
        GD_CUBLAS(cublasGemmEx(hb, CUBLAS_OP_T, CUBLAS_OP_N, rows, cols, k, &one,
                               hR + p * np + p, CUDA_R_16F, (int)np,
                               hR + p, CUDA_R_16F, (int)np, &beta,
                               A + p, CUDA_R_32F, (int)np,
                               CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    GD_CUDA(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Device buffer that grows on demand, so a batch of pairs reuses one allocation
// ---------------------------------------------------------------------------

class DevBuf {
public:
    DevBuf() = default;
    ~DevBuf() { if (p_) cudaFree(p_); }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    void* reserve(size_t bytes) {
        if (bytes > cap_) {
            if (p_) GD_CUDA(cudaFree(p_));
            p_ = nullptr;
            GD_CUDA(cudaMalloc(&p_, bytes));
            cap_ = bytes;
        }
        return p_;
    }
    template <class T> T* as() { return static_cast<T*>(p_); }
    size_t capacity() const { return cap_; }
private:
    void* p_ = nullptr;
    size_t cap_ = 0;
};

class HostBuf {
public:
    HostBuf() = default;
    ~HostBuf() { if (p_) cudaFreeHost(p_); }
    HostBuf(const HostBuf&) = delete;
    HostBuf& operator=(const HostBuf&) = delete;
    void* reserve(size_t bytes) {
        if (bytes > cap_) {
            if (p_) GD_CUDA(cudaFreeHost(p_));
            p_ = nullptr;
            GD_CUDA(cudaHostAlloc(&p_, bytes, cudaHostAllocDefault));
            cap_ = bytes;
        }
        return p_;
    }
    template <class T> T* as() { return static_cast<T*>(p_); }
private:
    void* p_ = nullptr;
    size_t cap_ = 0;
};

// Wall-clock of each stage, filled in when `profile` is set.
struct Timings {
    double upload = 0, covariance = 0, potrf = 0, trtri = 0, lauum = 0,
           score = 0, download = 0, total = 0;
};

// ---------------------------------------------------------------------------
// Session: holds the CUDA context, the handles and the scratch
// ---------------------------------------------------------------------------

class Session {
public:
    explicit Session(int device = 0) : device_(device) {
        GD_CUDA(cudaSetDevice(device));
        GD_CUDA(cudaFree(nullptr));                     // force context creation
        GD_CUBLAS(cublasCreate(&hb_));
        GD_CUSOLVER(cusolverDnCreate(&hs_));
        GD_CUDA(cudaStreamCreate(&stream_));
        GD_CUBLAS(cublasSetStream(hb_, stream_));
        GD_CUSOLVER(cusolverDnSetStream(hs_, stream_));
        fan_.reset(new StreamFan(4));
        cudaDeviceProp prop{};
        GD_CUDA(cudaGetDeviceProperties(&prop, device));
        name_ = prop.name;
        sm_major_ = prop.major;
        sm_minor_ = prop.minor;
    }
    ~Session() {
        if (hb_) cublasDestroy(hb_);
        if (hs_) cusolverDnDestroy(hs_);
        if (stream_) cudaStreamDestroy(stream_);
    }
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;

    const std::string& device_name() const { return name_; }
    const Timings& timings() const { return t_; }
    void set_profile(bool on) { profile_ = on; }

    bool has_tensor_cores() const { return sm_major_ >= 7; }
    // TF32 and the fp16/fp32-accumulate GEMM only pay from Ampere on. Below
    // that they are still correct -- cuBLAS quietly falls back to fp32 -- but
    // the extra flops the GEMM re-expressions cost are no longer bought back,
    // and the hand-written Cholesky loses to cuSOLVER's. So the tensor-core
    // path is taken exactly when the hardware can honour it.
    bool tensor_path() const { return sm_major_ >= 8; }
    const char* arithmetic() const {
        return tensor_path() ? "tf32 potrf/trtri + fp16 lauum" : "fp32";
    }

    // Free memory needed for one (N, B) pair, in bytes. Used for admission
    // control before the pair is dispatched.
    template <class S>
    static size_t device_bytes(int N, int B, int q) {
        const long NQ = (long)N * (q - 1);
        const long np = pad(NQ);
        const long Bp = padB(B);
        return 2 * (size_t)np * np * sizeof(S)
             + (size_t)np * np * sizeof(__half)         // fp16 lauum operand
             + (size_t)np * np * sizeof(S)              // TF32 merge scratch
             + (size_t)N * B + (size_t)N * Bp
             + (size_t)Bp * (sizeof(double) + sizeof(float))
             + (size_t)N * N * (sizeof(double) + sizeof(float)) + (32u << 20);
    }

    // Full pipeline: covariance -> inverse -> Frobenius norm -> APC.
    void score_into(const int8_t* MSA, int N, int B, const double* W, double Beff,
                    int q, double pseudocount, float* score_out) {
        run<CovScalar>(MSA, N, B, W, Beff, q, pseudocount, score_out, nullptr);
    }

    // Full symmetric (NQ x NQ) inverse in float64, column-major (== row-major,
    // the matrix is symmetric). Matches gaussdca::compute_couplings_into.
    void couplings_into(const int8_t* MSA, int N, int B, const double* W, double Beff,
                        int q, double pseudocount, double* out) {
        run<double>(MSA, N, B, W, Beff, q, pseudocount, nullptr, out);
    }

    static long pad(long n) { return ((n + GD_BOT - 1) / GD_BOT) * GD_BOT; }
    static long padB(int B) { return ((long)(B + 15) / 16) * 16; }

private:
    template <class S>
    void run(const int8_t* MSA, int N, int B, const double* W, double Beff, int q,
             double pseudocount, float* score_out, double* cpl_out);

    cublasHandle_t hb_ = nullptr;
    cusolverDnHandle_t hs_ = nullptr;
    cudaStream_t stream_ = nullptr;
    std::unique_ptr<StreamFan> fan_;
    std::vector<MergeNode> merge_;
    int merge_levels_ = 0;
    size_t merge_scratch_ = 0;
    long cached_np_ = -1;
    std::string name_;
    int device_ = 0;
    int sm_major_ = 0, sm_minor_ = 0;
    bool profile_ = false;
    Timings t_;

    DevBuf dMSA_, dMSAc_, dW_, dWd_, dWf_, dFi_, dGroups_, dA_, dR_, dTmp_,
           dFN_, dSi_, dScore_, dWork_, dInfo_, dHalf_, dNorm_, dMerge_;
    HostBuf hMSA_, hScore_;
    std::vector<int2> groups_;
    int cached_groups_N_ = -1;
};

}  // namespace cuda
}  // namespace gaussdca

namespace gaussdca {
namespace cuda {

// Stage timer that costs nothing when profiling is off.
class StageTimer {
public:
    StageTimer(bool on, cudaStream_t s) : on_(on), s_(s) {
        if (!on_) return;
        cudaEventCreate(&a_);
        cudaEventCreate(&b_);
    }
    ~StageTimer() { if (on_) { cudaEventDestroy(a_); cudaEventDestroy(b_); } }
    void start() { if (on_) cudaEventRecord(a_, s_); }
    void stop(double* out) {
        if (!on_) return;
        cudaEventRecord(b_, s_);
        cudaEventSynchronize(b_);
        float ms = 0;
        cudaEventElapsedTime(&ms, a_, b_);
        *out += ms * 1e-3;
    }
private:
    bool on_;
    cudaStream_t s_;
    cudaEvent_t a_ = nullptr, b_ = nullptr;
};

template <class S>
void Session::run(const int8_t* MSA, int N, int B, const double* W, double Beff,
                  int q, double pseudocount, float* score_out, double* cpl_out) {
    const int Q = q - 1;
    if (Q < 1 || Q > 32)
        throw std::runtime_error("gauss_dca_cuda: alphabet size q must satisfy 2 <= q <= 33");
    if (N < 2 || B < 1)
        throw std::runtime_error("gauss_dca_cuda: need N >= 2 and B >= 1");

    const long NQ = (long)N * Q;
    const long np = pad(NQ);
    const long Bp = padB(B);
    if (np > (long)INT32_MAX)
        throw std::runtime_error("gauss_dca_cuda: NQ exceeds the cuBLAS 32-bit index range");

    // CUDA's current device is per *thread*, so a Session handed between
    // worker threads (a pool hands them out per pair) has to re-select its own
    // device or the launches land wherever that thread was last pointed.
    GD_CUDA(cudaSetDevice(device_));

    if (cached_np_ != np) {
        merge_.clear();
        merge_levels_ = build_merge_tree(merge_, 0, np);
        merge_scratch_ = merge_scratch_elems(merge_, fan_->size());
        cached_np_ = np;
    }

    t_ = Timings();
    StageTimer tm(profile_, stream_);

    // ---- buffers -----------------------------------------------------------
    int8_t*  dMSA    = (int8_t*) dMSA_.reserve((size_t)N * B);
    uint8_t* dMSAc   = (uint8_t*)dMSAc_.reserve((size_t)N * Bp);
    double*  dW    = (double*) dW_.reserve((size_t)B * sizeof(double));
    double*  dWd   = (double*) dWd_.reserve((size_t)Bp * sizeof(double));
    float*   dWf   = (float*)  dWf_.reserve((size_t)Bp * sizeof(float));
    double*  dFi   = (double*) dFi_.reserve((size_t)NQ * sizeof(double));
    S*       dA    = (S*)      dA_.reserve((size_t)np * np * sizeof(S));
    S*       dR    = (S*)      dR_.reserve((size_t)np * np * sizeof(S));
    S*       dTmp  = (S*)      dTmp_.reserve((size_t)32 * np * sizeof(S));
    double*  dFN   = (double*) dFN_.reserve((size_t)N * N * sizeof(double));
    double*  dSi   = (double*) dSi_.reserve((size_t)(N + 1) * sizeof(double));
    float*   dSc   = (float*)  dScore_.reserve((size_t)N * N * sizeof(float));
    int*     dInfo = (int*)    dInfo_.reserve(sizeof(int));

    // ---- upload ------------------------------------------------------------
    tm.start();
    int8_t* hMSA = (int8_t*)hMSA_.reserve((size_t)N * B);
    std::memcpy(hMSA, MSA, (size_t)N * B);
    GD_CUDA(cudaMemcpyAsync(dMSA, hMSA, (size_t)N * B, cudaMemcpyHostToDevice, stream_));
    GD_CUDA(cudaMemcpyAsync(dW, W, (size_t)B * sizeof(double), cudaMemcpyHostToDevice,
                            stream_));

    if ((int)groups_.size() == 0 || cached_groups_N_ != N) {
        groups_.clear();
        for (int i = 0; i < N; ++i)
            for (int j0 = i + 1; j0 < N; j0 += GD_PAIR_WARPS)
                groups_.push_back(make_int2(i, j0));
        cached_groups_N_ = N;
        int2* dG = (int2*)dGroups_.reserve(groups_.size() * sizeof(int2));
        GD_CUDA(cudaMemcpyAsync(dG, groups_.data(), groups_.size() * sizeof(int2),
                                cudaMemcpyHostToDevice, stream_));
    }
    const int2* dGroups = (const int2*)dGroups_.as<int2>();
    tm.stop(&t_.upload);

    // ---- covariance --------------------------------------------------------
    tm.start();
    const double lambda = pseudocount;
    const double one_minus_lambda = 1.0 - lambda;
    const double unif_i = lambda / q;
    const double unif_ij = unif_i / q;
    const double invBeff = 1.0 / Beff;

    k_recode<<<1024, 256, 0, stream_>>>(dMSA, N, B, (int)Bp, Q, dMSAc);
    k_weights<<<256, 256, 0, stream_>>>(dW, B, (int)Bp, dWd, dWf);
    k_fi<<<N, 32 * GD_RED_WARPS, GD_RED_WARPS * (Q + 1) * sizeof(double), stream_>>>(
        dMSAc, (int)Bp, Q, dWd, invBeff, one_minus_lambda, unif_i, dFi);

    using Hist = typename HistFor<S>::type;
    const size_t hist_bytes =
        (size_t)GD_PAIR_WARPS * GD_HIST_COPIES * Q * Q * sizeof(Hist);
    const Hist* Wh = std::is_same<Hist, double>::value ? (const Hist*)dWd
                                                       : (const Hist*)dWf;
    k_cov_pairs<Hist, S><<<(int)groups_.size(), 32 * GD_PAIR_WARPS, hist_bytes,
                           stream_>>>(
        dMSAc, N, (int)Bp, Q, Wh, dGroups, dFi, np, one_minus_lambda, invBeff,
        unif_ij, dA);
    k_cov_diag<S><<<N, 256, 0, stream_>>>(N, Q, dFi, np, dA);
    if (np != NQ) k_pad_identity<S><<<256, 256, 0, stream_>>>(NQ, np, dA);
    GD_CUDA(cudaGetLastError());
    tm.stop(&t_.covariance);

    // ---- Cholesky ----------------------------------------------------------
    tm.start();
    int lwork = 0;
    GD_CUSOLVER(xpotrf_buffer(hs_, (int)np, dA, (int)np, &lwork));
    S* dWork = (S*)dWork_.reserve((size_t)lwork * sizeof(S));
    if (tensor_path() && std::is_same<S, float>::value) {
        potrf_blocked_tf32(hb_, hs_, (float*)dA, np, np, (float*)dWork, lwork, dInfo);
    } else {
        GD_CUSOLVER(xpotrf(hs_, (int)np, dA, (int)np, dWork, lwork, dInfo));
    }
    int info = 0;
    GD_CUDA(cudaMemcpyAsync(&info, dInfo, sizeof(int), cudaMemcpyDeviceToHost, stream_));
    GD_CUDA(cudaStreamSynchronize(stream_));
    if (info != 0)
        throw std::runtime_error("Cholesky factorization failed (C not SPD), potrf info=" +
                                 std::to_string(info));
    tm.stop(&t_.potrf);

    // ---- R = L^-1 ----------------------------------------------------------
    tm.start();
    GD_CUDA(cudaMemsetAsync(dR, 0, (size_t)np * np * sizeof(S), stream_));
    float* mscratch = nullptr;
    const bool tc = tensor_path() && std::is_same<S, float>::value;
    if (tc && merge_scratch_ > 0)
        mscratch = (float*)dMerge_.reserve(merge_scratch_ * sizeof(float));
    trtri<S>(hb_, stream_, *fan_, dA, dR, np, dTmp, merge_, merge_levels_,
             tc, mscratch, merge_scratch_);
    tm.stop(&t_.trtri);

    // ---- A = R^T R = C^-1 (reusing the now-dead factor buffer) -------------
    tm.start();
    if (tc) {
        // R in half plus the norm scratch; the result lands straight in A.
        __half* hbuf = (__half*)dHalf_.reserve((size_t)np * np * sizeof(__half));
        float* scratch = (float*)dNorm_.reserve(sizeof(float));
        lauum_blocked_half(hb_, stream_, (const float*)dR, (float*)dA, np, hbuf,
                           scratch);
    } else {
        lauum_blocked<S>(hb_, dR, dA, np, false);
    }
    tm.stop(&t_.lauum);

    // ---- reduction ---------------------------------------------------------
    if (score_out) {
        tm.start();
        // k_score fills the strictly lower and upper blocks and leaves the
        // diagonal alone, exactly as the CPU core does -- but there the vector
        // is zero-initialised per call, while dFN is reused across pairs in a
        // --serve run. Without this the APC row sums pick up the previous
        // pair's values (and, when N changes, read them at shifted offsets).
        GD_CUDA(cudaMemsetAsync(dFN, 0, (size_t)N * N * sizeof(double), stream_));
        k_score<S><<<N, 32 * GD_RED_WARPS, 0, stream_>>>(dA, N, Q, np, dFN);
        k_rowsums<<<N, 256, 0, stream_>>>(dFN, N, dSi);
        k_total<<<1, 256, 0, stream_>>>(dSi, N, dSi + N);
        k_apc<<<256, 256, 0, stream_>>>(dFN, dSi, dSi + N, N, dSc);
        GD_CUDA(cudaGetLastError());
        tm.stop(&t_.score);

        tm.start();
        float* hS = (float*)hScore_.reserve((size_t)N * N * sizeof(float));
        GD_CUDA(cudaMemcpyAsync(hS, dSc, (size_t)N * N * sizeof(float),
                                cudaMemcpyDeviceToHost, stream_));
        GD_CUDA(cudaStreamSynchronize(stream_));
        std::memcpy(score_out, hS, (size_t)N * N * sizeof(float));
        tm.stop(&t_.download);
    } else {
        tm.start();
        k_mirror<S><<<256, 256, 0, stream_>>>(dA, NQ, np);
        GD_CUDA(cudaGetLastError());
        GD_CUDA(cudaMemcpy2DAsync(cpl_out, (size_t)NQ * sizeof(S), dA,
                                  (size_t)np * sizeof(S), (size_t)NQ * sizeof(S),
                                  (size_t)NQ, cudaMemcpyDeviceToHost, stream_));
        GD_CUDA(cudaStreamSynchronize(stream_));
        tm.stop(&t_.download);
    }
    GD_CUDA(cudaStreamSynchronize(stream_));
    t_.total = t_.upload + t_.covariance + t_.potrf + t_.trtri + t_.lauum +
               t_.score + t_.download;
}

}  // namespace cuda
}  // namespace gaussdca
