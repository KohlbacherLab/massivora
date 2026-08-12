// gauss_dca.hpp
//
// GaussDCA covariance-inverse core (Baldassi et al. 2014, "Fast and Accurate
// Multivariate Gaussian Modeling of Protein Families", PLOS ONE 9(3):e92721;
//
// Namings:
//   N      number of residue positions (paper: L)
//   M      number of sequences
//   q      alphabet size, gap included (21)
//   Q      q - 1: number of amino-acid states, gap excluded (paper: Q = 20)
//   Z      integer-encoded alignment, residue-major (Z[i*M + k], values 1..q)
//   W      reweighting vector,  Meff = sum(W)
//   fi     reweighted single-site frequencies      (paper: x-bar, Eq. 1)
//   fij    reweighted pair frequencies             (paper: Eq. 2)
//   C      covariance = fij - fi*fi^T after pseudocount (paper: Sigma, Eq. 14)
//   J      couplings = C^-1                         (paper: e_kl = -J blocks, Eq. 5)
//   FN     gap-excluded Frobenius norm of each coupling block, after the
//          Ising ("zero-sum") gauge
//   S      FN after the Average Product Correction -- the pipeline's actual output
//
// ---------------------------------------------------------------------------
// Two entry points
//
//   compute_score_into()      C -> C^-1 -> per-block Frobenius norm -> APC.
//                             Returns the (N x N) score the pipeline actually
//                             wants and never materialises J for the caller.
//                             This is what gauss_infer uses.
//
//   compute_couplings_into()  Returns the full symmetric (NQ x NQ) inverse.
//                             Kept for the pybind entry point and the tests.

#pragma once

#include <Eigen/Dense>
#include <Eigen/Cholesky>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace gaussdca {

// Precision the covariance is stored and factorised in. Single precision halves
// the workspace -- which is now the whole memory footprint -- and roughly
// doubles the factorisation rate. The published score is float32 either way.
#ifdef GAUSSDCA_MIXED_PRECISION
using CovScalar = float;
#else
using CovScalar = double;
#endif

#ifdef GAUSSDCA_USE_LAPACK
extern "C" {
// LAPACK (column-major, LP64). `?potrf` factorises an SPD matrix in place;
// `?potri` overwrites that factor with the inverse of the original matrix.
void dpotrf_(const char* uplo, const int* n, double* a, const int* lda, int* info);
void dpotri_(const char* uplo, const int* n, double* a, const int* lda, int* info);
void spotrf_(const char* uplo, const int* n, float* a, const int* lda, int* info);
void spotri_(const char* uplo, const int* n, float* a, const int* lda, int* info);
}
inline void lapack_potrf(const char* u, const int* n, double* a, const int* l, int* i) {
    dpotrf_(u, n, a, l, i);
}
inline void lapack_potri(const char* u, const int* n, double* a, const int* l, int* i) {
    dpotri_(u, n, a, l, i);
}
inline void lapack_potrf(const char* u, const int* n, float* a, const int* l, int* i) {
    spotrf_(u, n, a, l, i);
}
inline void lapack_potri(const char* u, const int* n, float* a, const int* l, int* i) {
    spotri_(u, n, a, l, i);
}
// Thread-count hooks, weak so the binary still links against a BLAS that lacks
// them. Without these a threaded BLAS would ignore n_threads and grab every
// core, which breaks the executor's resource accounting.
extern "C" void openblas_set_num_threads(int) __attribute__((weak));
extern "C" void mkl_set_num_threads(int*) __attribute__((weak));
#endif

// Thread count used for the pair-frequency accumulation and the reductions.
inline int hardware_threads() {
    unsigned n = std::thread::hardware_concurrency();
    return n == 0 ? 1 : static_cast<int>(n);
}

inline int resolve_threads(int n_threads) {
    int n = (n_threads > 0) ? n_threads : hardware_threads();
    return n < 1 ? 1 : n;
}

// Split [0, total) into `nthreads` contiguous chunks and run f(lo, hi) on each.
template <class F>
inline void parallel_for(long total, int nthreads, F&& f) {
    if (total <= 0) return;
    if (nthreads <= 1) { f(0L, total); return; }
    const long chunk = (total + nthreads - 1) / nthreads;
    std::vector<std::thread> pool;
    pool.reserve(static_cast<size_t>(nthreads));
    for (int p = 0; p < nthreads; ++p) {
        const long lo = static_cast<long>(p) * chunk;
        const long hi = std::min(total, lo + chunk);
        if (lo >= hi) break;
        pool.emplace_back([&f, lo, hi] { f(lo, hi); });
    }
    for (auto& th : pool) th.join();
}

// Tell a threaded BLAS how many cores it may use.
inline void set_blas_threads(int n) {
#ifdef GAUSSDCA_USE_LAPACK
    if (openblas_set_num_threads) openblas_set_num_threads(n);
    if (mkl_set_num_threads) { int nn = n; mkl_set_num_threads(&nn); }
#else
    (void)n;
#endif
}

// 64-byte aligned, *uninitialised* scratch. The covariance is written before it
// is read and the upper triangle is never referenced, so zeroing it would be a
// wasted pass over the largest allocation in the program.
template <class T>
class AlignedBuffer {
public:
    explicit AlignedBuffer(size_t n) {
        const size_t bytes = ((n * sizeof(T)) + 63) & ~size_t(63);
        p_ = static_cast<T*>(std::aligned_alloc(64, bytes));
        if (!p_) throw std::bad_alloc();
    }
    ~AlignedBuffer() { std::free(p_); }
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;
    T* get() const { return p_; }
private:
    T* p_ = nullptr;
};

// Largest n an LP64 LAPACK can be trusted with: n*n must stay inside a signed
// 32-bit index. n = 46340 corresponds to ~2317 alignment columns.
inline constexpr long LAPACK_LP64_MAX_N = 46340L;

// Invert the SPD matrix held in the lower triangle of `A` (n x n, column-major,
// leading dimension n), in place. Only the lower triangle is read; on return
// only the lower triangle is valid.
inline void invert_spd_lower_inplace(CovScalar* A, long n, int nthreads) {
#ifdef GAUSSDCA_USE_LAPACK
    if (n <= LAPACK_LP64_MAX_N) {
        set_blas_threads(nthreads);
        const char uplo = 'L';
        int ni = static_cast<int>(n), info = 0;
        lapack_potrf(&uplo, &ni, A, &ni, &info);
        if (info != 0)
            throw std::runtime_error("Cholesky factorization failed (C not SPD), potrf info=" +
                                     std::to_string(info));
        lapack_potri(&uplo, &ni, A, &ni, &info);
        if (info != 0)
            throw std::runtime_error("Inversion failed, potri info=" + std::to_string(info));
        return;
    }
    // Past the LP64 limit an ILP64 build would be needed, so rather than fail on
    // a very wide pair fall through to the Eigen path: slower and single
    // threaded, but it still produces the couplings.
    std::fprintf(stderr,
                 "gauss_infer: NQ = %ld exceeds the LP64 LAPACK limit (%ld); "
                 "using the single-threaded Eigen inverse. Link an ILP64 BLAS "
                 "to keep the threaded path for alignments this wide.\n",
                 n, LAPACK_LP64_MAX_N);
#endif
    // Eigen fallback: in-place Cholesky
    (void)nthreads;
    using CovMatrix = Eigen::Matrix<CovScalar, Eigen::Dynamic, Eigen::Dynamic>;
    Eigen::Map<CovMatrix> C(A, n, n);
    Eigen::LLT<Eigen::Ref<CovMatrix>> llt(C);
    if (llt.info() != Eigen::Success)
        throw std::runtime_error("Cholesky factorization failed (C not SPD)");
    CovMatrix Linv = CovMatrix::Identity(n, n);
    C.triangularView<Eigen::Lower>().solveInPlace(Linv);
    C.setZero();
    C.selfadjointView<Eigen::Lower>().rankUpdate(Linv.transpose());
}

// Assemble the lower triangle of C = fij - fi*fi^T (with pseudocount) into `C`
// (NQ x NQ, column-major). The pseudocount mix and the rank-1 term are fused
// into the pair-block store, so no separate pass over the matrix is needed.
inline void build_covariance_lower(const int8_t* Z, int N, int M, const double* W,
                                   double Meff, int q, double pseudocount,
                                   int nthreads, CovScalar* C) {
    const int Q = q - 1;
    const int QP = Q + 1;                         // + one sink bin for gaps
    const long NQ = static_cast<long>(N) * Q;

    // ---- recode: [0, Q-1] for real symbols, Q for anything gap-like ----
    // Doing this once turns the two range tests in the O(N^2 M) inner loop into
    // a plain indexed accumulate.
    std::vector<uint8_t> Zc(static_cast<size_t>(N) * M);
    parallel_for(N, nthreads, [&](long lo, long hi) {
        for (long i = lo; i < hi; ++i) {
            const int8_t* zi = Z + static_cast<size_t>(i) * M;
            uint8_t* zc = Zc.data() + static_cast<size_t>(i) * M;
            for (int k = 0; k < M; ++k) {
                const int a = zi[k];
                zc[k] = static_cast<uint8_t>((a >= 1 && a <= Q) ? a - 1 : Q);
            }
        }
    });

    // ---- fi: single-site frequencies (paper x-bar, Eq. 1) ----
    Eigen::VectorXd fi(NQ);
    parallel_for(N, nthreads, [&](long lo, long hi) {
        std::vector<double> acc(QP);
        for (long i = lo; i < hi; ++i) {
            std::fill(acc.begin(), acc.end(), 0.0);
            const uint8_t* zc = Zc.data() + static_cast<size_t>(i) * M;
            for (int k = 0; k < M; ++k) acc[zc[k]] += W[k];
            double* fi_i = fi.data() + i * Q;
            for (int a = 0; a < Q; ++a) fi_i[a] = acc[a] / Meff;
        }
    });

    // ---- pseudocount (paper Eq. 14, weight lambda) applied to fi up front ----
    const double lambda = pseudocount;
    const double unif_i = lambda / q;             // uniform mass per state
    const double unif_ij = unif_i / q;            // lambda / q^2, the fij offset
    for (long x = 0; x < NQ; ++x) fi(x) = (1.0 - lambda) * fi(x) + unif_i;
    const double* fip = fi.data();

    // ---- fij + pseudocount + covariance, fused, lower triangle only ----
    // For a pair (i < j) the lower-triangle home of its block is rows j*Q+b,
    // columns i*Q+a, which is contiguous in column-major order: one Q-long
    // store per a.
    std::vector<std::pair<int, int>> pairs;
    pairs.reserve(static_cast<size_t>(N) * (N - 1) / 2);
    for (int i = 0; i < N; ++i)
        for (int j = i + 1; j < N; ++j) pairs.emplace_back(i, j);

    const int npair_threads =
        std::min(nthreads, std::max<int>(1, static_cast<int>(pairs.size())));
    parallel_for(static_cast<long>(pairs.size()), npair_threads, [&](long lo, long hi) {
        std::vector<double> acc(static_cast<size_t>(QP) * QP);
        double* a_ = acc.data();
        for (long idx = lo; idx < hi; ++idx) {
            const int i = pairs[idx].first, j = pairs[idx].second;
            const uint8_t* zi = Zc.data() + static_cast<size_t>(i) * M;
            const uint8_t* zj = Zc.data() + static_cast<size_t>(j) * M;
            std::fill(acc.begin(), acc.end(), 0.0);
            for (int k = 0; k < M; ++k)
                a_[static_cast<size_t>(zi[k]) * QP + zj[k]] += W[k];

            const long r0 = static_cast<long>(j) * Q;   // row block  (lower)
            const long c0 = static_cast<long>(i) * Q;   // col block
            for (int a = 0; a < Q; ++a) {
                CovScalar* dst = C + (c0 + a) * NQ + r0;
                const double fa = fip[c0 + a];
                const double* arow = a_ + static_cast<size_t>(a) * QP;
                for (int b = 0; b < Q; ++b)
                    dst[b] = static_cast<CovScalar>(
                        (1.0 - lambda) * (arow[b] / Meff) + unif_ij
                        - fip[r0 + b] * fa);
            }
        }
    });

    // ---- diagonal blocks: zero off-diagonal, fi on the diagonal, minus fi*fi^T ----
    parallel_for(N, nthreads, [&](long lo, long hi) {
        for (long i = lo; i < hi; ++i) {
            const long x0 = i * Q;
            for (int a = 0; a < Q; ++a) {
                CovScalar* dst = C + (x0 + a) * NQ + x0;
                const double fa = fip[x0 + a];
                for (int b = a; b < Q; ++b)          // lower triangle of the block
                    dst[b] = static_cast<CovScalar>(
                        (a == b ? fip[x0 + a] : 0.0) - fip[x0 + b] * fa);
            }
        }
    });
}

// Mirror the lower triangle of an (n x n) column-major matrix into the upper.
inline void mirror_lower_to_upper(CovScalar* A, long n, int nthreads) {
    parallel_for(n, nthreads, [&](long lo, long hi) {
        for (long c = lo; c < hi; ++c) {
            const CovScalar* src = A + c * n;        // column c, rows c..n-1
            for (long r = c + 1; r < n; ++r) A[r * n + c] = src[r];
        }
    });
}

// Reduce the inverse in `Jlow` (lower triangle only, NQ x NQ column-major) to
// the (N x N) APC-corrected score, written row-major into `score_out`.
//
// The Ising gauge is applied per (i, j) block and the Frobenius norm is taken
// over that block alone, so nothing outside the block is needed. FN comes out
// symmetric and its diagonal is discarded, so only the strictly lower blocks
// are read -- every one of which lies wholly inside the stored triangle.
inline void score_from_inverse_lower(const CovScalar* Jlow, int N, int q,
                                     int nthreads, float* score_out) {
    const int Q = q - 1;
    const long NQ = static_cast<long>(N) * Q;
    const double invQ = 1.0 / Q;
    const double invQ2 = invQ * invQ;

    std::vector<double> FN(static_cast<size_t>(N) * N, 0.0);

    parallel_for(N, nthreads, [&](long lo, long hi) {
        std::vector<double> blk(static_cast<size_t>(Q) * Q);
        std::vector<double> ra(Q), cb(Q);
        for (long i = lo; i < hi; ++i) {
            for (long j = 0; j < i; ++j) {
                // block(a, b) = Jinv[i*Q + a, j*Q + b]; column-major, so the a
                // index runs contiguously down each column.
                double grand = 0.0;
                std::fill(cb.begin(), cb.end(), 0.0);
                for (int b = 0; b < Q; ++b) {
                    const CovScalar* col = Jlow + (j * Q + b) * NQ + i * Q;
                    double s = 0.0;
                    double* bcol = blk.data() + static_cast<size_t>(b) * Q;
                    for (int a = 0; a < Q; ++a) {
                        const double v = static_cast<double>(col[a]);
                        bcol[a] = v;
                        s += v;
                    }
                    cb[b] = s * invQ;      // mean over a, for this b
                    grand += s;
                }
                grand *= invQ2;
                std::fill(ra.begin(), ra.end(), 0.0);
                for (int b = 0; b < Q; ++b) {
                    const double* bcol = blk.data() + static_cast<size_t>(b) * Q;
                    for (int a = 0; a < Q; ++a) ra[a] += bcol[a];
                }
                for (int a = 0; a < Q; ++a) ra[a] *= invQ;   // mean over b

                double ss = 0.0;
                for (int b = 0; b < Q; ++b) {
                    const double* bcol = blk.data() + static_cast<size_t>(b) * Q;
                    const double cbb = cb[b];
                    for (int a = 0; a < Q; ++a) {
                        const double e = bcol[a] - ra[a] - cbb + grand;
                        ss += e * e;
                    }
                }
                const double fn = std::sqrt(ss);
                FN[static_cast<size_t>(i) * N + j] = fn;
                FN[static_cast<size_t>(j) * N + i] = fn;
            }
        }
    });

    // ---- Average Product Correction ----
    std::vector<double> Si(N, 0.0), Sj(N, 0.0);
    double total = 0.0;
    for (long i = 0; i < N; ++i) {
        double rs = 0.0;
        const double* row = FN.data() + static_cast<size_t>(i) * N;
        for (long j = 0; j < N; ++j) { rs += row[j]; Si[j] += row[j]; }
        Sj[i] = rs;
        total += rs;
    }
    const double Sa = total * (1.0 - 1.0 / static_cast<double>(N));
    parallel_for(N, nthreads, [&](long lo, long hi) {
        for (long i = lo; i < hi; ++i) {
            const double* row = FN.data() + static_cast<size_t>(i) * N;
            float* out = score_out + static_cast<size_t>(i) * N;
            const double si = Sj[i];
            for (long j = 0; j < N; ++j)
                out[j] = static_cast<float>(row[j] - (si * Si[j]) / Sa);
        }
    });
}

// Full pipeline: covariance -> inverse -> Frobenius norm -> APC. Writes the
// (N x N) score row-major into `score_out` as float32. The (NQ x NQ) workspace
// is allocated and released here and is the program's peak allocation.
inline void compute_score_into(const int8_t* Z, int N, int M, const double* W,
                               double Meff, int q, double pseudocount,
                               int n_threads, float* score_out) {
    const long NQ = static_cast<long>(N) * (q - 1);
    const int nthreads = resolve_threads(n_threads);

    AlignedBuffer<CovScalar> ws(static_cast<size_t>(NQ) * NQ);
    CovScalar* C = ws.get();

    build_covariance_lower(Z, N, M, W, Meff, q, pseudocount, nthreads, C);
    invert_spd_lower_inplace(C, NQ, nthreads);
    score_from_inverse_lower(C, N, q, nthreads, score_out);
}

// Full symmetric (NQ x NQ) inverse, written column-major into `out`. Because J
// is symmetric the row-major and column-major views of `out` are identical.
// Used by the pybind entry point and the reference tests.
inline void compute_couplings_into(const int8_t* Z, int N, int M,
                                   const double* W, double Meff, int q,
                                   double pseudocount, int n_threads,
                                   double* out) {
    const long NQ = static_cast<long>(N) * (q - 1);
    const int nthreads = resolve_threads(n_threads);

#ifdef GAUSSDCA_MIXED_PRECISION
    AlignedBuffer<CovScalar> ws(static_cast<size_t>(NQ) * NQ);
    CovScalar* C = ws.get();
#else
    CovScalar* C = out;                    // already the right type; work in place
#endif

    build_covariance_lower(Z, N, M, W, Meff, q, pseudocount, nthreads, C);
    invert_spd_lower_inplace(C, NQ, nthreads);
    mirror_lower_to_upper(C, NQ, nthreads);

#ifdef GAUSSDCA_MIXED_PRECISION
    parallel_for(NQ * NQ, nthreads, [&](long lo, long hi) {
        for (long k = lo; k < hi; ++k) out[k] = static_cast<double>(C[k]);
    });
#endif
}

// Convenience wrapper: allocate the result and fill it.
inline Eigen::MatrixXd compute_couplings(const int8_t* Z, int N, int M,
                                         const double* W, double Meff, int q,
                                         double pseudocount, int n_threads = 0) {
    const long NQ = static_cast<long>(N) * (q - 1);
    Eigen::MatrixXd J(NQ, NQ);
    compute_couplings_into(Z, N, M, W, Meff, q, pseudocount, n_threads, J.data());
    return J;
}

}  // namespace gaussdca
