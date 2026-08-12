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
//

#pragma once

#include <Eigen/Dense>
#include <Eigen/Cholesky>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace gaussdca {

// The couplings scalar is a template parameter of compute_couplings_into: the
// pybind path uses double for full accuracy while gauss_infer uses float to 
// publish J in single precision

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
// them. Without these a threaded BLAS would silently ignore n_threads and grab
// every core, which breaks the executor's resource accounting.
extern "C" void openblas_set_num_threads(int) __attribute__((weak));
extern "C" void mkl_set_num_threads(int*) __attribute__((weak));
#endif

// Thread count used to parallelize the pair-frequency accumulation.
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

// Invert the SPD matrix held in the lower triangle of `A` 
// Valid for maximum ~2317 alignment columns.
inline constexpr long LAPACK_LP64_MAX_N = 46340L;

template <class T>
inline void invert_spd_lower_inplace(T* A, long n, int nthreads) {
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
    using CovMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    Eigen::Map<CovMatrix> C(A, n, n);
    Eigen::LLT<Eigen::Ref<CovMatrix>> llt(C);
    if (llt.info() != Eigen::Success)
        throw std::runtime_error("Cholesky factorization failed (C not SPD)");
    CovMatrix Linv = CovMatrix::Identity(n, n);
    C.template triangularView<Eigen::Lower>().solveInPlace(Linv);
    C.setZero();
    C.template selfadjointView<Eigen::Lower>().rankUpdate(Linv.transpose());
}

// Assemble the covariance C = fij - fi*fi^T (with pseudocount) from a residue-major
// alignment and weights, then overwrite it with the couplings J = C^-1. 
template <class T>
inline void compute_couplings_into(const int8_t* Z, int N, int M,
                                   const double* W, double Meff, int q,
                                   double pseudocount, int n_threads,
                                   T* out) {
    const int Q = q - 1;
    const int QP = Q + 1;
    const long NQ = static_cast<long>(N) * Q;     // covariance dimension
    const int nthreads = resolve_threads(n_threads);

    // ---- recode: [0, Q-1] for real symbols, Q for anything gap-like ----
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
    std::vector<std::pair<int, int>> pairs;
    pairs.reserve(static_cast<size_t>(N) * (N - 1) / 2);
    for (int i = 0; i < N; ++i)
        for (int j = i + 1; j < N; ++j) pairs.emplace_back(i, j);

    const int npair_threads =
        std::min(nthreads, std::max<int>(1, static_cast<int>(pairs.size())));
    T* C = out;                                  // covariance assembled in place
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
                T* dst = C + (c0 + a) * NQ + r0;
                const double fa = fip[c0 + a];
                const double* arow = a_ + static_cast<size_t>(a) * QP;
                for (int b = 0; b < Q; ++b)
                    dst[b] = static_cast<T>(
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
                T* dst = C + (x0 + a) * NQ + x0;
                const double fa = fip[x0 + a];
                for (int b = a; b < Q; ++b)          // lower triangle of the block
                    dst[b] = static_cast<T>(
                        (a == b ? fip[x0 + a] : 0.0) - fip[x0 + b] * fa);
            }
        }
    });

    // ---- couplings J = C^-1, in place on `out` ----
    invert_spd_lower_inplace(C, NQ, nthreads);

    // ---- mirror the lower triangle into the upper one ----
    parallel_for(NQ, nthreads, [&](long lo, long hi) {
        for (long c = lo; c < hi; ++c) {
            const T* src = out + c * NQ;             // column c, rows c..NQ-1
            for (long r = c + 1; r < NQ; ++r) out[r * NQ + c] = src[r];
        }
    });
}

// Convenience wrapper: allocate the result and fill it. Kept for callers that
// want an owning matrix (the pybind entry point in bindings.cpp).
inline Eigen::MatrixXd compute_couplings(const int8_t* Z, int N, int M,
                                         const double* W, double Meff, int q,
                                         double pseudocount, int n_threads = 0) {
    const long NQ = static_cast<long>(N) * (q - 1);
    Eigen::MatrixXd J(NQ, NQ);
    compute_couplings_into(Z, N, M, W, Meff, q, pseudocount, n_threads, J.data());
    return J;
}

}  // namespace gaussdca
