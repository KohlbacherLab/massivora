// gauss_dca.hpp
//
// GaussDCA covariance-inverse core (Baldassi et al. 2014, "Fast and Accurate
// Multivariate Gaussian Modeling of Protein Families", PLOS ONE 9(3):e92721;
//
// Naming follows the plmDCA codebase and the paper's formulas:
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
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace gaussdca {

// Thread count used to parallelize the pair-frequency accumulation.
inline int hardware_threads() {
    unsigned n = std::thread::hardware_concurrency();
    return n == 0 ? 1 : static_cast<int>(n);
}

// Assemble the covariance C = fij - fi*fi^T (with pseudocount)
inline Eigen::MatrixXd compute_couplings(const int8_t* Z, int N, int M,
                                         const double* W, double Meff, int q,
                                         double pseudocount, int n_threads = 0) {
    const int Q = q - 1;
    const long NQ = static_cast<long>(N) * Q;

    // ---- fi: single-site frequencies ----
    Eigen::VectorXd fi = Eigen::VectorXd::Zero(NQ);
    for (int i = 0; i < N; ++i) {
        const int8_t* Zi = Z + static_cast<size_t>(i) * M;
        double* fi_i = fi.data() + static_cast<long>(i) * Q;
        for (int k = 0; k < M; ++k) {
            int a = Zi[k];
            if (a >= 1 && a <= Q) fi_i[a - 1] += W[k];
        }
    }
    fi /= Meff;

    // ---- fij: pair frequencies, upper blocks mirrored ----
    Eigen::MatrixXd fij = Eigen::MatrixXd::Zero(NQ, NQ);

    // Parallelize over the (i<j) residue-pair space; threads write disjoint blocks.
    std::vector<std::pair<int, int>> pairs;
    pairs.reserve(static_cast<size_t>(N) * (N - 1) / 2);
    for (int i = 0; i < N; ++i)
        for (int j = i + 1; j < N; ++j) pairs.emplace_back(i, j);

    int req = (n_threads > 0) ? n_threads : hardware_threads();
    int nthreads = std::min(req, std::max<int>(1, static_cast<int>(pairs.size())));
    if (nthreads < 1) nthreads = 1;
    auto worker = [&](size_t lo, size_t hi) {
        for (size_t idx = lo; idx < hi; ++idx) {
            int i = pairs[idx].first, j = pairs[idx].second;
            const int8_t* Zi = Z + static_cast<size_t>(i) * M;
            const int8_t* Zj = Z + static_cast<size_t>(j) * M;
            // accumulate the Q x Q pair-frequency block for residues (i, j)
            Eigen::MatrixXd block = Eigen::MatrixXd::Zero(Q, Q);
            for (int k = 0; k < M; ++k) {
                int a = Zi[k], b = Zj[k];
                if (a >= 1 && a <= Q && b >= 1 && b <= Q)
                    block(a - 1, b - 1) += W[k];
            }
            block /= Meff;
            long r0 = static_cast<long>(i) * Q;
            long c0 = static_cast<long>(j) * Q;
            fij.block(r0, c0, Q, Q) = block;
            fij.block(c0, r0, Q, Q) = block.transpose();
        }
    };
    {
        std::vector<std::thread> pool;
        size_t chunk = (pairs.size() + nthreads - 1) / nthreads;
        for (int p = 0; p < nthreads; ++p) {
            size_t lo = static_cast<size_t>(p) * chunk;
            size_t hi = std::min(pairs.size(), lo + chunk);
            if (lo < hi) pool.emplace_back(worker, lo, hi);
        }
        for (auto& th : pool) th.join();
    }
    // Main diagonal holds fi (diagonal blocks are otherwise zero).
    for (long x = 0; x < NQ; ++x) fij(x, x) = fi(x);

    // ---- pseudocount: mix with the uniform prior ----
    const double lambda = pseudocount;
    const double unif_i = lambda / q;
    fij.array() = (1.0 - lambda) * fij.array() + unif_i / q;
    fi.array() = (1.0 - lambda) * fi.array() + unif_i;
    for (int i = 0; i < N; ++i) {
        long x0 = static_cast<long>(i) * Q;
        fij.block(x0, x0, Q, Q).setZero();
        for (int a = 0; a < Q; ++a) fij(x0 + a, x0 + a) = fi(x0 + a);
    }

    // ---- covariance C = fij - fi*fi^T ; reuse the buffer ----
    Eigen::MatrixXd& C = fij;
    C.noalias() -= fi * fi.transpose();

    // ---- couplings J = C^-1 = inv(cholesky(C)) ----
    Eigen::LLT<Eigen::MatrixXd> llt(C);
    if (llt.info() != Eigen::Success)
        throw std::runtime_error("Cholesky factorization failed (C not SPD)");
    Eigen::MatrixXd J = llt.solve(Eigen::MatrixXd::Identity(NQ, NQ));
    return J;
}

}  // namespace gaussdca
