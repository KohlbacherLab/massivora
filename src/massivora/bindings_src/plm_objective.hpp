// plm_objective.hpp -- the CPU plmDCA objective of one site, blocked over
// sequences.
//
// For every sequence b the objective reads one q-column of J per position
// (energies) and updates one q-column of the gradient per position. Done one
// sequence at a time that is N scattered column reads and N read-modify-writes
// across the whole (N-1) q^2 parameter vector -- several MB, L3-resident --
// per sequence. Taking the positions in the outer loop and a block of
// sequences in the inner one touches each position's q x q slab of J and of
// the gradient once per block, from L1.
//
// The result is bitwise identical to the sequence-at-a-time loop: every
// energy still sums its positions in increasing order, every gradient entry
// still receives its sequences in increasing order, and the per-sequence
// normalisation, the regularisation and the final updates are the same Eigen
// expressions on Eigen-allocated (so equally aligned) vectors.
#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <vector>

namespace plmobj {

using PLMV = double;

// Sequences per block: the energy accumulators (block x q doubles, 10.5 KB at
// q = 21) and one slab each of J and the gradient (q^2 doubles, 3.5 KB) stay
// in L1.
constexpr int kSeqBlock = 64;

// acc[b][l] += J(l, z_bi, i) over the positions i != r of a block of nb
// sequences. QC > 0 fixes q at compile time, so the q-long rows need no
// runtime trip count and tail loop.
template <int QC>
inline void block_energies(const PLMV* J, const signed char* Z, int N, int r, int nb,
                           int q_rt, PLMV* acc) {
    const int q = QC > 0 ? QC : q_rt;
    const long slab = (long)q * q;
    for (int i = 0; i < N; ++i) {
        if (i == r) continue;
        const PLMV* Ji = J + (long)(i < r ? i : i - 1) * slab;
        for (int bb = 0; bb < nb; ++bb) {
            const PLMV* col = Ji + (long)Z[(size_t)bb * N + i] * q;
            PLMV* a = acc + (size_t)bb * q;
            for (int l = 0; l < q; ++l) a[l] += col[l];
        }
    }
}

// G(s, z_bi, i) -= vg[b][s] over the positions i != r, sequences in order.
template <int QC>
inline void block_gradient(PLMV* G, const signed char* Z, int N, int r, int nb,
                           int q_rt, const PLMV* vg) {
    const int q = QC > 0 ? QC : q_rt;
    const long slab = (long)q * q;
    for (int i = 0; i < N; ++i) {
        if (i == r) continue;
        PLMV* Gi = G + (long)(i < r ? i : i - 1) * slab;
        for (int bb = 0; bb < nb; ++bb) {
            PLMV* g = Gi + (long)Z[(size_t)bb * N + i] * q;
            const PLMV* v = vg + (size_t)bb * q;
            for (int s = 0; s < q; ++s) g[s] -= v[s];
        }
    }
}

// The two scalar pieces keep the shape they had in plm_opt_site.cpp: GCC
// fuses multiply-adds (-ffp-contract=fast) by the shape of the code, so the
// same expression in a different shape can round differently in the last bit.
inline PLMV mergeEnergyTerm(const Eigen::Matrix<PLMV, -1, 1>& energies) {
    PLMV max = energies.maxCoeff();
    return max + std::log((energies.array() - max).exp().sum());
}

inline PLMV l2Regularization(const Eigen::Matrix<PLMV, -1, 1>& vec, int q,
                             PLMV lambdaH, PLMV lambdaJ) {
    PLMV reg_h = vec.head(q).array().square().sum() * lambdaH;
    PLMV reg_J = vec.tail(vec.size() - q).array().square().sum() * 0.5 * lambdaJ;
    return reg_h + reg_J;
}

// Objective of site r and its gradient. x, grad: (N-1) q^2 + q values,
// h first, then J as (q, q, N-1) column-major. MSA: (B, N) row-major.
inline PLMV pllGradient(const PLMV* x_in, int r, int q, int N, int B,
                        const signed char* MSA, const PLMV* W,
                        PLMV lambdaH, PLMV lambdaJ, PLMV* grad_out) {
    using Vec = Eigen::Matrix<PLMV, -1, 1>;
    const Eigen::Index n = (Eigen::Index)(N - 1) * q * q + q;
    const Vec x = Eigen::Map<const Vec>(x_in, n);   // aligned, as the reductions expect
    Eigen::Map<Vec> gradients(grad_out, n);
    gradients.setZero();
    const PLMV* J = x.data() + q;
    PLMV* G = gradients.data() + q;

    std::vector<PLMV> acc((size_t)kSeqBlock * q), vg((size_t)kSeqBlock * q);
    Vec energies(q), Ps(q), vGrad(q);
    PLMV pll = 0.0;

    for (int b0 = 0; b0 < B; b0 += kSeqBlock) {
        const int nb = std::min(kSeqBlock, B - b0);
        const signed char* Z = MSA + (size_t)b0 * N;

        // energies of the block: acc[b][l] = sum over i != r of J(l, z_bi, i)
        std::fill(acc.begin(), acc.begin() + (size_t)nb * q, 0.0);
        if (q == 21) block_energies<21>(J, Z, N, r, nb, q, acc.data());
        else         block_energies<0>(J, Z, N, r, nb, q, acc.data());

        for (int bb = 0; bb < nb; ++bb) {
            const int b = b0 + bb;
            const int zr = Z[(size_t)bb * N + r];
            for (int l = 0; l < q; ++l) energies(l) = x(l) + acc[(size_t)bb * q + l];
            const PLMV lnorm = mergeEnergyTerm(energies);
            pll -= W[b] * (energies(zr) - lnorm);
            Ps = (energies.array() - lnorm).exp();
            for (int s = 0; s < q; s++) {
                const int indicator = (s == zr) ? 1 : 0;
                vGrad(s) = W[b] * (indicator - Ps(s));
            }
            gradients.head(q) -= vGrad;
            std::copy(vGrad.data(), vGrad.data() + q, vg.data() + (size_t)bb * q);
        }

        // gradient of the block: G(s, z_bi, i) -= vGrad_b(s), b in order
        if (q == 21) block_gradient<21>(G, Z, N, r, nb, q, vg.data());
        else         block_gradient<0>(G, Z, N, r, nb, q, vg.data());
    }

    gradients.head(q) += 2.0f * lambdaH * x.head(q);
    gradients.tail(n - q) += lambdaJ * x.tail(n - q);
    pll += l2Regularization(x, q, lambdaH, lambdaJ);

    return pll;
}

}  // namespace plmobj
