// Check that the blocked CPU plmDCA objective (plm_objective.hpp) gives
// bitwise the same objective and gradient as the sequence-at-a-time loop it
// replaced. Synthetic alignment; B is not a multiple of the block, and sites
// at both ends and in the middle are tried.
//   g++ -O3 -march=native -std=c++17 -I<eigen3> -I../src/massivora/bindings_src \
//       test_plm_objective.cpp && ./a.out
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <cstdio>
#include <cstring>
#include <random>
#include <tuple>
#include <vector>

#include "plm_objective.hpp"

using PLMV = double;
using MSAScalar = signed char;
using MSAMap = Eigen::Map<const Eigen::Matrix<MSAScalar, -1, -1, Eigen::RowMajor>>;
using WMap   = Eigen::Map<const Eigen::Matrix<PLMV, -1, 1>>;

// The sequence-at-a-time objective plm_opt_site.cpp used before plm_objective.hpp.
Eigen::Matrix<PLMV, -1, 1> getEnergies(const Eigen::Matrix<PLMV, -1, 1>& x, int r, int q,
                                       int N, int b, const MSAMap& MSA) {
    Eigen::Matrix<PLMV, -1, 1> energies = Eigen::Matrix<PLMV, -1, 1>::Zero(q);
    Eigen::Map<const Eigen::Matrix<PLMV, -1, 1>> h(x.data(), q);
    Eigen::TensorMap<const Eigen::Tensor<PLMV, 3>> Jr(x.data() + q, q, q, N-1);
    for (int l = 0; l < q; l++) {
        PLMV sum_Jri = 0.0;
        for (int i = 0; i < r; i++) { int k = MSA(b, i); sum_Jri += Jr(l, k, i); }
        for (int i = r + 1; i < N; i++) { int k = MSA(b, i); sum_Jri += Jr(l, k, i-1); }
        energies(l) = h(l) + sum_Jri;
    }
    return energies;
}
PLMV mergeEnergyTerm(const Eigen::Matrix<PLMV, -1, 1>& energies) {
    PLMV max = energies.maxCoeff();
    return max + std::log((energies.array() - max).exp().sum());
}
PLMV l2Regularization(const Eigen::Matrix<PLMV, -1, 1>& vec, int q, PLMV lambdaH, PLMV lambdaJ) {
    PLMV reg_h = vec.head(q).array().square().sum() * lambdaH;
    PLMV reg_J = vec.tail(vec.size() - q).array().square().sum() * 0.5 * lambdaJ;
    return reg_h + reg_J;
}
std::tuple<PLMV, Eigen::Matrix<PLMV, -1, 1>> perSitePllGradient(
    const Eigen::Matrix<PLMV, -1, 1>& x, int r, int q, int N, int B,
    const MSAMap& MSA, const WMap& W, PLMV lambdaH, PLMV lambdaJ) {
    PLMV pll = 0.0;
    Eigen::Matrix<PLMV, -1, 1> gradients = Eigen::Matrix<PLMV, -1, 1>::Zero(x.size());
    Eigen::TensorMap<Eigen::Tensor<PLMV, 3>> Tgradients(gradients.data() + q, q, q, N - 1);
    for (int b = 0; b < B; b++) {
        Eigen::Matrix<PLMV, -1, 1> energies = getEnergies(x, r, q, N, b, MSA);
        PLMV lnorm = mergeEnergyTerm(energies);
        pll -= W(b) * (energies(MSA(b, r)) - lnorm);
        Eigen::Matrix<PLMV, -1, 1> Ps = (energies.array() - lnorm).exp();
        Eigen::Matrix<PLMV, -1, 1> vGrad = Eigen::Matrix<PLMV, -1, 1>::Zero(q);
        for (int s = 0; s < q; s++) {
            int indicator = (s == MSA(b, r)) ? 1 : 0;
            vGrad(s) = W(b) * (indicator - Ps(s));
        }
        gradients.head(q) -= vGrad;
        for (int i = 0; i < r; i++) {
            int s_ib = MSA(b, i);
            for (int s = 0; s < q; s++) Tgradients(s, s_ib, i) -= vGrad(s);
        }
        for (int i = r+1; i < N; i++) {
            int s_ib = MSA(b, i);
            for (int s = 0; s < q; s++) Tgradients(s, s_ib, i-1) -= vGrad(s);
        }
    }
    gradients.head(q) += 2.0f * lambdaH * x.head(q);
    gradients.tail(x.size() - q) += lambdaJ * x.tail(x.size() - q);
    pll += l2Regularization(x, q, lambdaH, lambdaJ);
    return std::make_tuple(pll, gradients);
}

int main() {
    const int B = 203, N = 57, q = 21;
    std::mt19937_64 rng(7);
    std::uniform_int_distribution<int> state(0, q - 1);
    std::normal_distribution<double> nd(0.0, 0.05);
    std::vector<MSAScalar> msa((size_t)B * N);
    for (auto& z : msa) z = (MSAScalar)state(rng);
    std::vector<PLMV> w(B);
    for (auto& x : w) x = 1.0 / B + 1e-4 * nd(rng);
    MSAMap MSA(msa.data(), B, N);
    WMap W(w.data(), B);
    const size_t n = (size_t)(N - 1) * q * q + q;
    int bad = 0;
    for (int r : {0, 1, N / 2, N - 1}) {
        Eigen::VectorXd x(n);
        for (size_t k = 0; k < n; ++k) x[k] = nd(rng);
        auto [fa, ga] = perSitePllGradient(x, r, q, N, B, MSA, W, 0.01, 0.01);
        std::vector<double> gb(n + 1);      // deliberately not 64-byte aligned
        const PLMV fb = plmobj::pllGradient(x.data(), r, q, N, B, msa.data(), w.data(),
                                            0.01, 0.01, gb.data() + 1);
        const bool same = std::memcmp(&fa, &fb, sizeof fa) == 0 &&
                          std::memcmp(ga.data(), gb.data() + 1, n * sizeof(PLMV)) == 0;
        std::printf("site %d: %s\n", r, same ? "identical" : "DIFFERENT");
        bad += !same;
    }
    return bad != 0;
}
