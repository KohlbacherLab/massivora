/**
 * Minimal executable to test process startup overhead.
 * This program simulates a single site optimization task.
 * 
 * Usage:
 *   ./site_optimizer <data_dir> <site_id> [--time-only]
 *   
 * Output (JSON):
 *   {"site": 0, "startup_ms": 1.5, "load_ms": 50.0, "compute_ms": 800.0, "total_ms": 851.5}
 */

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <nlopt.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cstring>

using PLMV = double;

// High-resolution timer
using Clock = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;

// Record startup time as early as possible
static TimePoint g_programStart = Clock::now();

double elapsedMs(TimePoint start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

double elapsedMs(TimePoint start, TimePoint end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// ============================================================================
// Core computation functions (same as compute.cpp)
// ============================================================================

Eigen::Matrix<PLMV, -1, 1> getEnergies(
    const Eigen::Matrix<PLMV, -1, 1>& x,
    int r, int q, int N, int b,
    const Eigen::MatrixXi& MSA) {

    Eigen::Matrix<PLMV, -1, 1> energies = Eigen::Matrix<PLMV, -1, 1>::Zero(q);
    Eigen::Map<const Eigen::Matrix<PLMV, -1, 1>> h(x.data(), q);
    Eigen::TensorMap<const Eigen::Tensor<PLMV, 3>> Jr(x.data() + q, q, q, N-1);

    for (int l = 0; l < q; l++) {
        PLMV sum_Jri = 0.0;
        for (int i = 0; i < r; i++) {
            sum_Jri += Jr(l, MSA(b, i), i);
        }
        for (int i = r + 1; i < N; i++) {
            sum_Jri += Jr(l, MSA(b, i), i-1);
        }
        energies(l) = h(l) + sum_Jri;
    }
    return energies;
}

PLMV mergeEnergyTerm(const Eigen::Matrix<PLMV, -1, 1>& energies) {
    PLMV max = energies.maxCoeff();
    return max + std::log((energies.array() - max).exp().sum());
}

PLMV l2Regularization(const Eigen::Matrix<PLMV, -1, 1>& vec, int q, PLMV lambdaH, PLMV lambdaJ) {
    return vec.head(q).array().square().sum() * lambdaH +
           vec.tail(vec.size() - q).array().square().sum() * 0.5 * lambdaJ;
}

std::tuple<PLMV, Eigen::Matrix<PLMV, -1, 1>> perSitePllGradient(
    const Eigen::Matrix<PLMV, -1, 1>& x, 
    int r, int q, int N, int B,
    const Eigen::MatrixXi& MSA,
    const Eigen::Matrix<PLMV, -1, 1>& W,
    PLMV lambdaH, PLMV lambdaJ) {

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
            vGrad(s) = W(b) * ((s == MSA(b, r) ? 1 : 0) - Ps(s));
        }

        gradients.head(q) -= vGrad;

        for (int i = 0; i < r; i++) {
            int s_ib = MSA(b, i);
            for (int s = 0; s < q; s++) {
                Tgradients(s, s_ib, i) -= vGrad(s);
            }
        }
        for (int i = r+1; i < N; i++) {
            int s_ib = MSA(b, i);
            for (int s = 0; s < q; s++) {
                Tgradients(s, s_ib, i-1) -= vGrad(s);
            }
        }
    }

    gradients.head(q) += 2.0f * lambdaH * x.head(q);
    gradients.tail(x.size() - q) += lambdaJ * x.tail(x.size() - q);
    pll += l2Regularization(x, q, lambdaH, lambdaJ);
    
    return std::make_tuple(pll, gradients);
}

struct NLoptContext {
    int r, q, N, B;
    const Eigen::MatrixXi* MSA;
    const Eigen::Matrix<PLMV, -1, 1>* W;
    PLMV lambdaH, lambdaJ;
};

double pllTarget(const std::vector<double> &x, std::vector<double> &grad, void* f_data) {
    auto* ctx = reinterpret_cast<NLoptContext*>(f_data);
    Eigen::Map<const Eigen::VectorXd> x_d(x.data(), x.size());
    
    auto [pll, gradients] = perSitePllGradient(
        x_d, ctx->r, ctx->q, ctx->N, ctx->B,
        *(ctx->MSA), *(ctx->W), ctx->lambdaH, ctx->lambdaJ
    );

    if (!grad.empty()) {
        Eigen::Map<Eigen::VectorXd>(grad.data(), grad.size()) = gradients;
    }
    return pll;
}

Eigen::Matrix<PLMV, -1, 1> perSiteNLopt(
    unsigned nParams, int r, int q, int N, int B,
    const Eigen::MatrixXi& MSA,
    const Eigen::Matrix<PLMV, -1, 1>& W,
    PLMV lambdaH, PLMV lambdaJ,
    double eps_conv = 1e-4, int maxeval = 500) {
    
    std::vector<double> x0(nParams, 0.0);
    nlopt::opt opt(nlopt::LD_LBFGS, nParams);
    NLoptContext ctx{r, q, N, B, &MSA, &W, lambdaH, lambdaJ};
    opt.set_min_objective(pllTarget, &ctx);
    opt.set_maxeval(maxeval);
    opt.set_ftol_abs(eps_conv);
    opt.set_xtol_abs(eps_conv);
    
    double opt_f;
    opt.optimize(x0, opt_f);
    
    Eigen::Matrix<PLMV, -1, 1> result(nParams);
    std::memcpy(result.data(), x0.data(), nParams * sizeof(double));
    return result;
}

// ============================================================================
// Data loading
// ============================================================================

struct MSAData {
    Eigen::MatrixXi MSA;
    Eigen::Matrix<PLMV, -1, 1> W;
    int B, N, q;
    double Beff;
};

MSAData loadData(const std::string& dataDir) {
    MSAData data;
    
    std::string metaPath = dataDir + "/meta.bin";
    std::ifstream metaFile(metaPath, std::ios::binary);
    if (!metaFile) throw std::runtime_error("Cannot open " + metaPath);
    
    int meta[3];
    metaFile.read(reinterpret_cast<char*>(meta), 3 * sizeof(int));
    metaFile.read(reinterpret_cast<char*>(&data.Beff), sizeof(double));
    data.B = meta[0]; data.N = meta[1]; data.q = meta[2];
    
    std::string msaPath = dataDir + "/msa.bin";
    std::ifstream msaFile(msaPath, std::ios::binary);
    if (!msaFile) throw std::runtime_error("Cannot open " + msaPath);
    
    std::vector<int32_t> msaBuffer(data.B * data.N);
    msaFile.read(reinterpret_cast<char*>(msaBuffer.data()), data.B * data.N * sizeof(int32_t));
    Eigen::Map<Eigen::Matrix<int32_t, -1, -1, Eigen::RowMajor>> msaMap(msaBuffer.data(), data.B, data.N);
    data.MSA = msaMap.cast<int>();
    
    std::string wPath = dataDir + "/weights.bin";
    std::ifstream wFile(wPath, std::ios::binary);
    if (!wFile) throw std::runtime_error("Cannot open " + wPath);
    
    data.W.resize(data.B);
    wFile.read(reinterpret_cast<char*>(data.W.data()), data.B * sizeof(double));
    data.W /= data.Beff;
    
    return data;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    TimePoint afterStartup = Clock::now();
    
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <data_dir> <site_id> [--time-only] [--no-compute]" << std::endl;
        return 1;
    }
    
    std::string dataDir = argv[1];
    int siteId = std::atoi(argv[2]);
    bool timeOnly = false;
    bool noCompute = false;
    
    for (int i = 3; i < argc; i++) {
        if (std::string(argv[i]) == "--time-only") timeOnly = true;
        if (std::string(argv[i]) == "--no-compute") noCompute = true;
    }
    
    double lambdaH = 0.01, lambdaJ = 0.01;
    
    // Measure startup time (from program start to after argument parsing)
    double startupMs = elapsedMs(g_programStart, afterStartup);
    
    // Load data
    TimePoint loadStart = Clock::now();
    MSAData data = loadData(dataDir);
    double loadMs = elapsedMs(loadStart);
    
    // Compute
    double computeMs = 0.0;
    if (!noCompute) {
        unsigned nParams = data.q + (data.N - 1) * data.q * data.q;
        
        TimePoint computeStart = Clock::now();
        auto result = perSiteNLopt(
            nParams, siteId, data.q, data.N, data.B,
            data.MSA, data.W, lambdaH, lambdaJ,
            1e-4, 500
        );
        computeMs = elapsedMs(computeStart);
        
        // Optionally output result checksum to prevent optimization
        if (!timeOnly) {
            std::cerr << "Result sum: " << result.sum() << std::endl;
        }
    }
    
    double totalMs = elapsedMs(g_programStart);
    
    // Output timing as JSON
    std::cout << "{\"site\": " << siteId 
              << ", \"startup_ms\": " << startupMs
              << ", \"load_ms\": " << loadMs
              << ", \"compute_ms\": " << computeMs
              << ", \"total_ms\": " << totalMs
              << ", \"B\": " << data.B
              << ", \"N\": " << data.N
              << ", \"q\": " << data.q
              << "}" << std::endl;
    
    return 0;
}
