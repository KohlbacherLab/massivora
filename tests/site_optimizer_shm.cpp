/**
 * Site optimizer using shared memory for data input.
 * 
 * Usage:
 *   ./site_optimizer_shm --shm <prefix> <site_id>
 *   ./site_optimizer_shm <data_dir> <site_id>  (fallback to file mode)
 * 
 * Shared memory segments (created by Python):
 *   <prefix>_meta:   B, N, q (int32), Beff (float64)
 *   <prefix>_msa:    MSA matrix (int32, row-major, B x N)
 *   <prefix>_w:      W vector (float64, B)
 *   <prefix>_result: Result vector (float64, nParams) - for output
 */

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <nlopt.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cstring>
#include <string>

// Shared memory headers
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

using PLMV = double;

using Clock = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;

static TimePoint g_programStart = Clock::now();

double elapsedMs(TimePoint start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

double elapsedMs(TimePoint start, TimePoint end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// ============================================================================
// Core computation functions
// ============================================================================

Eigen::Matrix<PLMV, -1, 1> getEnergies(
    const Eigen::Matrix<PLMV, -1, 1>& x,
    int r, int q, int N, int b,
    const Eigen::Map<Eigen::Matrix<int, -1, -1, Eigen::RowMajor>>& MSA) {

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

struct NLoptContext {
    int r, q, N, B;
    const Eigen::Map<Eigen::Matrix<int, -1, -1, Eigen::RowMajor>>* MSA;
    const Eigen::Map<Eigen::Matrix<PLMV, -1, 1>>* W;
    PLMV lambdaH, lambdaJ;
};

std::tuple<PLMV, Eigen::Matrix<PLMV, -1, 1>> perSitePllGradient(
    const Eigen::Matrix<PLMV, -1, 1>& x, 
    int r, int q, int N, int B,
    const Eigen::Map<Eigen::Matrix<int, -1, -1, Eigen::RowMajor>>& MSA,
    const Eigen::Map<Eigen::Matrix<PLMV, -1, 1>>& W,
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
    const Eigen::Map<Eigen::Matrix<int, -1, -1, Eigen::RowMajor>>& MSA,
    const Eigen::Map<Eigen::Matrix<PLMV, -1, 1>>& W,
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
// Shared memory helpers
// ============================================================================

struct ShmMapping {
    void* ptr = nullptr;
    size_t size = 0;
    int fd = -1;
    
    ~ShmMapping() {
        if (ptr && ptr != MAP_FAILED) {
            munmap(ptr, size);
        }
        if (fd >= 0) {
            close(fd);
        }
    }
};

ShmMapping openShm(const std::string& name, size_t expected_size = 0) {
    ShmMapping m;
    
    std::string shm_path = "/" + name;
    m.fd = shm_open(shm_path.c_str(), O_RDWR, 0666);
    if (m.fd < 0) {
        throw std::runtime_error("Failed to open shared memory: " + name);
    }
    
    struct stat sb;
    if (fstat(m.fd, &sb) < 0) {
        throw std::runtime_error("Failed to stat shared memory: " + name);
    }
    m.size = sb.st_size;
    
    if (expected_size > 0 && m.size < expected_size) {
        throw std::runtime_error("Shared memory too small: " + name);
    }
    
    m.ptr = mmap(nullptr, m.size, PROT_READ | PROT_WRITE, MAP_SHARED, m.fd, 0);
    if (m.ptr == MAP_FAILED) {
        throw std::runtime_error("Failed to mmap shared memory: " + name);
    }
    
    return m;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    TimePoint afterStartup = Clock::now();
    
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " --shm <prefix> <site_id>" << std::endl;
        std::cerr << "   or: " << argv[0] << " <data_dir> <site_id>" << std::endl;
        return 1;
    }
    
    bool useShm = (std::string(argv[1]) == "--shm");
    std::string prefix_or_dir = argv[2];
    int siteId = std::atoi(argv[useShm ? 3 : 2]);
    
    if (useShm && argc < 4) {
        std::cerr << "Usage: " << argv[0] << " --shm <prefix> <site_id>" << std::endl;
        return 1;
    }
    
    double lambdaH = 0.01, lambdaJ = 0.01;
    double startupMs = elapsedMs(g_programStart, afterStartup);
    
    int B, N, q;
    double Beff;
    double shmLoadMs = 0.0;
    double computeMs = 0.0;
    
    if (useShm) {
        // Shared memory mode
        TimePoint shmStart = Clock::now();
        
        // Open metadata
        ShmMapping metaShm = openShm(prefix_or_dir + "_meta", 20);
        char* metaPtr = static_cast<char*>(metaShm.ptr);
        std::memcpy(&B, metaPtr, 4);
        std::memcpy(&N, metaPtr + 4, 4);
        std::memcpy(&q, metaPtr + 8, 4);
        std::memcpy(&Beff, metaPtr + 12, 8);
        
        // Open MSA (map directly, no copy!)
        ShmMapping msaShm = openShm(prefix_or_dir + "_msa", B * N * sizeof(int32_t));
        Eigen::Map<Eigen::Matrix<int, -1, -1, Eigen::RowMajor>> MSA(
            static_cast<int*>(msaShm.ptr), B, N);
        
        // Open W (map directly, no copy!)
        ShmMapping wShm = openShm(prefix_or_dir + "_w", B * sizeof(double));
        Eigen::Map<Eigen::Matrix<PLMV, -1, 1>> W_raw(
            static_cast<double*>(wShm.ptr), B);
        
        // Normalize W (need a copy for this)
        Eigen::Matrix<PLMV, -1, 1> W = W_raw / Beff;
        Eigen::Map<Eigen::Matrix<PLMV, -1, 1>> W_map(W.data(), B);
        
        shmLoadMs = elapsedMs(shmStart);
        
        // Compute
        unsigned nParams = q + (N - 1) * q * q;
        
        TimePoint computeStart = Clock::now();
        auto result = perSiteNLopt(
            nParams, siteId, q, N, B,
            MSA, W_map, lambdaH, lambdaJ,
            1e-4, 500
        );
        computeMs = elapsedMs(computeStart);
        
        // Write result to shared memory
        ShmMapping resultShm = openShm(prefix_or_dir + "_result", nParams * sizeof(double));
        std::memcpy(resultShm.ptr, result.data(), nParams * sizeof(double));
        
    } else {
        // File mode (same as site_optimizer.cpp)
        TimePoint loadStart = Clock::now();
        
        std::string metaPath = prefix_or_dir + "/meta.bin";
        std::ifstream metaFile(metaPath, std::ios::binary);
        if (!metaFile) throw std::runtime_error("Cannot open " + metaPath);
        
        int meta[3];
        metaFile.read(reinterpret_cast<char*>(meta), 12);
        metaFile.read(reinterpret_cast<char*>(&Beff), 8);
        B = meta[0]; N = meta[1]; q = meta[2];
        
        std::string msaPath = prefix_or_dir + "/msa.bin";
        std::ifstream msaFile(msaPath, std::ios::binary);
        if (!msaFile) throw std::runtime_error("Cannot open " + msaPath);
        
        std::vector<int32_t> msaBuffer(B * N);
        msaFile.read(reinterpret_cast<char*>(msaBuffer.data()), B * N * sizeof(int32_t));
        Eigen::Map<Eigen::Matrix<int, -1, -1, Eigen::RowMajor>> MSA(msaBuffer.data(), B, N);
        
        std::string wPath = prefix_or_dir + "/weights.bin";
        std::ifstream wFile(wPath, std::ios::binary);
        if (!wFile) throw std::runtime_error("Cannot open " + wPath);
        
        Eigen::Matrix<PLMV, -1, 1> W(B);
        wFile.read(reinterpret_cast<char*>(W.data()), B * sizeof(double));
        W /= Beff;
        Eigen::Map<Eigen::Matrix<PLMV, -1, 1>> W_map(W.data(), B);
        
        shmLoadMs = elapsedMs(loadStart);  // Actually file load time
        
        // Compute
        unsigned nParams = q + (N - 1) * q * q;
        
        TimePoint computeStart = Clock::now();
        auto result = perSiteNLopt(
            nParams, siteId, q, N, B,
            MSA, W_map, lambdaH, lambdaJ,
            1e-4, 500
        );
        computeMs = elapsedMs(computeStart);
    }
    
    double totalMs = elapsedMs(g_programStart);
    
    // Output JSON
    std::cout << "{\"site\": " << siteId
              << ", \"startup_ms\": " << startupMs
              << ", \"shm_load_ms\": " << shmLoadMs
              << ", \"compute_ms\": " << computeMs
              << ", \"total_ms\": " << totalMs
              << ", \"mode\": \"" << (useShm ? "shm" : "file") << "\""
              << ", \"B\": " << B
              << ", \"N\": " << N
              << ", \"q\": " << q
              << "}" << std::endl;
    
    return 0;
}
