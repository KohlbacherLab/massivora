/**
 * Standalone C++ test program for memory usage analysis.
 * This bypasses Python/pybind11 to verify if memory issues are caused by the binding layer.
 * 
 * Compile:
 *   g++ -std=c++17 -O2 -fopenmp -I/path/to/eigen -I/path/to/nlopt/include \
 *       test_memory.cpp -o test_memory -lnlopt -lpthread
 * 
 * Or with CMake (see CMakeLists.txt in this directory)
 * 
 * Usage:
 *   ./test_memory <data_dir> [num_threads]
 *   
 *   data_dir: directory containing meta.bin, msa.bin, weights.bin (from export_test_data.py)
 *   num_threads: number of parallel threads (default: 256)
 */

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <nlopt.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <chrono>
#include <atomic>
#include <mutex>
#include <cstring>
#include <iomanip>

#ifdef __APPLE__
#include <mach/mach.h>
#elif __linux__
#include <fstream>
#include <sstream>
#include <unistd.h>
#endif

using PLMV = double;

// ============================================================================
// Memory monitoring utilities
// ============================================================================

size_t getCurrentRSS() {
#ifdef __APPLE__
    struct mach_task_basic_info info;
    mach_msg_type_number_t size = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &size) == KERN_SUCCESS) {
        return info.resident_size;
    }
    return 0;
#elif __linux__
    std::ifstream statm("/proc/self/statm");
    if (statm.is_open()) {
        size_t size, resident;
        statm >> size >> resident;
        return resident * sysconf(_SC_PAGESIZE);
    }
    return 0;
#else
    return 0;
#endif
}

std::string formatBytes(size_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB"};
    int unit = 0;
    double value = static_cast<double>(bytes);
    while (value >= 1024 && unit < 3) {
        value /= 1024;
        unit++;
    }
    char buf[64];
    snprintf(buf, sizeof(buf), "%.2f %s", value, units[unit]);
    return std::string(buf);
}

// ============================================================================
// Core computation functions (copied from compute.cpp)
// ============================================================================

Eigen::Matrix<PLMV, -1, 1> getEnergies(
    const Eigen::Matrix<PLMV, -1, 1>& x,
    int r,
    int q, 
    int N,
    int b,
    const Eigen::MatrixXi& MSA) {

    Eigen::Matrix<PLMV, -1, 1> energies = Eigen::Matrix<PLMV, -1, 1>::Zero(q);

    Eigen::Map<const Eigen::Matrix<PLMV, -1, 1>> h(x.data(), q);
    Eigen::TensorMap<const Eigen::Tensor<PLMV, 3>> Jr(x.data() + q, q, q, N-1);

    for (int l = 0; l < q; l++) {
        PLMV sum_Jri = 0.0;

        for (int i = 0; i < r; i++) {
            int k = MSA(b, i);
            sum_Jri += Jr(l, k, i);
        }
        for (int i = r + 1; i < N; i++) {
            int k = MSA(b, i);
            sum_Jri += Jr(l, k, i-1);
        }

        energies(l) = h(l) + sum_Jri;
    }

    return energies;
}

PLMV mergeEnergyTerm(const Eigen::Matrix<PLMV, -1, 1>& energies) {
    PLMV max = energies.maxCoeff();
    return max + std::log((energies.array() - max).exp().sum());
}

PLMV l2Regularization(
    const Eigen::Matrix<PLMV, -1, 1>& vec, 
    int q,
    PLMV lambdaH,
    PLMV lambdaJ) {

    PLMV reg_h = vec.head(q).array().square().sum() * lambdaH;
    PLMV reg_J = vec.tail(vec.size() - q).array().square().sum() * 0.5 * lambdaJ;

    return reg_h + reg_J;
}

std::tuple<PLMV, Eigen::Matrix<PLMV, -1, 1>> perSitePllGradient(
    const Eigen::Matrix<PLMV, -1, 1>& x, 
    int r,
    int q,
    int N,
    int B,
    const Eigen::MatrixXi& MSA,
    const Eigen::Matrix<PLMV, -1, 1>& W,
    PLMV lambdaH,
    PLMV lambdaJ) {

    PLMV pll = 0.0;
    Eigen::Matrix<PLMV, -1, 1> gradients = Eigen::Matrix<PLMV, -1, 1>::Zero(x.size());
    Eigen::TensorMap<Eigen::Tensor<PLMV, 3>> Tgradients(gradients.data() + q, q, q, N - 1);

    for (int b = 0; b < B; b++) {
        Eigen::Matrix<PLMV, -1, 1> energies = getEnergies(x, r, q, N, b, MSA);
        PLMV lnorm = mergeEnergyTerm(energies);
        pll -= W(b) * (energies(MSA(b, r)) - lnorm);

        Eigen::Matrix<PLMV, -1, 1> Ps = energies.array() - lnorm;
        Ps = Ps.array().exp();

        Eigen::Matrix<PLMV, -1, 1> vGrad = Eigen::Matrix<PLMV, -1, 1>::Zero(q);
        for (int s = 0; s < q; s++) {
            int indicator = (s == MSA(b, r)) ? 1 : 0;
            vGrad(s) = W(b) * (indicator - Ps(s));
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
    int r = 0;
    int q = 0;
    int N = 0;
    int B = 0;
    const Eigen::MatrixXi* MSA = nullptr;
    const Eigen::Matrix<PLMV, -1, 1>* W = nullptr;
    PLMV lambdaH = 0.0;
    PLMV lambdaJ = 0.0;
};

double pllTarget(const std::vector<double> &x, std::vector<double> &grad, void* f_data){
    auto* ctx = reinterpret_cast<NLoptContext*>(f_data);
    const int n = x.size();

    Eigen::Map<const Eigen::VectorXd> x_d(x.data(), n);
    auto [pll, gradients] = perSitePllGradient(
        x_d,
        ctx->r,
        ctx->q,
        ctx->N,
        ctx->B,
        *(ctx->MSA),
        *(ctx->W),
        ctx->lambdaH,
        ctx->lambdaJ
    );

    if (!grad.empty()) {
        if (grad.size() != static_cast<size_t>(n)) grad.resize(n);
        Eigen::Map<Eigen::VectorXd> g_d(grad.data(), n);
        g_d = gradients;
    }

    return pll;
}

Eigen::Matrix<PLMV, -1, 1> perSiteNLopt(
    unsigned nParams,
    int r,
    int q,
    int N,
    int B,
    const Eigen::MatrixXi& MSA,
    const Eigen::Matrix<PLMV, -1, 1>& W,
    PLMV lambdaH,
    PLMV lambdaJ,
    double eps_conv = 1e-4,
    int maxeval = 500) {
    
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
    int B = 0;
    int N = 0;
    int q = 0;
    double Beff = 0.0;
};

MSAData loadData(const std::string& dataDir) {
    MSAData data;
    
    // Read metadata
    std::string metaPath = dataDir + "/meta.bin";
    std::ifstream metaFile(metaPath, std::ios::binary);
    if (!metaFile) {
        throw std::runtime_error("Cannot open " + metaPath);
    }
    
    int meta[3];
    metaFile.read(reinterpret_cast<char*>(meta), 3 * sizeof(int));
    metaFile.read(reinterpret_cast<char*>(&data.Beff), sizeof(double));
    metaFile.close();
    
    data.B = meta[0];
    data.N = meta[1];
    data.q = meta[2];
    
    std::cout << "Loading data: B=" << data.B << ", N=" << data.N 
              << ", q=" << data.q << ", Beff=" << data.Beff << std::endl;
    
    // Read MSA (row-major int32)
    std::string msaPath = dataDir + "/msa.bin";
    std::ifstream msaFile(msaPath, std::ios::binary);
    if (!msaFile) {
        throw std::runtime_error("Cannot open " + msaPath);
    }
    
    // Eigen uses column-major by default, but we read row-major data
    // So we read into a temporary and transpose, or use RowMajor storage
    std::vector<int32_t> msaBuffer(data.B * data.N);
    msaFile.read(reinterpret_cast<char*>(msaBuffer.data()), data.B * data.N * sizeof(int32_t));
    msaFile.close();
    
    // Map as row-major and copy to column-major Eigen matrix
    Eigen::Map<Eigen::Matrix<int32_t, -1, -1, Eigen::RowMajor>> msaMap(msaBuffer.data(), data.B, data.N);
    data.MSA = msaMap.cast<int>();
    
    std::cout << "Loaded MSA: " << formatBytes(data.B * data.N * sizeof(int)) << std::endl;
    
    // Read weights
    std::string wPath = dataDir + "/weights.bin";
    std::ifstream wFile(wPath, std::ios::binary);
    if (!wFile) {
        throw std::runtime_error("Cannot open " + wPath);
    }
    
    data.W.resize(data.B);
    wFile.read(reinterpret_cast<char*>(data.W.data()), data.B * sizeof(double));
    wFile.close();
    
    // Normalize weights
    data.W /= data.Beff;
    
    std::cout << "Loaded weights: " << formatBytes(data.B * sizeof(double)) << std::endl;
    
    return data;
}

// ============================================================================
// Multi-threaded test with thread pool
// ============================================================================

void runParallelTest(const MSAData& data, int numThreads, double lambdaH, double lambdaJ) {
    int N = data.N;
    int numSites = N;  // Compute ALL sites
    unsigned nParams = data.q + (N - 1) * data.q * data.q;
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Running parallel test with " << numThreads << " concurrent threads" << std::endl;
    std::cout << "Total sites to compute: " << numSites << std::endl;
    std::cout << "nParams per site: " << nParams << std::endl;
    std::cout << "Expected memory per optimizer: ~" << formatBytes(nParams * 8 * 25) << std::endl;
    std::cout << "Expected total optimizer memory (" << numThreads << " concurrent): ~" 
              << formatBytes(static_cast<size_t>(nParams) * 8 * 25 * numThreads) << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    size_t memBefore = getCurrentRSS();
    std::cout << "Memory before thread creation: " << formatBytes(memBefore) << std::endl;
    
    std::atomic<int> completed{0};
    std::atomic<int> nextSite{0};
    std::atomic<size_t> peakMem{memBefore};
    std::mutex printMutex;
    std::vector<std::thread> threads;
    std::vector<Eigen::Matrix<PLMV, -1, 1>> results(numSites);
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Thread pool worker function
    auto worker = [&]() {
        while (true) {
            int site = nextSite.fetch_add(1);
            if (site >= numSites) {
                break;
            }
            
            try {
                results[site] = perSiteNLopt(
                    nParams, site, data.q, data.N, data.B,
                    data.MSA, data.W, lambdaH, lambdaJ,
                    1e-4, 500  // Full iterations
                );
                
                int done = ++completed;
                auto result = perSiteNLopt(
                    nParams, site, data.q, data.N, data.B,
                    data.MSA, data.W, lambdaH, lambdaJ,
                    1e-4, 500
                );
                
                // Update peak memory
                size_t currentMem = getCurrentRSS();
                size_t prevPeak = peakMem.load();
                while (currentMem > prevPeak && !peakMem.compare_exchange_weak(prevPeak, currentMem)) {}
                
                if (done % 50 == 0 || done == numSites) {
                    std::lock_guard<std::mutex> lock(printMutex);
                    auto now = std::chrono::high_resolution_clock::now();
                    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - startTime).count();
                    double rate = done / std::max(1.0, static_cast<double>(elapsed));
                    int remaining = numSites - done;
                    double eta = remaining / std::max(0.1, rate);
                    
                    std::cout << "Progress: " << done << "/" << numSites 
                              << " (" << (done * 100 / numSites) << "%)"
                              << " | RSS: " << formatBytes(currentMem)
                              << " | Peak: " << formatBytes(peakMem.load())
                              << " | Rate: " << std::fixed << std::setprecision(1) << rate << " sites/s"
                              << " | ETA: " << static_cast<int>(eta) << "s"
                              << std::endl;
                }
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lock(printMutex);
                std::cerr << "Error in site " << site << ": " << e.what() << std::endl;
                ++completed;
            }
        }
    };
    
    // Create thread pool
    for (int i = 0; i < numThreads; i++) {
        threads.emplace_back(worker);
    }
    
    // Join all threads
    for (auto& t : threads) {
        t.join();
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    
    size_t memAfter = getCurrentRSS();
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "RESULTS" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Completed: " << completed << " sites" << std::endl;
    std::cout << "Time: " << duration.count() / 1000.0 << " seconds" << std::endl;
    std::cout << "Average rate: " << (numSites * 1000.0 / duration.count()) << " sites/s" << std::endl;
    std::cout << "Memory before: " << formatBytes(memBefore) << std::endl;
    std::cout << "Peak memory: " << formatBytes(peakMem.load()) << std::endl;
    std::cout << "Memory after: " << formatBytes(memAfter) << std::endl;
    std::cout << "Memory increase: " << formatBytes(peakMem.load() - memBefore) << std::endl;
    std::cout << "========================================\n" << std::endl;
}

void runSingleThreadTest(const MSAData& data, double lambdaH, double lambdaJ) {
    unsigned nParams = data.q + (data.N - 1) * data.q * data.q;
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Single thread memory test (10 iterations)" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    size_t memBefore = getCurrentRSS();
    std::cout << "Memory before: " << formatBytes(memBefore) << std::endl;
    
    for (int i = 0; i < 10; i++) {
        int site = i % data.N;
        
        size_t memPre = getCurrentRSS();
        
        {
            auto result = perSiteNLopt(
                nParams, site, data.q, data.N, data.B,
                data.MSA, data.W, lambdaH, lambdaJ,
                1e-4, 500
            );
        }
        
        size_t memPost = getCurrentRSS();
        std::cout << "Iteration " << i << ": pre=" << formatBytes(memPre) 
                  << " post=" << formatBytes(memPost) 
                  << " delta=" << formatBytes(memPost - memPre) << std::endl;
    }
    
    size_t memAfter = getCurrentRSS();
    std::cout << "\nMemory after all iterations: " << formatBytes(memAfter) << std::endl;
    std::cout << "Total leak (if any): " << formatBytes(memAfter - memBefore) << std::endl;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <data_dir> [num_threads]" << std::endl;
        std::cerr << "  data_dir: directory containing meta.bin, msa.bin, weights.bin" << std::endl;
        std::cerr << "  num_threads: number of parallel threads (default: 256)" << std::endl;
        std::cerr << "\nFirst run export_test_data.py to generate the binary files from zarr." << std::endl;
        return 1;
    }
    
    std::string dataDir = argv[1];
    int numThreads = (argc >= 3) ? std::atoi(argv[2]) : 256;
    
    double lambdaH = 0.01;
    double lambdaJ = 0.01;
    
    std::cout << "========================================" << std::endl;
    std::cout << "C++ Memory Test for perSiteNLopt" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Data directory: " << dataDir << std::endl;
    std::cout << "Requested threads: " << numThreads << std::endl;
    std::cout << "lambdaH: " << lambdaH << ", lambdaJ: " << lambdaJ << std::endl;
    std::cout << "Initial memory: " << formatBytes(getCurrentRSS()) << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    try {
        MSAData data = loadData(dataDir);
        
        std::cout << "\nMemory after loading data: " << formatBytes(getCurrentRSS()) << std::endl;
        
        // runParallelTest(data, numThreads, lambdaH, lambdaJ);
        runSingleThreadTest(data, lambdaH, lambdaJ);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
