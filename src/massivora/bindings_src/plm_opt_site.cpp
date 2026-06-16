#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <cstdlib>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <nlopt.hpp>

using PLMV = double;


// ===========================================================================
// CPU implementation (NLopt L-BFGS, one process per site)
// ===========================================================================

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

        Eigen::Matrix<PLMV, -1, 1> Ps = (energies.array() - lnorm).exp();
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
    PLMV lambdaH = 0.01;
    PLMV lambdaJ = 0.01;
};

double pllTarget(const std::vector<double> &x, std::vector<double> &grad, void* f_data){
    auto* ctx = reinterpret_cast<NLoptContext*>(f_data);
    // Map NLopt's vector<double> to Eigen without copying
    Eigen::Map<const Eigen::VectorXd> x_d(x.data(), x.size());

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
        Eigen::Map<Eigen::VectorXd>(grad.data(), grad.size()) = gradients;
    }

    return pll;
}


Eigen::Matrix<PLMV, -1, 1> perSiteNLopt(
    int r,
    const Eigen::MatrixXi& MSA,
    const Eigen::Matrix<PLMV, -1, 1>& W,
    int N,
    int B,
    int q,
    PLMV lambdaH = 0.01,
    PLMV lambdaJ = 0.01,
    double eps_conv = 1e-4,
    int maxeval = 500) {

    unsigned nParams = (N - 1) * q * q + q;

    std::vector<double> x0(nParams, 0.0);
    nlopt::opt opt(nlopt::LD_LBFGS, nParams);
    NLoptContext ctx{r, q, N, B, &MSA, &W, lambdaH, lambdaJ};
    opt.set_min_objective(pllTarget, &ctx);
    opt.set_maxeval(maxeval);
    opt.set_ftol_abs(eps_conv);
    opt.set_xtol_abs(eps_conv);

    double opt_f;
    opt.optimize(x0, opt_f);

    Eigen::Map<const Eigen::Matrix<PLMV, -1, 1>> J_map(x0.data() + q, nParams - q);
    Eigen::Matrix<PLMV, -1, 1> J = J_map;
    return J;
}

void applyIsingGauge(Eigen::Matrix<PLMV, -1, 1>& J, int q) {
    const int site_size = q * q;

    // Iterate over each site
    for (int i = 0; i < J.rows(); i+=site_size) {
        Eigen::Map<Eigen::Matrix<PLMV, -1, -1>> Jij(J.data() + i, q, q);
        Eigen::Matrix<PLMV, -1, 1> row_mean = Jij.rowwise().mean();
        Eigen::Matrix<PLMV, -1, 1> col_mean = Jij.colwise().mean();
        PLMV total_mean = row_mean.mean();

        // Apply Ising gauge transformation
        for (int k = 0; k < q; ++k) {
            for (int l = 0; l < q; ++l) {
                Jij(k, l) = Jij(k, l) - row_mean(k) - col_mean(l) + total_mean;
            }
        }
    }
}


struct MSAData {
    Eigen::MatrixXi MSA;
    Eigen::Matrix<PLMV, -1, 1> W;
};

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

// Shared-memory layout is identical for the CPU and GPU pipelines:
//   MSA : (B, N) int32  (row-major)
//   W   : (B,)   float64
// The GPU path converts these into the int8 / fp32 / fp16 buffers the CUDA
// kernels expect after loading.
const MSAData loadData(const std::string& pairName, int B, int N) {
    size_t msa_size = (size_t)B * N * sizeof(int);
    size_t w_size = (size_t)B * sizeof(PLMV);
    ShmMapping mapping = openShm(pairName, msa_size + w_size);

    Eigen::Map<Eigen::Matrix<int, -1, -1, Eigen::RowMajor>> MSA(
            static_cast<int*>(mapping.ptr), B, N);

    Eigen::Matrix<PLMV, -1, 1> W(B);
    std::memcpy(W.data(), static_cast<char*>(mapping.ptr) + msa_size, w_size);
    return MSAData{MSA, W};
}


#ifdef ENABLE_CUDA
// ===========================================================================
// GPU implementation (CUDA, one process per pair, all sites on the device)
// ===========================================================================
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <map>
#include <thread>
#include "bindings.cuh"

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _err = (call);                                             \
        if (_err != cudaSuccess) {                                             \
            size_t _free = 0, _total = 0;                                      \
            cudaMemGetInfo(&_free, &_total);                                   \
            std::cerr << "CUDA error " << cudaGetErrorString(_err)             \
                      << " at " << __FILE__ << ":" << __LINE__                 \
                      << " (free " << (_free >> 20) << " MiB / total "         \
                      << (_total >> 20) << " MiB)" << std::endl;               \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

// Free device memory in bytes (0 if the query fails).
static size_t gpuFreeBytes() {
    size_t freeB = 0, totalB = 0;
    if (cudaMemGetInfo(&freeB, &totalB) != cudaSuccess) return 0;
    return freeB;
}

// Ising gauge on the host (float32). The result buffer stores each q*q coupling
// block in row-major order (row index a, col index b), matching the GPU
// `apply_ising_gauge` kernel convention.
void applyIsingGaugeF32(float* J, size_t nElems, int q) {
    const int site_size = q * q;
    for (size_t off = 0; off + (size_t)site_size <= nElems; off += site_size) {
        Eigen::Map<Eigen::Matrix<float, -1, -1, Eigen::RowMajor>> Jij(J + off, q, q);
        Eigen::Matrix<float, -1, 1> row_mean = Jij.rowwise().mean();
        Eigen::Matrix<float, -1, 1> col_mean = Jij.colwise().mean();
        float total_mean = row_mean.mean();

        for (int k = 0; k < q; ++k) {
            for (int l = 0; l < q; ++l) {
                Jij(k, l) = Jij(k, l) - row_mean(k) - col_mean(l) + total_mean;
            }
        }
    }
}

int runGpu(const std::string& pairName, int B, int N, int q,
           float lambdaH, float lambdaJ, float eps_conv, int maxeval,
           int n_streams) {
    MSAData data = loadData(pairName, B, N);
    const Eigen::MatrixXi& MSA = data.MSA;
    const Eigen::Matrix<PLMV, -1, 1>& W = data.W;

    const int q_pad = ((q + 7) / 8) * 8;
    const size_t n_params = (size_t)q_pad + (size_t)N * q_pad * q_pad;

    // ---- Convert host inputs into the dtypes the CUDA kernels expect ----
    std::vector<signed char> msaHost((size_t)B * N);
    std::vector<float>        wHost((size_t)B);
    std::vector<__half>       msaPadHost((size_t)B * N * q_pad, __float2half(0.0f));

    for (int b = 0; b < B; ++b) {
        wHost[b] = static_cast<float>(W(b));
        for (int i = 0; i < N; ++i) {
            const int s = MSA(b, i);
            msaHost[(size_t)b * N + i] = static_cast<signed char>(s);
            // One-hot encode into (B, N, q_pad) fp16.
            msaPadHost[((size_t)b * N + i) * q_pad + s] = __float2half(1.0f);
        }
    }

    // ---- Upload to device ----
    signed char* d_MSA      = nullptr;
    float*       d_W        = nullptr;
    __half*      d_MSA_pad  = nullptr;
    // TODO: Now the x0 for all sites are on GPU, but if this is going to occupy 
    // 60%+ of GPU memory on current device, only put one site on GPU at a time 
    // and keep the rest on CPU.
    __half*      d_x0       = nullptr;  // (N, n_params) fp16, one row per site

    CUDA_CHECK(cudaMalloc(&d_MSA,     msaHost.size()    * sizeof(signed char)));
    CUDA_CHECK(cudaMalloc(&d_W,       wHost.size()      * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_MSA_pad, msaPadHost.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_x0,      (size_t)N * n_params * sizeof(__half)));

    CUDA_CHECK(cudaMemcpy(d_MSA,     msaHost.data(),
                          msaHost.size() * sizeof(signed char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W,       wHost.data(),
                          wHost.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_MSA_pad, msaPadHost.data(),
                          msaPadHost.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_x0, 0, (size_t)N * n_params * sizeof(__half)));

    // Host one-hot is no longer needed; free it before the GPU optimization.
    std::vector<__half>().swap(msaPadHost);

    // ---- Cap n_streams to what actually fits in free device memory ----
    // Each concurrent cudaOptimizeSite call allocates its own scratch buffers,
    // dominated by a full (B, N*q_pad) fp16 copy of the one-hot MSA. Running
    // more streams than memory allows is the main cause of OOM here.
    if (n_streams < 1) n_streams = 1;
    {
        const size_t per_stream =
            (size_t)B * N * q_pad * sizeof(__half)   // MSA_pad_flat (dominant)
            + n_params * sizeof(float)               // grad_pad
            + (size_t)B * q_pad * sizeof(__half) * 2 // vgrad_pad + energies_fp16
            + (size_t)B * q_pad * sizeof(float)      // energies_fp32
            + 4 * sizeof(float);                     // pll_out
        // Reserve a margin for cuBLAS workspaces and allocator fragmentation.
        const size_t free_now = gpuFreeBytes();
        const size_t margin   = (size_t)256 << 20;   // 256 MiB
        size_t budget = (free_now > margin) ? (free_now - margin) : 0;
        int fit = (per_stream > 0) ? (int)(budget / per_stream) : n_streams;
        if (fit < 1) fit = 1;
        if (fit < n_streams) {
            std::cerr << "plm_opt_site: capping n_streams " << n_streams
                      << " -> " << fit << " (free " << (free_now >> 20)
                      << " MiB, ~" << (per_stream >> 20)
                      << " MiB per stream)" << std::endl;
            n_streams = fit;
        }
    }
    if (n_streams > N) n_streams = N;

    // ---- Optimize every site, distributing them across CUDA streams ----

    auto worker = [&](int tid) {
        cudaStream_t stream;
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        for (int r = tid; r < N; r += n_streams) {
            cudaOptimizeSite(
                d_MSA_pad, d_MSA, d_W,
                r, B, N, q, q_pad,
                lambdaH, lambdaJ, eps_conv, maxeval,
                d_x0 + (size_t)r * n_params,
                std::map<std::string, float>{},
                /*ext_handle=*/nullptr,
                stream);
        }
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    };

    std::vector<std::thread> threads;
    for (int t = 0; t < n_streams; ++t) {
        threads.emplace_back(worker, t);
    }
    for (auto& th : threads) th.join();
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- Copy optimized parameters back to host ----
    std::vector<__half> x0Host((size_t)N * n_params);
    CUDA_CHECK(cudaMemcpy(x0Host.data(), d_x0,
                          (size_t)N * n_params * sizeof(__half), cudaMemcpyDeviceToHost));

    cudaFree(d_MSA);
    cudaFree(d_W);
    cudaFree(d_MSA_pad);
    cudaFree(d_x0);

    // ---- Extract the (q x q) coupling blocks, cropping away q_pad padding ----
    // Output layout: row r holds site r's couplings against every site i as a
    // contiguous (N, q, q) block -> a flat (N, N*q*q) float32 buffer.
    const size_t outElems = (size_t)N * N * q * q;
    std::vector<float> outBuf(outElems);
    for (int r = 0; r < N; ++r) {
        const __half* xr = x0Host.data() + (size_t)r * n_params + q_pad;  // skip h
        float* out_r = outBuf.data() + (size_t)r * N * q * q;
        for (int i = 0; i < N; ++i) {
            const __half* blk = xr + (size_t)i * q_pad * q_pad;
            float* out_blk = out_r + (size_t)i * q * q;
            for (int a = 0; a < q; ++a) {
                for (int b = 0; b < q; ++b) {
                    out_blk[a * q + b] = __half2float(blk[a * q_pad + b]);
                }
            }
        }
    }

    // ---- Ising gauge on the host (float32), then publish to shared memory ----
    applyIsingGaugeF32(outBuf.data(), outElems, q);

    ShmMapping shmJ = openShm(pairName + "_J", outElems * sizeof(float));
    std::memcpy(shmJ.ptr, outBuf.data(), outElems * sizeof(float));

    return 0;
}
#endif // ENABLE_CUDA


// CPU: optimize a single site and write its column into the shared `_J` matrix.
int runCpu(const std::string& pairName, int r, int B, int N, int q,
           double lambdaH, double lambdaJ, double eps_conv, int maxeval) {
    MSAData data = loadData(pairName, B, N);
    Eigen::Matrix<PLMV, -1, 1> Jr = perSiteNLopt(r, data.MSA, data.W, N, B, q, lambdaH, lambdaJ, eps_conv, maxeval);
    applyIsingGauge(Jr, q);

    const int nParamsJSite = q * q;
    ShmMapping shmJ = openShm(pairName + "_J", (size_t)N * N * q * q * sizeof(PLMV));
    Eigen::Map<Eigen::Matrix<PLMV, -1, -1, Eigen::RowMajor>> J(
        reinterpret_cast<PLMV*>(static_cast<char*>(shmJ.ptr)), q * q * N, N);
    for (int i = 0; i < r*nParamsJSite; i+=nParamsJSite) {
        J.col(r).segment(i, nParamsJSite) = Jr.segment(i, nParamsJSite);
    }
    J.col(r).segment(r*nParamsJSite, nParamsJSite) = Eigen::Matrix<PLMV, -1, 1>::Zero(nParamsJSite);
    for (int i = (r+1)*nParamsJSite; i < N*nParamsJSite; i+=nParamsJSite) {
        J.col(r).segment(i, nParamsJSite) = Jr.segment(i - nParamsJSite, nParamsJSite);
    }
    return 0;
}


int main(int argc, char* argv[]) {
    // The device is selected at runtime: pass --use-gpu to run the CUDA path
    // (only available when built with -DENABLE_CUDA=ON), otherwise the CPU path.
    bool use_gpu = false;
    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--use-gpu") use_gpu = true;
        else pos.push_back(std::move(a));
    }

    if (use_gpu) {
#ifdef ENABLE_CUDA
        // GPU: optimize all sites of a pair in a single process.
        // args: pairName B N q lambdaH lambdaJ eps_conv maxeval n_streams
        if (pos.size() < 9) {
            std::cerr << "Usage: " << argv[0]
                      << " --use-gpu <pairName> <B> <N> <q> <lambdaH> <lambdaJ>"
                         " <eps_conv> <maxeval> <n_streams>" << std::endl;
            return 1;
        }
        std::string pairName = pos[0];
        int   B         = std::stoi(pos[1]);
        int   N         = std::stoi(pos[2]);
        int   q         = std::stoi(pos[3]);
        float lambdaH   = std::stof(pos[4]);
        float lambdaJ   = std::stof(pos[5]);
        float eps_conv  = std::stof(pos[6]);
        int   maxeval   = std::stoi(pos[7]);
        int   n_streams = std::stoi(pos[8]);
        return runGpu(pairName, B, N, q, lambdaH, lambdaJ, eps_conv, maxeval, n_streams);
#else
        std::cerr << argv[0] << ": built without CUDA support; rebuild with "
                     "-DENABLE_CUDA=ON to use --use-gpu." << std::endl;
        return 1;
#endif
    }

    // CPU: optimize a single site.
    // args: pairName r B N q lambdaH lambdaJ eps_conv maxeval
    if (pos.size() < 9) {
        std::cerr << "Usage: " << argv[0]
                  << " <pairName> <r> <B> <N> <q> <lambdaH> <lambdaJ>"
                     " <eps_conv> <maxeval>" << std::endl;
        return 1;
    }
    std::string pairName = pos[0];
    int    r        = std::stoi(pos[1]);
    int    B        = std::stoi(pos[2]);
    int    N        = std::stoi(pos[3]);
    int    q        = std::stoi(pos[4]);
    double lambdaH  = std::stod(pos[5]);
    double lambdaJ  = std::stod(pos[6]);
    double eps_conv = std::stod(pos[7]);
    int    maxeval  = std::stoi(pos[8]);
    return runCpu(pairName, r, B, N, q, lambdaH, lambdaJ, eps_conv, maxeval);
}
