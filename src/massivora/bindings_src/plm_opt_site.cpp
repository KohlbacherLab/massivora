#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <utility>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <nlopt.hpp>

using PLMV = double;

// The MSA in Zarr is int8
using MSAScalar = signed char;
using MSAMap = Eigen::Map<const Eigen::Matrix<MSAScalar, -1, -1, Eigen::RowMajor>>;
using WMap   = Eigen::Map<const Eigen::Matrix<PLMV, -1, 1>>;


// ===========================================================================
// CPU implementation (NLopt L-BFGS, one process per site)
// ===========================================================================

Eigen::Matrix<PLMV, -1, 1> getEnergies(
    const Eigen::Matrix<PLMV, -1, 1>& x,
    int r,
    int q,
    int N,
    int b,
    const MSAMap& MSA) {

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
    const MSAMap& MSA,
    const WMap& W,
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
    const MSAMap* MSA = nullptr;
    const WMap* W = nullptr;
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
    const MSAMap& MSA,
    const WMap& W,
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

// Only used in CPU calculations, not the GPU path.
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


struct ShmMapping {
    void* ptr = nullptr;
    size_t size = 0;
    int fd = -1;

    ShmMapping() = default;
    ShmMapping(const ShmMapping&) = delete;
    ShmMapping& operator=(const ShmMapping&) = delete;

    // Movable so a mapping can be owned by the MSAData it backs.
    ShmMapping(ShmMapping&& o) noexcept : ptr(o.ptr), size(o.size), fd(o.fd) {
        o.ptr = nullptr; o.size = 0; o.fd = -1;
    }
    ShmMapping& operator=(ShmMapping&& o) noexcept {
        if (this != &o) {
            reset();
            ptr = o.ptr; size = o.size; fd = o.fd;
            o.ptr = nullptr; o.size = 0; o.fd = -1;
        }
        return *this;
    }

    void reset() {
        if (ptr && ptr != MAP_FAILED) munmap(ptr, size);
        if (fd >= 0) close(fd);
        ptr = nullptr; size = 0; fd = -1;
    }

    ~ShmMapping() { reset(); }
};

// MSA and W are views into the mapping this struct keeps alive -- nothing is
// copied out of shared memory.
struct MSAData {
    ShmMapping mapping;
    MSAMap MSA;
    WMap   W;

    MSAData(ShmMapping&& m, const MSAScalar* msa_ptr, const PLMV* w_ptr, int B, int N)
        : mapping(std::move(m)), MSA(msa_ptr, B, N), W(w_ptr, B) {}
};

static const std::string SHM_PREFIX = "Massivora_";

ShmMapping openShm(const std::string& name, size_t expected_size = 0) {
    ShmMapping m;

    std::string shm_path = "/" + SHM_PREFIX + name;
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

// Shared-memory layout, identical for the CPU and GPU pipelines and written by
// the executors in couple.py:
//   MSA : (B, N) int8    (row-major, at offset 0)
//   W   : (B,)   float64 (at msaBytes rounded up to 8, so the doubles are aligned)
// Both are mapped in place: the CPU path indexes them directly and the GPU path
// uploads the int8 bytes as-is, so there is no dtype conversion on the host.
static inline size_t weightsOffset(int B, int N) {
    const size_t msa_size = (size_t)B * N * sizeof(MSAScalar);
    return (msa_size + alignof(PLMV) - 1) & ~(size_t)(alignof(PLMV) - 1);
}

MSAData loadData(const std::string& pairName, int B, int N) {
    const size_t w_off  = weightsOffset(B, N);
    const size_t w_size = (size_t)B * sizeof(PLMV);
    ShmMapping mapping = openShm(pairName, w_off + w_size);

    char* base = static_cast<char*>(mapping.ptr);
    const MSAScalar* msa_ptr = reinterpret_cast<const MSAScalar*>(base);
    const PLMV*      w_ptr   = reinterpret_cast<const PLMV*>(base + w_off);

    return MSAData(std::move(mapping), msa_ptr, w_ptr, B, N);
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

int runGpu(const std::string& pairName, int B, int N, int q,
           float lambdaH, float lambdaJ, float eps_conv, int maxeval,
           int n_streams) {
    MSAData data = loadData(pairName, B, N);
    // Points straight into the shared-memory mapping; already the dtype the
    // kernels want, so it is uploaded byte-for-byte.
    const MSAScalar* msaHost = data.MSA.data();
    const WMap& W = data.W;

    const int q_pad = ((q + 7) / 8) * 8;
    const size_t n_params = (size_t)q_pad + (size_t)N * q_pad * q_pad;
    // Must track the L-BFGS m_corr default in cudaOptimizeSite; only used to
    // size the stream budget below.
    const int m_corr_hint = 8;

    // ---- The only host-side conversion left is the (B,) weight vector ----
    // The MSA goes up as the int8 it already is, and the (B, N, q_pad) fp16
    // one-hot is expanded on the device (see cudaBuildMsaOneHot below), so it
    // never costs B*N*q_pad*2 bytes of host RAM nor the same again over PCIe.
    std::vector<float> wHost((size_t)B);
    for (int b = 0; b < B; ++b) wHost[b] = static_cast<float>(W(b));

    // ---- Upload to device ----
    signed char* d_MSA      = nullptr;
    float*       d_W        = nullptr;
    __half*      d_MSA_pad  = nullptr;
    // TODO: Now the x0 for all sites are on GPU, but if this is going to occupy 
    // 60%+ of GPU memory on current device, only put one site on GPU at a time 
    // and keep the rest on CPU.
    __half*      d_x0       = nullptr;  // (N, n_params) fp16, one row per site

    const size_t msa_elems     = (size_t)B * N;
    const size_t msa_pad_elems = msa_elems * (size_t)q_pad;

    CUDA_CHECK(cudaMalloc(&d_MSA,     msa_elems    * sizeof(signed char)));
    CUDA_CHECK(cudaMalloc(&d_W,       wHost.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_MSA_pad, msa_pad_elems * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_x0,      (size_t)N * n_params * sizeof(__half)));

    CUDA_CHECK(cudaMemcpy(d_MSA, msaHost, msa_elems * sizeof(signed char),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W,   wHost.data(), wHost.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_x0, 0, (size_t)N * n_params * sizeof(__half)));

    // Expand the one-hot on the device from the int8 MSA. It runs on the legacy
    // default stream, which the per-site non-blocking streams do NOT sync
    // against, so the barrier below is what makes it visible to them.
    cudaBuildMsaOneHot(d_MSA, d_MSA_pad, B, N, q_pad, /*stream=*/0);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- Cap n_streams to what actually fits in free device memory ----
    // Each concurrent cudaOptimizeSite call allocates its own scratch buffers.
    // The one-hot MSA is no longer among them -- it is shared read-only across
    // sites -- so per-stream cost is now dominated by the L-BFGS state, and far
    // more streams fit than before.
    if (n_streams < 1) n_streams = 1;
    {
        const size_t per_stream =
            n_params * sizeof(float)                 // grad_pad
            + (size_t)m_corr_hint * 2 * n_params * sizeof(float)  // s_buf + y_buf
            + 4 * n_params * sizeof(float)           // x_fp32, x_trial, g_cur, dir
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
        // One cuBLAS handle per worker, reused across every site this thread
        // owns. Passing nullptr made cudaOptimizeSite create and destroy a
        // handle per site, and cublasDestroy() implicitly synchronises the
        // whole device -- a barrier across every concurrent stream, once per
        // site. Hoisting it is worth ~1.25x on the whole optimize phase.
        cublasHandle_t handle;
        cublasCreate(&handle);
        for (int r = tid; r < N; r += n_streams) {
            cudaOptimizeSite(
                d_MSA_pad, d_MSA, d_W,
                r, B, N, q, q_pad,
                lambdaH, lambdaJ, eps_conv, maxeval,
                d_x0 + (size_t)r * n_params,
                // n_streams lets the optimizer pick between the cooperative
                // and fused two-loop: the cooperative kernel is near-full-grid
                // and serialises concurrent streams past ~8.
                std::map<std::string, float>{{"n_streams", (float)n_streams}},
                /*ext_handle=*/handle,
                stream);
        }
        cudaStreamSynchronize(stream);
        cublasDestroy(handle);
        cudaStreamDestroy(stream);
    };

    std::vector<std::thread> threads;
    for (int t = 0; t < n_streams; ++t) {
        threads.emplace_back(worker, t);
    }
    for (auto& th : threads) th.join();
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- Copy optimized parameters back to host (pinned: ~25% faster D2H) ----
    __half* x0Host = nullptr;
    CUDA_CHECK(cudaHostAlloc(&x0Host, (size_t)N * n_params * sizeof(__half),
                             cudaHostAllocDefault));
    CUDA_CHECK(cudaMemcpy(x0Host, d_x0,
                          (size_t)N * n_params * sizeof(__half), cudaMemcpyDeviceToHost));

    cudaFree(d_MSA);
    cudaFree(d_W);
    cudaFree(d_MSA_pad);
    cudaFree(d_x0);

    // ---- Extract the (q x q) coupling blocks, cropping away q_pad padding ----
    // Output layout: row r holds site r's couplings against every site i as a
    // contiguous (N, q, q) block -> a flat (N, N*q*q) float32 buffer.
    // ---- Publish the RAW (un-gauged) J to shared memory. The Ising gauge is
    // applied downstream on the CPU (cpp_bindings.applyIsingGauge in the GPU
    // executor's collect_results), so it no longer occupies this GPU process.
    //
    // The destination is mapped first and the blocks are cropped straight into
    // it: the intermediate buffer only existed to be memcpy'd here. The loop is
    // a pure gather with no cross-iteration state, so it parallelises directly.
    const size_t outElems = (size_t)N * N * q * q;
    ShmMapping shmJ = openShm(pairName + "_J", outElems * sizeof(float));
    float* outBuf = static_cast<float*>(shmJ.ptr);

    #pragma omp parallel for schedule(static)
    for (int r = 0; r < N; ++r) {
        const __half* xr = x0Host + (size_t)r * n_params + q_pad;  // skip h
        float* out_r = outBuf + (size_t)r * N * q * q;
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
    cudaFreeHost(x0Host);

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
