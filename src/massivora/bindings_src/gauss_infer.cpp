#define EIGEN_DONT_PARALLELIZE

#include <Eigen/Dense>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "gauss_dca.hpp"

using GV = double;


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

// Infer the couplings for one pair and publish them to the `_J` segment.
int runInfer(const std::string& pairName, int M, int N, int q,
             double pseudocount, int n_threads) {
    const int nthreads = gaussdca::resolve_threads(n_threads);
    const int Q = q - 1;
    const size_t NQ = static_cast<size_t>(N) * Q;

    // Input segment: the alignment as int8 in residue-major (N, M) order
    const size_t msa_size = static_cast<size_t>(N) * M * sizeof(int8_t);
    const size_t w_size = static_cast<size_t>(M) * sizeof(GV);
    ShmMapping in = openShm(pairName, msa_size + w_size);

    const int8_t* Z = static_cast<const int8_t*>(in.ptr);
    const char* w_raw = static_cast<const char*>(in.ptr) + msa_size;

    // The weights follow the MSA with no padding, so they are only guaranteed
    // 8-byte aligned when M*N is even. Copy in the rare case it is not.
    std::vector<GV> w_copy;
    const GV* W;
    if (reinterpret_cast<uintptr_t>(w_raw) % alignof(GV) == 0) {
        W = reinterpret_cast<const GV*>(w_raw);
    } else {
        w_copy.resize(M);
        std::memcpy(w_copy.data(), w_raw, w_size);
        W = w_copy.data();
    }

    GV Meff = 0.0;
    for (int k = 0; k < M; ++k) Meff += W[k];

    // Output segment: the (N, N) APC-corrected score, float32, row-major.
    (void)NQ;
    ShmMapping shmScore = openShm(pairName + "_J",
                                  static_cast<size_t>(N) * N * sizeof(float));
    float* score = static_cast<float*>(shmScore.ptr);

    gaussdca::compute_score_into(Z, N, M, W, Meff, q, pseudocount,
                                 nthreads, score);
    return 0;
}


int main(int argc, char* argv[]) {
    // args: pairName M N q pseudocount n_threads
    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) pos.push_back(argv[i]);

    // Report the build configuration. Nothing depends on this at run time --
    // the precision and the BLAS interface are both fixed -- but it is the
    // quickest way to confirm an installed binary is the one you think it is.
    if (pos.size() == 1 && pos[0] == "--info") {
        std::cout << "cov_bytes=" << sizeof(gaussdca::CovScalar) << "\n";
        std::cout << "blas_int_bytes=" << sizeof(gaussdca::blas_int) << "\n";
        std::cout << "ilp64=" << (sizeof(gaussdca::blas_int) == 8 ? 1 : 0) << "\n";
        return 0;
    }

    if (pos.size() < 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <pairName> <M> <N> <q> <pseudocount> <n_threads>"
                  << std::endl;
        return 1;
    }

    std::string pairName    = pos[0];
    int         M           = std::stoi(pos[1]);   // number of sequences
    int         N           = std::stoi(pos[2]);   // number of residues
    int         q           = std::stoi(pos[3]);   // alphabet size (21)
    double      pseudocount = std::stod(pos[4]);
    int         n_threads   = std::stoi(pos[5]);

    try {
        return runInfer(pairName, M, N, q, pseudocount, n_threads);
    } catch (const std::exception& e) {
        std::cerr << "gauss_infer: " << e.what() << std::endl;
        return 1;
    }
}
