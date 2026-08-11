#define EIGEN_DONT_PARALLELIZE

#include <Eigen/Dense>

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

struct MSAData {
    Eigen::MatrixXi MSA;              // (M, N)
    Eigen::Matrix<GV, -1, 1> W;      // (M,)
};

// Map the input segment: MSA (M, N) int32 row-major, then W (M,) float64.
const MSAData loadData(const std::string& pairName, int M, int N) {
    size_t msa_size = (size_t)M * N * sizeof(int);
    size_t w_size = (size_t)M * sizeof(GV);
    ShmMapping mapping = openShm(pairName, msa_size + w_size);

    Eigen::Map<Eigen::Matrix<int, -1, -1, Eigen::RowMajor>> MSA(
            static_cast<int*>(mapping.ptr), M, N);

    Eigen::Matrix<GV, -1, 1> W(M);
    std::memcpy(W.data(), static_cast<char*>(mapping.ptr) + msa_size, w_size);
    return MSAData{MSA, W};
}


// Infer the couplings for one pair and publish them to the `_J` segment.
int runInfer(const std::string& pairName, int M, int N, int q,
             double pseudocount, int n_threads) {
    MSAData data = loadData(pairName, M, N);
    const Eigen::MatrixXi& MSA = data.MSA;
    const Eigen::Matrix<GV, -1, 1>& W = data.W;

    // Transpose into the residue-major int8 buffer Z[i*M + k] the core expects.
    std::vector<int8_t> Z((size_t)N * M);
    for (int k = 0; k < M; ++k) {
        for (int i = 0; i < N; ++i) {
            Z[(size_t)i * M + k] = static_cast<int8_t>(MSA(k, i));
        }
    }

    const double Meff = W.sum();

    Eigen::MatrixXd J = gaussdca::compute_couplings(
        Z.data(), N, M, W.data(), Meff, q, pseudocount, n_threads);

    // Publish J (N*Q, N*Q) float64, row-major. J is symmetric, so the row-major
    // buffer matches what the Python side reshapes to (N, Q, N, Q).
    const int Q = q - 1;
    const size_t NQ = (size_t)N * Q;
    ShmMapping shmJ = openShm(pairName + "_J", NQ * NQ * sizeof(double));
    Eigen::Map<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>> Jout(
        static_cast<double*>(shmJ.ptr), NQ, NQ);
    Jout = J;
    return 0;
}


int main(int argc, char* argv[]) {
    // args: pairName M N q pseudocount n_threads
    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) pos.push_back(argv[i]);

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
