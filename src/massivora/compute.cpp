#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <omp.h>

namespace py = pybind11;
using PLMV = double;


Eigen::VectorXf getEnergies(
    const Eigen::VectorXf& x,
    int r,
    int q, 
    int N,
    int b,
    const Eigen::MatrixXi& MSA) {

    Eigen::VectorXf energies = Eigen::VectorXf::Zero(q);

    Eigen::Map<const Eigen::VectorXf> h(x.data(), q);
    Eigen::TensorMap<const Eigen::Tensor<float, 3>> Jr(x.data() + q, q, q, N-1);

    // #pragma omp parallel for
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

PLMV mergeEnergyTerm(const Eigen::VectorXf& energies) {
    PLMV max = energies.maxCoeff();
    return max + std::log((energies.array() - max).exp().sum());
}

PLMV l2Regularization(
    const Eigen::VectorXf& vec, 
    int q,
    PLMV lambdaH,
    PLMV lambdaJ) {

    PLMV reg_h = vec.head(q).array().square().sum() * lambdaH;
    PLMV reg_J = vec.tail(vec.size() - q).array().square().sum() * 0.5 * lambdaJ;

    return reg_h + reg_J;
}

std::vector<int> buildIndexMap(int q, int N) {
    std::vector<int> index_map(N-1);
    for (int i = 0; i < N-1; i++) {
        index_map[i] = q + i * q * q;  // Base offset for J[i,:,:]
    }

    return index_map;
}

std::tuple<PLMV, Eigen::VectorXf> perSitePllGradient(
    const Eigen::VectorXf& x, 
    int r,
    int q,
    int N,
    int B,
    const Eigen::MatrixXi& MSA,
    const Eigen::VectorXf& W,
    PLMV lambdaH,
    PLMV lambdaJ,
    int num_threads = 0) {

    py::gil_scoped_release release;

    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }

    PLMV pll = 0.0;
    Eigen::VectorXf gradients = Eigen::VectorXf::Zero(x.size());
    Eigen::TensorMap<Eigen::Tensor<float, 3>> Tgradients(gradients.data() + q, q, q, N - 1);

    #pragma omp parallel
    {
        PLMV local_pll = 0.0;
        Eigen::VectorXf local_gradients = Eigen::VectorXf::Zero(x.size());
        Eigen::TensorMap<Eigen::Tensor<float, 3>> local_Tgradients(
            local_gradients.data() + q, q, q, N - 1);

        #pragma omp for
        for (int b = 0; b < B; b++) {
            Eigen::VectorXf energies = getEnergies(x, r, q, N, b, MSA);
            PLMV lnorm = mergeEnergyTerm(energies);
            local_pll -= W(b) * (energies(MSA(b, r)) - lnorm);
            // pll -= W(b) * (energies(MSA(b, r)) - lnorm);

            Eigen::VectorXf Ps = energies.array() - lnorm;
            Ps = Ps.array().exp();

            Eigen::VectorXf vGrad = Eigen::VectorXf::Zero(q);
            for (int s = 0; s < q; s++) {
                int indicator = (s == MSA(b, r)) ? 1 : 0;
                vGrad(s) = W(b) * (indicator - Ps(s));
            }

            local_gradients.head(q) -= vGrad;
            // gradients.head(q) -= vGrad;

            for (int i = 0; i < r; i++) {
                int s_ib = MSA(b, i);
                for (int s = 0; s < q; s++) {
                    local_Tgradients(s, s_ib, i) -= vGrad(s);
                    // Tgradients(s, s_ib, i) -= vGrad(s);
                }
            }
            for (int i = r+1; i < N; i++) {
                int s_ib = MSA(b, i);
                for (int s = 0; s < q; s++) {
                    local_Tgradients(s, s_ib, i-1) -= vGrad(s);
                    // Tgradients(s, s_ib, i-1) -= vGrad(s);
                }
            }
        }

        #pragma omp critical
        {
            pll += local_pll;
            gradients += local_gradients;
        }
    }

    gradients.head(q) += 2.0f * lambdaH * x.head(q);
    gradients.tail(x.size() - q) += lambdaJ * x.tail(x.size() - q);

    pll += l2Regularization(x, q, lambdaH, lambdaJ);
    return std::make_tuple(pll, gradients);
}

std::tuple<PLMV, Eigen::VectorXf> perSitePllGradientNew(
    const Eigen::VectorXf& x, 
    int r,
    int q,
    int N,
    int B,
    const Eigen::MatrixXi& MSA,
    const Eigen::VectorXf& W,
    PLMV lambdaH,
    PLMV lambdaJ,
    const std::vector<int>& index_map,
    int num_threads = 0) {

    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }

    PLMV pll = 0.0;
    Eigen::VectorXf gradients = Eigen::VectorXf::Zero(x.size());

    #pragma omp parallel
    {
        PLMV local_pll = 0.0;
        Eigen::VectorXf local_gradients = Eigen::VectorXf::Zero(x.size());

        #pragma omp for
        for (int b = 0; b < B; b++) {
            Eigen::VectorXf energies = getEnergies(x, r, q, N, b, MSA);
            PLMV lnorm = mergeEnergyTerm(energies);
            local_pll -= W(b) * (energies(MSA(b, r)) - lnorm);

            Eigen::VectorXf Ps = energies.array() - lnorm;
            Ps = Ps.array().exp();

            Eigen::VectorXf vGrad = Eigen::VectorXf::Zero(q);
            for (int s = 0; s < q; s++) {
                int indicator = (s == MSA(b, r)) ? 1 : 0;
                vGrad(s) = W(b) * (indicator - Ps(s));
            }

            local_gradients.head(q) -= vGrad;

            // Use precomputed index map for J gradient updates
            for (int i = 0; i < r; i++) {
                int s_ib = MSA(b, i);
                int base_idx = index_map[i] + s_ib * q;
                for (int s = 0; s < q; s++) {
                    local_gradients(base_idx + s) -= vGrad(s);
                }
            }
            for (int i = r+1; i < N; i++) {
                int s_ib = MSA(b, i);
                int base_idx = index_map[i-1] + s_ib * q;
                for (int s = 0; s < q; s++) {
                    local_gradients(base_idx + s) -= vGrad(s);
                }
            }
        }

        #pragma omp critical
        {
            pll += local_pll;
            gradients += local_gradients;
        }
    }

    gradients.head(q) += 2.0f * lambdaH * x.head(q);
    gradients.tail(x.size() - q) += lambdaJ * x.tail(x.size() - q);

    pll += l2Regularization(x, q, lambdaH, lambdaJ);
    return std::make_tuple(pll, gradients);
}

Eigen::MatrixXf& applyIsingGauge(Eigen::MatrixXf& J, int q) {
    // Iterate over each site
    for (int i = 0; i < J.rows(); i=i+q*q) {
        Eigen::Map<Eigen::MatrixXf> Jij(J.data() + i, q, q);
        Eigen::VectorXf row_mean = Jij.rowwise().mean();
        Eigen::VectorXf col_mean = Jij.colwise().mean();
        double total_mean = row_mean.mean();

        // Apply Ising gauge transformation
        for (int k = 0; k < q; ++k) {
            for (int l = 0; l < q; ++l) {
                Jij(k, l) = Jij(k, l) - row_mean(k) - col_mean(l) + total_mean;
            }
        }
    }
    return J;
}

py::array_t<char[1]> getAlignmentInNumpy(const std::string& alignment, size_t seq_length, size_t align_length) {
    py::array_t<char[1]> result({align_length, seq_length});
    auto buf = result.request();
    char* ptr = static_cast<char*>(buf.ptr);

    auto worker = [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            for (size_t j = 0; j < seq_length; ++j) {
                ptr[i * seq_length + j] = alignment[i * seq_length + j];
            }
        }
    };

    int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    size_t chunk_size = align_length / num_threads + 1;

    for (int i = 0; i < num_threads; ++i) {
        size_t start = i * chunk_size;
        size_t end = std::min(start + chunk_size, align_length);
        if (start < align_length) {
            threads.emplace_back(worker, start, end);
        }
    }

    for (auto& t : threads) t.join();

    return result;
}

PYBIND11_MODULE(compute, m) {
    m.def("getAlignmentInNumpy", &getAlignmentInNumpy);
    m.def("buildIndexMap", &buildIndexMap,
        py::arg("q"),
        py::arg("N"));
    m.def("perSitePllGradient", &perSitePllGradient,
        py::arg("x"),
        py::arg("r"),
        py::arg("q"),
        py::arg("N"),
        py::arg("B"),
        py::arg("MSA"),
        py::arg("W"),
        py::arg("lambdaH"),
        py::arg("lambdaJ"),
        py::arg("num_threads") = 0);
    m.def("perSitePllGradientNew", &perSitePllGradientNew,
        py::arg("x"),
        py::arg("r"),
        py::arg("q"),
        py::arg("N"),
        py::arg("B"),
        py::arg("MSA"),
        py::arg("W"),
        py::arg("lambdaH"),
        py::arg("lambdaJ"),
        py::arg("index_map"),
        py::arg("num_threads") = 0);
    m.def("applyIsingGauge", &applyIsingGauge,
        py::arg("J"),
        py::arg("q"));
}

