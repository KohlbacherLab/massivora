#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <iostream>
#include <map>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace py = pybind11;
using PLMV = double;


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

#ifdef ENABLE_CUDA
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include "bindings.cuh"

void cudaFillPllGradientsWrapper(
    intptr_t MSA_pad_flat,  // (B, N*q_pad)             fp16
    intptr_t x_pad,         // (q_pad + N*q_pad*q_pad,) fp16 = x0_pad[r]
    intptr_t MSA,           // (B, N)                   int8
    intptr_t W,             // (B,)                     fp32
    int r, int B, int N, int q, int q_pad,
    float lambdaH, float lambdaJ,
    intptr_t pll_out,                   // (4,)                     fp32
    intptr_t grad_pad,                  // (q_pad + N*q_pad*q_pad,) fp32
    intptr_t vgrad_pad,                 // (B, q_pad)               fp16 scratch
    intptr_t energies_fp32,             // (B, q_pad)               fp32 scratch
    intptr_t ext_handle,
    intptr_t stream
) {
    cudaFillPllGradients(
        reinterpret_cast<const __half*>     (MSA_pad_flat),
        reinterpret_cast<const __half*>     (x_pad),
        reinterpret_cast<const signed char*>(MSA),
        reinterpret_cast<const float*>      (W),
        r, B, N, q, q_pad,
        lambdaH, lambdaJ,
        reinterpret_cast<float*>  (pll_out),
        reinterpret_cast<float*>  (grad_pad),
        reinterpret_cast<__half*> (vgrad_pad),
        reinterpret_cast<float*>  (energies_fp32),
        reinterpret_cast<cublasHandle_t>(ext_handle),
        reinterpret_cast<cudaStream_t>  (stream)
    );
}


std::pair<int, float> cudaOptimizeSiteWrapper(
    intptr_t MSA_pad,       // (B, N, q_pad) fp16 – full one-hot
    intptr_t MSA,           // (B, N)        int8
    intptr_t W,             // (B,)          fp32
    int r, int B, int N, int q, int q_pad,
    float lambdaH, float lambdaJ,
    float eps_conv, int maxeval,
    intptr_t x_pad,                     // (n_params,)  fp16 in/out
    const std::map<std::string, float>& hyperparams,
    intptr_t ext_handle,
    intptr_t stream = 0
) {
    return cudaOptimizeSite(
        reinterpret_cast<const __half*>     (MSA_pad),
        reinterpret_cast<const signed char*>(MSA),
        reinterpret_cast<const float*>      (W),
        r, B, N, q, q_pad,
        lambdaH, lambdaJ, eps_conv, maxeval,
        reinterpret_cast<__half*>(x_pad),
        hyperparams,
        reinterpret_cast<cublasHandle_t>(ext_handle),
        reinterpret_cast<cudaStream_t>  (stream)
    );
}
#endif

PYBIND11_MODULE(cpp_bindings, m) {
    m.def("getAlignmentInNumpy", &getAlignmentInNumpy);
    m.def("applyIsingGauge", &applyIsingGauge, py::arg("J"), py::arg("q"));

#ifdef ENABLE_CUDA
    m.def("cudaFillPllGradients", &cudaFillPllGradientsWrapper,
        py::arg("MSA_pad_flat_ptr"),
        py::arg("x_pad_ptr"),
        py::arg("MSA_ptr"),
        py::arg("W_ptr"),
        py::arg("r"), py::arg("B"), py::arg("N"), py::arg("q"), py::arg("q_pad"),
        py::arg("lambdaH"), py::arg("lambdaJ"),
        py::arg("pll_out_ptr"),
        py::arg("grad_pad_ptr"),
        py::arg("vgrad_pad_ptr"),
        py::arg("energies_fp32_ptr"),
        py::arg("cublas_handle") = intptr_t(0),
        py::arg("stream") = intptr_t(0));
    m.def("cudaOptimizeSite", &cudaOptimizeSiteWrapper,
        py::arg("MSA_pad_ptr"),
        py::arg("MSA_ptr"),
        py::arg("W_ptr"),
        py::arg("r"), py::arg("B"), py::arg("N"), py::arg("q"), py::arg("q_pad"),
        py::arg("lambdaH"), py::arg("lambdaJ"),
        py::arg("eps_conv"), py::arg("maxeval"),
        py::arg("x_pad_ptr"),
        py::arg("hyperparams") = std::map<std::string, float>{},
        py::arg("cublas_handle") = intptr_t(0),
        py::arg("stream") = intptr_t(0),
        py::call_guard<py::gil_scoped_release>());
#endif
}
