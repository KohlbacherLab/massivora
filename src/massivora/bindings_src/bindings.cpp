#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <cstdint>
#include <iostream>
#include <map>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "gauss_dca.hpp"

namespace py = pybind11;
using PLMV = double;


// GaussDCA couplings J = inv(cholesky(C)) for one alignment.
//
//   MSA         : (M, N) integer alignment. Gap = 0, amino acids 1..q-1
//                 (massivora BinaryAlignment encoding). Values outside
//                 [1, q-1] are treated as gaps and skipped in the frequencies.
//   W           : (M,) reweighting vector (NOT normalised; Meff = sum(W)).
//   q           : alphabet size (21 for proteins => Q = q-1 = 20 amino-acid blocks).
//   pseudocount : mixing weight with the uniform distribution (GaussDCA default 0.8).
//
// Returns J as a C-contiguous (N*Q, N*Q) float64 array (Q = q-1).
py::array_t<double> gaussCouplings(py::array MSA,
                                   py::array_t<double, py::array::c_style | py::array::forcecast> W,
                                   int q = 21,
                                   double pseudocount = 0.8) {
    auto MSA_i32 = py::array_t<int32_t, py::array::c_style | py::array::forcecast>(MSA);
    auto msa_buf = MSA_i32.request();
    if (msa_buf.ndim != 2)
        throw std::runtime_error("gaussCouplings: MSA must be 2-D (M, N)");
    const int M = static_cast<int>(msa_buf.shape[0]);
    const int N = static_cast<int>(msa_buf.shape[1]);

    auto wbuf = W.request();
    if (wbuf.ndim != 1 || static_cast<int>(wbuf.shape[0]) != M)
        throw std::runtime_error("gaussCouplings: W must be 1-D of length M");
    if (q < 2)
        throw std::runtime_error("gaussCouplings: q must be >= 2");

    // Transpose into the residue-major int8 buffer Z[i*M + k] the core expects.
    const int32_t* msa_ptr = static_cast<const int32_t*>(msa_buf.ptr);
    std::vector<int8_t> Z(static_cast<size_t>(N) * M);
    for (int k = 0; k < M; ++k) {
        const int32_t* row = msa_ptr + static_cast<size_t>(k) * N;
        for (int i = 0; i < N; ++i)
            Z[static_cast<size_t>(i) * M + k] = static_cast<int8_t>(row[i]);
    }

    const double* Wp = static_cast<const double*>(wbuf.ptr);
    double Meff = 0.0;
    for (int k = 0; k < M; ++k) Meff += Wp[k];

    const long NQ = static_cast<long>(N) * (q - 1);
    py::array_t<double> out({static_cast<py::ssize_t>(NQ), static_cast<py::ssize_t>(NQ)});
    double* outp = static_cast<double*>(out.request().ptr);

    {
        py::gil_scoped_release release;
        Eigen::MatrixXd J = gaussdca::compute_couplings(Z.data(), N, M, Wp, Meff, q, pseudocount);
        for (long r = 0; r < NQ; ++r)
            for (long c = 0; c < NQ; ++c)
                outp[static_cast<size_t>(r) * NQ + c] = J(r, c);
    }
    return out;
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

// In-place zero-sum (Ising) gauge on every q×q block of a C-contiguous,
// row-major flat buffer. GPU path only.
template <typename T>
void applyIsingGauge(py::array_t<T> J, int q) {
    auto buf = J.request();
    if (buf.readonly)
        throw std::runtime_error("applyIsingGauge: array must be writeable");
    // Require C-contiguous so the flat q×q block walk is valid and in place.
    py::ssize_t expected = buf.itemsize;
    for (py::ssize_t d = buf.ndim - 1; d >= 0; --d) {
        if (buf.strides[d] != expected)
            throw std::runtime_error("applyIsingGauge: array must be C-contiguous");
        expected *= buf.shape[d];
    }

    T* data = static_cast<T*>(buf.ptr);
    const size_t n = static_cast<size_t>(buf.size);
    const size_t site_size = static_cast<size_t>(q * q);

    if (site_size == 0 || n % static_cast<size_t>(site_size) != 0)
        throw std::runtime_error("applyIsingGauge: array size must be a multiple of q*q");
    for (size_t off = 0; off + site_size <= n; off += site_size) {
        Eigen::Map<Eigen::Matrix<T, -1, -1, Eigen::RowMajor>> Jij(data + off, q, q);
        Eigen::Matrix<T, -1, 1> row_mean = Jij.rowwise().mean();
        Eigen::Matrix<T, -1, 1> col_mean = Jij.colwise().mean();
        T total_mean = row_mean.mean();

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
    // Overloaded on dtype: float32 (GPU J) and float64 (CPU J). pybind dispatches
    // by the array's dtype without a copy, so the gauge is applied in place.
    m.def("applyIsingGauge", &applyIsingGauge<double>, py::arg("J"), py::arg("q"));
    m.def("applyIsingGauge", &applyIsingGauge<float>,  py::arg("J"), py::arg("q"));

    m.def("gaussCouplings", &gaussCouplings,
        py::arg("MSA"), py::arg("W"), py::arg("q") = 21,
        py::arg("pseudocount") = 0.8);

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
