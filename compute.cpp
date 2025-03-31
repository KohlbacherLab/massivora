#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <string>
#include <vector>
#include <thread>
#include <iostream>

namespace py = pybind11;

py::array_t<char[1]> numpylize_alignment(const std::string& alignment, size_t seq_length, size_t align_length) {
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
    m.def("numpylize_alignment", &numpylize_alignment);
}

