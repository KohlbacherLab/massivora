// gauss_cuda_py -- the GaussDCA CUDA core as an importable module.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>
#include <vector>

#include "gauss_dca_cuda.cuh"

namespace py = pybind11;

namespace {

// One device Session plus the numpy plumbing
class PySession {
public:
    explicit PySession(int device) : sess_(new gaussdca::cuda::Session(device)) {}

    py::array_t<float> score(py::array_t<int8_t, py::array::c_style | py::array::forcecast> MSA,
                             py::array_t<double, py::array::c_style | py::array::forcecast> W,
                             py::object Meff_obj, int q, double pseudocount) {
        if (MSA.ndim() != 2)
            throw std::invalid_argument("MSA must be (N, M), residue-major");
        if (W.ndim() != 1)
            throw std::invalid_argument("W must be (M,)");
        const int N = (int)MSA.shape(0);
        const int M = (int)MSA.shape(1);
        if ((int)W.shape(0) != M)
            throw std::invalid_argument("W length must match MSA's second axis");
        double Meff;
        if (Meff_obj.is_none()) {
            const double* w = W.data();
            Meff = 0.0;
            for (int k = 0; k < M; ++k) Meff += w[k];
        } else {
            Meff = Meff_obj.cast<double>();
        }
        if (!(Meff > 0.0)) throw std::invalid_argument("Meff must be positive");

        py::array_t<float> out({(py::ssize_t)N, (py::ssize_t)N});
        const int8_t* zp = MSA.data();
        const double* wp = W.data();
        float* op = out.mutable_data();
        {
            // Everything below touches only the C++ side, so other threads can
            // load the next alignment while this one is on the device.
            py::gil_scoped_release unlock;
            sess_->score_into(zp, N, M, wp, Meff, q, pseudocount, op);
        }
        return out;
    }

    py::array_t<double> couplings(py::array_t<int8_t, py::array::c_style | py::array::forcecast> MSA,
                                  py::array_t<double, py::array::c_style | py::array::forcecast> W,
                                  py::object Meff_obj, int q, double pseudocount) {
        const int N = (int)MSA.shape(0), M = (int)MSA.shape(1);
        double Meff;
        if (Meff_obj.is_none()) {
            const double* w = W.data();
            Meff = 0.0;
            for (int k = 0; k < M; ++k) Meff += w[k];
        } else {
            Meff = Meff_obj.cast<double>();
        }
        const long NQ = (long)N * (q - 1);
        py::array_t<double> out({(py::ssize_t)NQ, (py::ssize_t)NQ});
        const int8_t* zp = MSA.data();
        const double* wp = W.data();
        double* op = out.mutable_data();
        {
            py::gil_scoped_release unlock;
            sess_->couplings_into(zp, N, M, wp, Meff, q, pseudocount, op);
        }
        return out;
    }

    std::string device_name() const { return sess_->device_name(); }
    std::string arithmetic() const { return sess_->arithmetic(); }
    py::dict timings() const {
        const auto& t = sess_->timings();
        py::dict d;
        d["upload"] = t.upload;       d["covariance"] = t.covariance;
        d["potrf"] = t.potrf;         d["trtri"] = t.trtri;
        d["lauum"] = t.lauum;         d["score"] = t.score;
        d["download"] = t.download;   d["total"] = t.total;
        return d;
    }
    void set_profile(bool on) { sess_->set_profile(on); }

private:
    std::unique_ptr<gaussdca::cuda::Session> sess_;
};

int device_count() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) return 0;
    return n;
}

py::dict device_info(int d) {
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, d) != cudaSuccess)
        throw std::runtime_error("no such CUDA device");
    py::dict info;
    info["name"] = std::string(p.name);
    info["sm"] = p.major * 10 + p.minor;
    info["total_mem_mb"] = (long)(p.totalGlobalMem >> 20);
    info["tensor_cores"] = (p.major >= 7);
    return info;
}

// Free device memory on `d`, so the caller can size a Session pool.
py::dict device_memory(int d) {
    int prev = 0;
    cudaGetDevice(&prev);
    if (cudaSetDevice(d) != cudaSuccess)
        throw std::runtime_error("cannot select CUDA device");
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    cudaSetDevice(prev);
    py::dict m;
    m["free_mb"] = (long)(free_b >> 20);
    m["total_mb"] = (long)(total_b >> 20);
    return m;
}

size_t pair_bytes(int N, int M, int q) {
    return gaussdca::cuda::Session::device_bytes<gaussdca::cuda::CovScalar>(N, M, q);
}

}  // namespace

PYBIND11_MODULE(gauss_cuda, m) {
    m.doc() = "GaussDCA CUDA core, in-process (see gauss_infer for the CLI)";
    py::class_<PySession>(m, "Session",
        "One device context worth of scratch. Not thread-safe -- give each\n"
        "worker thread its own; Sessions in one process share the device's\n"
        "primary CUDA context, so the pool costs buffers but not contexts.")
        .def(py::init<int>(), py::arg("device") = 0)
        .def("score", &PySession::score, py::arg("MSA"), py::arg("W"),
             py::arg("Meff") = py::none(), py::arg("q") = 21,
             py::arg("pseudocount") = 0.8,
             "(N, M) int8 alignment and (M,) weights -> (N, N) float32 score.\n"
             "Meff defaults to sum(W) computed exactly as the CLI does.")
        .def("couplings", &PySession::couplings, py::arg("MSA"), py::arg("W"),
             py::arg("Meff") = py::none(), py::arg("q") = 21,
             py::arg("pseudocount") = 0.8,
             "Full (NQ, NQ) float64 inverse; for reference comparisons.")
        .def_property_readonly("device_name", &PySession::device_name)
        .def_property_readonly("arithmetic", &PySession::arithmetic)
        .def_property_readonly("timings", &PySession::timings)
        .def("set_profile", &PySession::set_profile, py::arg("on"));
    m.def("device_count", &device_count);
    m.def("device_info", &device_info, py::arg("device"));
    m.def("device_memory", &device_memory, py::arg("device"));
    m.def("pair_bytes", &pair_bytes, py::arg("N"), py::arg("M"), py::arg("q") = 21,
          "Device memory one pair of this size needs, in bytes.");
}
