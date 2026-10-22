// plm_cuda_py -- the batched plmDCA CUDA core as an importable module.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>

#include "plm_dca_cuda.cuh"

namespace py = pybind11;

namespace {

using MSAArray = py::array_t<int8_t, py::array::c_style | py::array::forcecast>;
using WArray   = py::array_t<double, py::array::c_style | py::array::forcecast>;

// One device Session plus the numpy plumbing
class PySession {
public:
    PySession(int device, double l2_share)
        : sess_(new plm::cuda::Session(device, l2_share)) {}

    py::array_t<float> score(MSAArray MSA, WArray W, int q, float lambdaH, float lambdaJ,
                             int maxeval, int site_batch, float eps_g) {
        int B, N;
        shape(MSA, W, &B, &N);
        py::array_t<float> out({(py::ssize_t)N, (py::ssize_t)N});
        const plm::cuda::Config cfg = config(lambdaH, lambdaJ, maxeval, site_batch, eps_g);
        const int8_t* zp = MSA.data();
        const double* wp = W.data();
        float* op = out.mutable_data();
        {
            // Everything below touches only the C++ side, so other threads can
            // load the next alignment while this one is on the device.
            py::gil_scoped_release unlock;
            sess_->score_into(zp, wp, B, N, q, cfg, op);
        }
        return out;
    }

    py::array_t<float> couplings(MSAArray MSA, WArray W, int q, float lambdaH, float lambdaJ,
                                 int maxeval, int site_batch, float eps_g) {
        int B, N;
        shape(MSA, W, &B, &N);
        py::array_t<float> out({(py::ssize_t)N, (py::ssize_t)N * q * q});
        const plm::cuda::Config cfg = config(lambdaH, lambdaJ, maxeval, site_batch, eps_g);
        const int8_t* zp = MSA.data();
        const double* wp = W.data();
        float* op = out.mutable_data();
        {
            py::gil_scoped_release unlock;
            sess_->couplings_into(zp, wp, B, N, q, cfg, op);
        }
        return out;
    }

    py::array_t<float> score_couplings(
            py::array_t<float, py::array::c_style | py::array::forcecast> J, int q) {
        if (J.ndim() != 2 || J.shape(1) != J.shape(0) * q * q)
            throw std::invalid_argument("J must be (N, N*q*q)");
        const int N = (int)J.shape(0);
        py::array_t<float> out({(py::ssize_t)N, (py::ssize_t)N});
        const float* jp = J.data();
        float* op = out.mutable_data();
        {
            py::gil_scoped_release unlock;
            sess_->score_couplings_into(jp, N, q, op);
        }
        return out;
    }

    py::array_t<int32_t> neighbour_counts(MSAArray MSA, double identical_threshold) {
        if (MSA.ndim() != 2)
            throw std::invalid_argument("MSA must be (B, N), sequence-major");
        const int B = (int)MSA.shape(0), N = (int)MSA.shape(1);
        py::array_t<int32_t> out((py::ssize_t)B);
        const int8_t* zp = MSA.data();
        int32_t* op = out.mutable_data();
        {
            py::gil_scoped_release unlock;
            sess_->neighbour_counts_into(zp, B, N, identical_threshold, op);
        }
        return out;
    }

    py::dict stats() const {
        const plm::cuda::Stats& s = sess_->last_stats();
        py::dict d;
        d["slots"] = s.slots;           d["rounds"] = s.rounds;
        d["evals"] = s.evals;           d["iterations"] = s.iterations;
        d["converged"] = s.converged;   d["stalled"] = s.stalled;
        d["max_iter"] = s.max_iter;     d["ls_failed"] = s.ls_failed;
        d["f_sum"] = s.f_sum;
        return d;
    }
    size_t last_pair_bytes() const { return sess_->last_pair_bytes(); }
    std::string device_name() const { return sess_->device_name(); }
    int device() const { return sess_->device(); }
    double l2_share() const { return sess_->l2_share(); }
    void trim() {
        py::gil_scoped_release unlock;
        sess_->trim();
    }

private:
    static void shape(const MSAArray& MSA, const WArray& W, int* B, int* N) {
        if (MSA.ndim() != 2)
            throw std::invalid_argument("MSA must be (B, N), sequence-major");
        if (W.ndim() != 1)
            throw std::invalid_argument("W must be (B,)");
        *B = (int)MSA.shape(0);
        *N = (int)MSA.shape(1);
        if ((int)W.shape(0) != *B)
            throw std::invalid_argument("W length must match MSA's first axis");
    }
    static plm::cuda::Config config(float lambdaH, float lambdaJ, int maxeval, int site_batch, float eps_g) {
        plm::cuda::Config cfg;
        cfg.lambdaH = lambdaH;
        cfg.lambdaJ = lambdaJ;
        cfg.maxeval = maxeval;
        cfg.site_batch = site_batch;
        cfg.eps_g = eps_g;
        return cfg;
    }

    std::unique_ptr<plm::cuda::Session> sess_;
};

int device_count() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) return 0;
    return n;
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

size_t pair_bytes(int B, int N, int q, int site_batch, int device, double l2_share) {
    return plm::cuda::Session::device_bytes(device, B, N, q, site_batch, l2_share);
}

int slots(int B, int N, int q, int site_batch, int device, double l2_share) {
    return plm::cuda::Session::slots(device, B, N, q, site_batch, l2_share);
}

}  // namespace

PYBIND11_MODULE(plm_cuda, m) {
    m.doc() = "Batched plmDCA CUDA core, in-process (see plm_opt_site --use-gpu for the CLI)";
    py::class_<PySession>(m, "Session",
        "One device context worth of scratch. Not thread-safe -- give each\n"
        "worker thread its own; Sessions in one process share the device's\n"
        "primary CUDA context, so the pool costs buffers but not contexts.\n"
        "l2_share is the part of the persisting L2 this Session may pin:\n"
        "1 / (Sessions on the device) keeps them from evicting each other's.")
        .def(py::init<int, double>(), py::arg("device") = 0, py::arg("l2_share") = 1.0)
        .def("score", &PySession::score, py::arg("MSA"), py::arg("W"), py::arg("q"),
             py::arg("lambdaH") = 0.01f, py::arg("lambdaJ") = 0.01f,
             py::arg("maxeval") = 200, py::arg("site_batch") = 24, py::arg("eps_g") = 1e-3f,
             "(B, N) int8 alignment (states 0 .. q-1) and (B,) weights divided by\n"
             "Beff -> (N, N) float32 score: compute_score of the Ising-gauged\n"
             "couplings, computed on the device. Takes at most\n"
             "pair_bytes(B, N, q, site_batch) of device memory. Raises\n"
             "OutOfMemoryError when the device cannot hold the pair, and\n"
             "DeviceLostError when the CUDA context is gone for the process.")
        .def("couplings", &PySession::couplings, py::arg("MSA"), py::arg("W"), py::arg("q"),
             py::arg("lambdaH") = 0.01f, py::arg("lambdaJ") = 0.01f,
             py::arg("maxeval") = 200, py::arg("site_batch") = 24, py::arg("eps_g") = 1e-3f,
             "The raw (N, N*q*q) float32 couplings instead, as plm_opt_site --use-gpu\n"
             "publishes them; for reference comparisons.")
        .def("score_couplings", &PySession::score_couplings, py::arg("J"), py::arg("q"),
             "(N, N*q*q) couplings -> (N, N) score by the device path score() takes,\n"
             "J rounded to fp16 as the optimiser keeps it; for checking that path\n"
             "against compute_score.")
        .def("neighbour_counts", &PySession::neighbour_counts, py::arg("MSA"),
             py::arg("identical_threshold"),
             "(B, N) int8 alignment -> (B,) int32 reweighting neighbour counts: how\n"
             "many sequences, each one itself included, match it at >=\n"
             "identical_threshold (x * N) positions. The same counts as\n"
             "cpp_bindings.reweightNeighbourCounts; the weights are 1 / counts. Takes\n"
             "less device memory than pair_bytes(B, N, ...) of the same alignment.")
        .def_property_readonly("last_stats", &PySession::stats,
             "Optimiser counters of the last pair.")
        .def_property_readonly("last_pair_bytes", &PySession::last_pair_bytes,
             "Device memory the last successful call took, in bytes.")
        .def_property_readonly("device_name", &PySession::device_name)
        .def_property_readonly("device", &PySession::device)
        .def_property_readonly("l2_share", &PySession::l2_share)
        .def("trim", &PySession::trim,
             "Hand the device memory the pool keeps cached back to the driver.");
    // The pair's memory is back in the pool and the next pair can run.
    py::register_exception<plm::cuda::OutOfMemory>(m, "OutOfMemoryError", PyExc_MemoryError);
    // Sticky: every later CUDA call in this process fails too.
    py::register_exception<plm::cuda::DeviceLost>(m, "DeviceLostError", PyExc_RuntimeError);
    m.def("device_count", &device_count);
    m.def("device_memory", &device_memory, py::arg("device"));
    m.def("pair_bytes", &pair_bytes, py::arg("B"), py::arg("N"), py::arg("q") = 21,
          py::arg("site_batch") = 24, py::arg("device") = 0, py::arg("l2_share") = 1.0,
          "Device memory one pair of this size needs, in bytes; what admission\n"
          "control reserves for it.");
    m.def("slots", &slots, py::arg("B"), py::arg("N"), py::arg("q") = 21,
          py::arg("site_batch") = 24, py::arg("device") = 0, py::arg("l2_share") = 1.0,
          "Sites such a pair optimises in lockstep.");
}
