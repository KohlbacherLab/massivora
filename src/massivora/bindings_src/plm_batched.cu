// plm_batched.cu -- plm_opt_site's entry to the plmDCA Session.
#include <cstdio>

#include "plm_dca_cuda.cuh"
#include "plm_batched.h"

extern "C" int plmCouplingsBatched(
    const signed char* MSA, const double* W,
    int B, int N, int q,
    float lambdaH, float lambdaJ, int maxeval, int site_batch,
    float* J_out, PlmBatchStats* stats, char* err, size_t err_len)
{
    using plm::cuda::Session;
    try {
        int device = 0;
        PLM_CUDA(cudaGetDevice(&device));
        Session session(device);

        // A process of its own: there is no admission control, so fit the
        // batch to what the device has free once the context exists.
        size_t free_b = 0, total_b = 0;
        PLM_CUDA(cudaMemGetInfo(&free_b, &total_b));
        int batch = site_batch;
        while (batch > 1 && Session::device_bytes(device, B, N, q, batch, 1.0) > free_b)
            batch /= 2;

        plm::cuda::Config cfg;
        cfg.lambdaH = lambdaH;
        cfg.lambdaJ = lambdaJ;
        cfg.maxeval = maxeval;
        cfg.site_batch = batch;
        session.couplings_into(reinterpret_cast<const int8_t*>(MSA), W, B, N, q, cfg, J_out);

        if (stats) {
            const plm::cuda::Stats& s = session.last_stats();
            stats->slots = s.slots;
            stats->rounds = s.rounds;
            stats->evals = s.evals;
            stats->iterations = s.iterations;
            stats->converged = s.converged;
            stats->stalled = s.stalled;
            stats->max_iter = s.max_iter;
            stats->ls_failed = s.ls_failed;
            stats->f_sum = s.f_sum;
        }
        return 0;
    } catch (const plm::cuda::OutOfMemory& e) {
        if (err && err_len) std::snprintf(err, err_len, "%s", e.what());
        return 2;
    } catch (const plm::cuda::DeviceLost& e) {
        if (err && err_len) std::snprintf(err, err_len, "%s", e.what());
        return 3;
    } catch (const std::exception& e) {
        if (err && err_len) std::snprintf(err, err_len, "%s", e.what());
        return 1;
    }
}
