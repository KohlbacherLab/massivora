#ifndef BINDINGS_CUH
#define BINDINGS_CUH

/* Expand the (B, N) int8 MSA already resident on the device into the
 * (B, N*q_pad) fp16 one-hot matrix the pll GEMMs consume. */
extern "C" void cudaBuildMsaOneHot(
    const signed char* MSA,
    __half*            MSA_pad,
    int B, int N, int q_pad,
    cudaStream_t stream
);

extern "C" void cudaFillPllGradients(
    const __half*      MSA_pad_flat,
    const __half*      x_pad,
    const signed char* MSA,
    const float*       W,
    int r, int B, int N, int q, int q_pad,
    float lambdaH, float lambdaJ,
    float*         pll_out,
    float*         grad_pad,
    __half*        vgrad_pad,
    float*         energies_fp32,
    cublasHandle_t ext_handle,
    cudaStream_t   stream
);

std::pair<int, float> cudaOptimizeSite(
    const __half*      MSA_pad,
    const signed char* MSA,
    const float*       W,
    int r, int B, int N, int q, int q_pad,
    float lambdaH, float lambdaJ,
    float eps_conv, int maxeval,
    __half*        x_pad,
    const std::map<std::string, float>& hyperparams,
    cublasHandle_t ext_handle,
    cudaStream_t   stream
);

#endif // BINDINGS_CUH