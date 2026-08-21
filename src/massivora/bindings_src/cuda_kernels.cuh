#ifndef CUDA_KERNELS_CUH
#define CUDA_KERNELS_CUH

#include <cuda_fp16.h>

#define MAX_NB 32
#define MAX_Q  32
#define SS_BLOCK 256

extern "C" __global__ void fused_vgrad(
    const float*       __restrict__ energies,   // (B, q_pad) float32 
    const signed char* __restrict__ MSA,        // (B, N)     int8    
    const float*       __restrict__ W,          // (B,)       float32 
    int r,
    int B,
    int N,
    int q,                                      
    int q_pad,
    float* __restrict__ pll_out,                // (1,)       float32 – atomic 
    float* __restrict__ grad_h,                 // (q_pad,)   float32 – atomic 
    __half* __restrict__ vgrad_pad              // (B, q_pad) float16 
)
{
    // Thread layout mirrors the Python kernel:
    //   blockDim.x = q_pad   (tx iterates over alphabet dimension)
    //   blockDim.y = nb      (ty iterates over sequences within the block)
    //   b = blockIdx.x * blockDim.y + ty

    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int b  = blockIdx.x * blockDim.y + ty;

    if (b >= B || tx >= q_pad) return;

    __shared__ float sh_E   [MAX_NB][MAX_Q];
    __shared__ float sh_logZ[MAX_NB];

    // ------------------------------------------------------------------ 
    // 1. Load energy into shared memory (zero-pad if tx >= q)             
    // ------------------------------------------------------------------ 
    sh_E[ty][tx] = (tx < q) ? energies[b * q_pad + tx] : 0.0f;
    __syncthreads();

    // ------------------------------------------------------------------ 
    // 2. logsumexp per sequence b – computed by thread tx == 0            
    // ------------------------------------------------------------------ 
    if (tx == 0) {
        float max_e = sh_E[ty][0];
        for (int l = 1; l < q; ++l) {
            float v = sh_E[ty][l];
            if (v > max_e) max_e = v;
        }

        float sum_exp = 0.0f;
        for (int l = 0; l < q; ++l)
            sum_exp += expf(sh_E[ty][l] - max_e);

        sh_logZ[ty] = max_e + logf(sum_exp);
    }
    __syncthreads();

    const float logZ = sh_logZ[ty];

    // ------------------------------------------------------------------ 
    // 3. Softmax probability p_l = exp(E_l - logZ)                        
    // ------------------------------------------------------------------ 
    const float p_l = (tx < q) ? expf(sh_E[ty][tx] - logZ) : 0.0f;

    // ------------------------------------------------------------------ 
    // 4. vgrad = W[b] * (indicator(tx == s_r) - p_l)                     
    // ------------------------------------------------------------------ 
    const int   s_r = (int)MSA[b * N + r];
    const float w_b = W[b];

    if (tx < q) {
        const float vg = (tx == s_r) ? w_b * (1.0f - p_l)
                                      : -w_b * p_l;

        // grad_h[tx] -= vg  (accumulate over all b) 
        atomicAdd(&grad_h[tx], -vg);

        // store fp16 vgrad for downstream GEMM 
        vgrad_pad[b * q_pad + tx] = __float2half(vg);
    } else {
        vgrad_pad[b * q_pad + tx] = __float2half(0.0f);
    }

    // ------------------------------------------------------------------ 
    // 5. PLL contribution: -w_b * (E[s_r] - logZ)                        
    //    One thread per sequence (tx == 0) to avoid redundant adds.       
    // ------------------------------------------------------------------ 
    if (tx == 0) {
        const float e_sr = sh_E[ty][s_r];
        atomicAdd(pll_out, -w_b * (e_sr - logZ));
    }
}

/* =========================================================================
 * pairwise_similarity - sequence reweighting neighbour counts.
 *
 *   - Rows are staged through shared memory in TILE x TILE blocks, so each row
 *     is read from global memory once per tile rather than once per pair.
 *   - The alignment is read as uint32, four residues at a time, and matches are
 *     found with the zero-byte trick plus a popcount instead of four compares.
 *   - Shared tiles are stored transposed and padded to TILE+1 words so that both
 *     the staging writes and the inner-loop reads are bank-conflict free.
 *   - Hits are accumulated in shared memory and flushed with one global atomic
 *     per row per block, rather than two atomics per matching pair.
 *
 * The input is the alignment viewed as uint32, zero-padded on the right to a
 * whole number of words. Padding bytes are zero in every row, so they match in
 * every pair and contribute exactly `pad` to each count; the kernel subtracts
 * that back off.
 * =========================================================================*/
#define RW_TILE   32   // rows per tile
#define RW_CHUNKW 32   // uint32 words staged per pass (128 residues)
#define RW_TY      8   // threadIdx.y extent; each thread owns RW_PT rows
#define RW_PT      (RW_TILE / RW_TY)

extern "C" __global__ void pairwise_similarity_tiled(
    const unsigned int* __restrict__ msa32,  // (B, Nw) uint32, zero-padded
    int*                __restrict__ simM,   // (B,)    int32, atomic
    int   B,
    int   Nw,                                // words per sequence
    int   pad,                               // padding residues per row
    float threshold)
{
    const int tI = blockIdx.x, tJ = blockIdx.y;
    if (tJ < tI) return;                     // upper triangle of tiles only

    const int i0 = tI * RW_TILE, j0 = tJ * RW_TILE;
    if (i0 >= B || j0 >= B) return;

    __shared__ unsigned int sA[RW_CHUNKW][RW_TILE + 1];
    __shared__ unsigned int sB[RW_CHUNKW][RW_TILE + 1];
    __shared__ int cntA[RW_TILE], cntB[RW_TILE];

    const int tx = threadIdx.x;              // 0..RW_TILE-1 -> row in the J tile
    const int ty = threadIdx.y;              // 0..RW_TY-1                       
    const int tid = tx + ty * RW_TILE;
    const int nthread = RW_TILE * RW_TY;

    if (tid < RW_TILE) { cntA[tid] = 0; cntB[tid] = 0; }

    int acc[RW_PT];
#pragma unroll
    for (int k = 0; k < RW_PT; ++k) acc[k] = 0;

    for (int c0 = 0; c0 < Nw; c0 += RW_CHUNKW) {
        __syncthreads();
        // Stage both tiles. idx runs so that consecutive threads read
        // consecutive words of the same row (coalesced), while the shared
        // write strides by RW_TILE+1 words (odd, so conflict free).
        for (int idx = tid; idx < RW_TILE * RW_CHUNKW; idx += nthread) {
            const int r = idx / RW_CHUNKW, w = idx % RW_CHUNKW;
            const int c = c0 + w;
            const int ga = i0 + r, gb = j0 + r;
            sA[w][r] = (ga < B && c < Nw) ? msa32[(size_t)ga * Nw + c] : 0u;
            sB[w][r] = (gb < B && c < Nw) ? msa32[(size_t)gb * Nw + c] : 0u;
        }
        __syncthreads();

        const int wmax = (Nw - c0) < RW_CHUNKW ? (Nw - c0) : RW_CHUNKW;
        for (int w = 0; w < wmax; ++w) {
            const unsigned int bw = sB[w][tx];      // 32 distinct banks
#pragma unroll
            for (int k = 0; k < RW_PT; ++k) {
                const unsigned int aw = sA[w][ty + k * RW_TY];  // broadcast
                const unsigned int x = aw ^ bw;
                // Zero bytes of x are the residues that matched.
                const unsigned int nz =
                    ((x & 0x7F7F7F7Fu) + 0x7F7F7F7Fu) | x | 0x7F7F7F7Fu;
                acc[k] += __popc(~nz);
            }
        }
    }
    __syncthreads();

    const int j = j0 + tx;
#pragma unroll
    for (int k = 0; k < RW_PT; ++k) {
        const int i = i0 + ty + k * RW_TY;
        if (i < B && j < B && i < j && (float)(acc[k] - pad) >= threshold) {
            atomicAdd(&cntA[ty + k * RW_TY], 1);
            atomicAdd(&cntB[tx], 1);
        }
    }
    __syncthreads();

    if (tid < RW_TILE) {
        if (i0 + tid < B && cntA[tid]) atomicAdd(&simM[i0 + tid], cntA[tid]);
        if (j0 + tid < B && cntB[tid]) atomicAdd(&simM[j0 + tid], cntB[tid]);
    }
}


extern "C" __global__ void grad_l2(
    float* __restrict__ grad_h,     // (q_pad,)                 fp32, inout   
    float* __restrict__ grad_J,     // (N*q_pad, q_pad) fp32 
    const __half* __restrict__ h_pad,      // (q_pad,)          fp16 
    const __half* __restrict__ Jr_pad,     // (N, q_pad, q_pad) fp16, C-order 
    int N,
    int q,
    int q_pad,
    float lambdaH,
    float lambdaJ
)
{
    const int n = blockIdx.x;
    const int a = threadIdx.x;   // output row 
    const int b = threadIdx.y;   // output col 

    if (n >= N || a >= q || b >= q) return;

    // ---------- grad_J --------
    // Both Jr_pad and grad_J are in [i, k, s] layout (same as GEMM2 output).
    // L2 regularization: grad_J[n, a, b] = lambdaJ * Jr_pad[n, a, b] - grad_J[n, a, b]
    // No axis swap needed — same position for both.

    const int flat     = n * q_pad * q_pad + a * q_pad + b;
    const float jr_val = __half2float(Jr_pad[flat]);

    grad_J[flat] = lambdaJ * jr_val - grad_J[flat];

    // ---------- grad_h --------
    // each a only need update once, run by thread (n==0, b==0), without race condition
    if (n == 0 && b == 0) {
        grad_h[a] += 2.0f * lambdaH * __half2float(h_pad[a]);
    }
}

extern "C" __global__ void sum_squares(
    const __half* __restrict__ h_pad,    // (q_pad,)         fp16 
    const __half* __restrict__ Jr_flat,  // (N*q_pad*q_pad,) fp16 
    float*        __restrict__ pll_out,  // [2] and [3]      fp32, atomic-add 
    int q,
    int q_pad,
    int Jr_total                         // N * q_pad * q_pad 
)
{
    __shared__ float sh_h[SS_BLOCK];
    __shared__ float sh_J[SS_BLOCK];

    const int tid    = threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int total  = Jr_total > q_pad ? Jr_total : q_pad;

    float acc_h = 0.0f, acc_J = 0.0f;

    for (int i = blockIdx.x * blockDim.x + tid; i < total; i += stride) {
        if (i < q) {
            const float v = __half2float(h_pad[i]);
            acc_h += v * v;
        }
        if (i < Jr_total) {
            // Jr_flat layout: (N, q_pad, q_pad), only accumulate real q×q block
            const int pos = i % (q_pad * q_pad);  // position within one (q_pad, q_pad) slice
            const int a   = pos / q_pad;           // row index
            const int b   = pos % q_pad;           // col index
            if (a < q && b < q) {
                const float v = __half2float(Jr_flat[i]);
                acc_J += v * v;
            }
        }
    }

    sh_h[tid] = acc_h;
    sh_J[tid] = acc_J;
    __syncthreads();

    // Block-level reduction (assumes SS_BLOCK is a power of 2) 
    for (int s = SS_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sh_h[tid] += sh_h[tid + s];
            sh_J[tid] += sh_J[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(&pll_out[2], sh_h[0]);
        atomicAdd(&pll_out[3], sh_J[0]);
    }
}

/* =========================================================================
 * half2float_copy – cast fp16 array to fp32
 *   Grid: ((n+255)/256,)  Block: (256,)
 * =========================================================================*/
__global__ void half2float_copy(const __half* __restrict__ src,
                                   float*        __restrict__ dst,
                                   int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __half2float(src[i]);
}

/* =========================================================================
 * float2half_copy – cast fp32 array to fp16
 *   Grid: ((n+255)/256,)  Block: (256,)
 * =========================================================================*/
__global__ void float2half_copy(const float* __restrict__ src,
                                   __half*      __restrict__ dst,
                                   int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

/* =========================================================================
 * pll_l2 – GPU equivalent of the CuPy @fuse pll_l2 kernel:
 *   pll_out[1] += lambdaH * pll_out[2] + 0.5f * lambdaJ * pll_out[3]
 * =========================================================================*/
__global__ void pll_l2(float* __restrict__ pll_out,
                              float lambdaH, float lambdaJ)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
        pll_out[1] += lambdaH * pll_out[2] + 0.5f * lambdaJ * pll_out[3];
}

/* =========================================================================
 * fill_scalar_float – write a single float value to a device scalar
 *   Grid: (1,)  Block: (1,)
 * =========================================================================*/
__global__ void fill_scalar_float(float* __restrict__ ptr, float val)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) *ptr = val;
}

/* =========================================================================
 * broadcast_hr_to_energies – broadcast fp16 hr_pad vector to all B rows of
 *   fp32 energies_fp32: dst[b, s] = (float)hr_pad[s]
 *   Flat 1D launch for high occupancy.
 *   Grid : ((B*q_pad+255)/256,)  Block : (256,)
 * =========================================================================*/
__global__ void broadcast_hr_to_energies(
    const __half* __restrict__ hr_pad,  /* (q_pad,)   fp16 */
    float*        __restrict__ dst,     /* (B, q_pad) fp32 */
    int q_pad,
    int total)                          /* B * q_pad        */
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total)
        dst[idx] = __half2float(hr_pad[idx % q_pad]);
}

/* =========================================================================
 * msa_one_hot – expand the (B, N) int8 MSA into the (B, N*q_pad) fp16 one-hot
 *   matrix the pll GEMMs consume:
 *       MSA_pad[(b*N + i)*q_pad + l] = (l == MSA[b*N + i]) ? 1 : 0
 *   Building it here keeps a B*N*q_pad*2 byte buffer off the host and off the
 *   PCIe bus; only the B*N int8 matrix is uploaded.
 *   Flat grid-stride launch: consecutive threads write consecutive elements.
 * =========================================================================*/
__global__ void msa_one_hot(
    const signed char* __restrict__ MSA,      /* (B, N)       int8 */
    __half*            __restrict__ MSA_pad,  /* (B, N*q_pad) fp16 */
    long long total,                          /* B * N * q_pad     */
    int q_pad)
{
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += stride) {
        const int l = (int)(idx % q_pad);
        const int s = MSA[idx / q_pad];
        MSA_pad[idx] = __float2half(l == s ? 1.0f : 0.0f);
    }
}

extern "C" __global__ void NAdam_step(
    float*             __restrict__ param,        // (n_elem,) fp32 in/out          
    const float*       __restrict__ grad,         // (n_elem,) fp32 read-only
    float*             __restrict__ exp_avg,      // (n_elem,) fp32 in/out
    float*             __restrict__ exp_avg_sq,   // (n_elem,) fp32 in/out
    float*             __restrict__ mu_product,   // (1,)      fp32 in/out
    int    step,                  
    int    n_elem,                                // total params (include padding)
    float  lr,
    float  beta1,
    float  beta2,
    float  eps,
    float  weight_decay,
    float  momentum_decay
)
{
    // ------------------------------------------------------------------ 
    // 1. read mu_product from global memory
    // ------------------------------------------------------------------ 
    const float mu_product_in = *mu_product;

    // ------------------------------------------------------------------ 
    // 2. calculate the hyperparameters
    // ------------------------------------------------------------------ 
    const float step_f = (float)step;

    const float bias_correction2 = 1.0f - powf(beta2, step_f);

    // μ_t = β1 * (1 - 0.5 * 0.96^(t * decay)) 
    const float mu      = beta1 * (1.0f - 0.5f * powf(0.96f,  step_f         * momentum_decay));
    const float mu_next = beta1 * (1.0f - 0.5f * powf(0.96f, (step_f + 1.0f) * momentum_decay));

    // μ̂_t = (∏_{i=1}^{t} μ_i) ≈ mu_product_in * mu  
    const float mu_product_new  = mu_product_in * mu;
    // μ̂_{t+1} = μ̂_t * μ_{t+1}                       
    const float mu_product_next = mu_product_new * mu_next;

    // NAdam two terms of learning rate
    const float coeff1 = -lr * (1.0f - mu)  / (1.0f - mu_product_new);
    const float coeff2 = -lr * mu_next       / (1.0f - mu_product_next);

    // ------------------------------------------------------------------ 
    // 3. per-element update
    // ------------------------------------------------------------------ 
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_elem;
         i += gridDim.x * blockDim.x)
    {
        float p = param[i];
        float g = grad[i];

        // L2-grad weight decay: grad += weight_decay * param NOT USED 
        if (weight_decay != 0.0f)
            g += weight_decay * p;

        // EMA of grad: m = lerp(m, g, 1-beta1) 
        float m = exp_avg[i];
        m += (1.0f - beta1) * (g - m);            // = beta1*m + (1-beta1)*g 

        // EMA of grad^2
        float v = exp_avg_sq[i];
        v = beta2 * v + (1.0f - beta2) * g * g;

        // sqrt(v / bias_correction2) + eps 
        const float denom = sqrtf(v / bias_correction2) + eps;

        // two addcdiv 
        p += coeff1 * g / denom;   // gradient with bias correction
        p += coeff2 * m / denom;   // first moment with bias correction

        param[i]      = p;
        exp_avg[i]    = m;
        exp_avg_sq[i] = v;
    }

    // ------------------------------------------------------------------ 
    // 4. write back new value of mu_product
    // ------------------------------------------------------------------ 
    if (blockIdx.x == 0 && threadIdx.x == 0)
        *mu_product = mu_product_new;
}

extern "C" __global__ void RMSprop_step(
    float*       __restrict__ param,         // (n_elem,) fp32 in/out
    const float* __restrict__ grad,          // (n_elem,) fp32 read-only
    float*       __restrict__ square_avg,    // (n_elem,) fp32 in/out
    float*       __restrict__ grad_avg,      // (n_elem,) fp32 in/out  (centered only)
    float*       __restrict__ momentum_buf,  // (n_elem,) fp32 in/out  (momentum > 0 only)
    int   n_elem,
    float lr,
    float alpha,
    float eps,
    float weight_decay,
    float momentum,
    int   centered                           // 0 = false, 1 = true
)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_elem;
         i += gridDim.x * blockDim.x)
    {
        float p = param[i];
        float g = grad[i];

        // L2 weight decay: grad += weight_decay * param
        if (weight_decay != 0.0f)
            g += weight_decay * p;

        // Running average of squared gradients
        float sq = square_avg[i];
        sq = alpha * sq + (1.0f - alpha) * g * g;
        square_avg[i] = sq;

        // Denominator
        float avg;
        if (centered) {
            float ga = grad_avg[i];
            ga = alpha * ga + (1.0f - alpha) * g;   // lerp_(grad, 1-alpha)
            grad_avg[i] = ga;
            avg = sqrtf(sq - ga * ga + eps);         // addcmul(-1).sqrt + eps inside sqrt
        } else {
            avg = sqrtf(sq) + eps;
        }

        // Update with or without momentum
        if (momentum != 0.0f) {
            float buf = momentum_buf[i];
            buf = momentum * buf + g / avg;          // buf.mul_(m).addcdiv_(grad, avg)
            momentum_buf[i] = buf;
            p -= lr * buf;                           // param.add_(buf, alpha=-lr)
        } else {
            p -= lr * g / avg;                       // param.addcdiv_(grad, avg, value=-lr)
        }

        param[i] = p;
    }
}

extern "C" __global__ void apply_ising_gauge(
    float* __restrict__ J,   /* (N, N, q, q) fp32 in/out */
    int N,
    int q
)
{
    const int i = blockIdx.x;    // first site index
    const int j = blockIdx.y;    // second site index
    const int k = threadIdx.x;   // row index
    const int l = threadIdx.y;   // col index

    if (i >= N || j >= N || k >= q || l >= q) return;

    /* shared memory：row_sum[q], col_sum[q], total_sum[1] */
    __shared__ float row_sum  [MAX_Q];
    __shared__ float col_sum  [MAX_Q];
    __shared__ float total_sum[1];

    if (l == 0) row_sum[k] = 0.0f;
    if (k == 0) col_sum[l] = 0.0f;
    if (k == 0 && l == 0) total_sum[0] = 0.0f;
    __syncthreads();

    const int flat = (i * N + j) * q * q + k * q + l;
    const float val = J[flat];

    atomicAdd(&row_sum[k], val);
    atomicAdd(&col_sum[l], val);
    __syncthreads();

    if (l == 0) atomicAdd(&total_sum[0], row_sum[k]);
    __syncthreads();

    const float inv_q  = 1.0f / (float)q;
    const float inv_q2 = inv_q * inv_q;

    const float rm = row_sum[k]   * inv_q;
    const float cm = col_sum[l]   * inv_q;
    const float tm = total_sum[0] * inv_q2;

    J[flat] = val - rm - cm + tm;
}

#endif