/*
 * rh_cuda.cu  —  Riemann-Siegel Z-function, GPU-accelerated
 *
 * ALGORITHM: Odlyzko-Schönhage style NUFFT replaces the O(√t) Riemann-Siegel scan.
 *
 * Instead of evaluating Σ n^{-½-it_j} independently for each t_j (O(N) per point),
 * the Non-Uniform FFT evaluates all K t-values simultaneously in O(N·W + M·log M)
 * where M = next-power-of-2(K) and W=6 is the Kaiser-Bessel window width.
 *
 * For t~3e12, N=691K, K=2.5M: ~170MB GPU, ~6ms per chunk vs ~51s with RS (~8500x).
 *
 * Math:
 *   F(j) = Σ_{n=1}^{N} n^{-½-it_j}  where t_j = t_0 + j·h
 *        = Σ_n a_n · e^{-2πij·u_n}   (NUFFT Type 1)
 *
 *   a_n = (1/√n)·e^{-it_0·ln n},  u_n = h·ln(n)/(2π)
 *
 *   Spread:     g[m] = Σ_n a_n · φ(m − M·u_n)    (Kaiser-Bessel window)
 *   FFT:        G[k] = FFT{g}[k]
 *   Deconvolve: F(j) = G[j] / Φ̂[j]               (Φ̂[j] = DFT of φ at freq j)
 *   Z(t_j)    = 2·Re[e^{iθ(t_j)} · F(j)] + C₀
 *
 * Zero detection: sign changes in Z on fine grid (h=NUFFT_H=0.002), no bisection.
 * Zero positions accurate to NUFFT_H/2 = 0.001 — sufficient for Backlund counting.
 *
 * Interface: reads job JSON from stdin, outputs result JSON to stdout.
 * Build: nvcc -gencode arch=compute_120,code=sm_120 -O3 -std=c++17
 *        -Xcompiler "-O3 -march=native" -o rh_worker_cuda rh_cuda.cu -lm -lcufft -cudart static
 */

#include <cuda_runtime.h>
#include <cufft.h>
#include <cuComplex.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <sys/time.h>
#include <unistd.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>

#include "rh_stats.h"

#include <cub/device/device_select.cuh>
#include <thrust/iterator/counting_iterator.h>

// ─── tunables ──────────────────────────────────────────────────────────────

#define THREADS    256

// NUFFT parameters
#define NUFFT_W     6        // Kaiser-Bessel window half-width (grid pts)
#define NUFFT_BETA  13.8     // KB shape param → ~1e-6 accuracy
// Fine scan step: 235 samples per average zero gap (0.47); reliable sign detection
#define NUFFT_H     0.002

// ─── host math ─────────────────────────────────────────────────────────────

static inline double theta_host(double t) {
    double t2=t*t, t3=t2*t, t5=t3*t2;
    return (t/2.)*log(t/(2.*M_PI)) - t/2. - M_PI/8.
         + 1./(48.*t) + 7./(5760.*t3) - 31./(80640.*t5);
}

static inline long long backlund_count(double T) {
    return (long long)floor(theta_host(T)/M_PI) + 1LL;
}

// Modified Bessel I₀ — A&S polynomial approximation
static double bessel_I0_host(double x) {
    if (x < 3.75) {
        double t = (x*x)/(3.75*3.75);
        return 1.0 + t*(3.5156329 + t*(3.0899424 + t*(1.2067492
             + t*(0.2659732 + t*(0.0360768 + t*0.0045813)))));
    }
    double t = 3.75/x;
    double p = 0.39894228 + t*(0.01328592 + t*(0.00225319
           + t*(-0.00157565 + t*(0.00916281 + t*(-0.02057706
           + t*(0.02635537 + t*(-0.01647633 + t*0.00392377)))))));
    return exp(x)/sqrt(x)*p;
}

// ─── device math ──────────────────────────────────────────────────────────

__device__ __forceinline__ double theta_dev(double t) {
    double t2=t*t, t3=t2*t, t5=t3*t2;
    return (t/2.)*log(t/(2.*M_PI)) - t/2. - M_PI/8.
         + 1./(48.*t) + 7./(5760.*t3) - 31./(80640.*t5);
}

__device__ double bessel_I0_dev(double x) {
    if (x < 3.75) {
        double t = (x*x)/(3.75*3.75);
        return 1.0 + t*(3.5156329 + t*(3.0899424 + t*(1.2067492
             + t*(0.2659732 + t*(0.0360768 + t*0.0045813)))));
    }
    double t = 3.75/x;
    double p = 0.39894228 + t*(0.01328592 + t*(0.00225319
           + t*(-0.00157565 + t*(0.00916281 + t*(-0.02057706
           + t*(0.02635537 + t*(-0.01647633 + t*0.00392377)))))));
    return exp(x)/sqrt(x)*p;
}

// KB window: φ(dx) = I₀(β·√(1-(2dx/W)²)) / I₀(β),  |dx| ≤ W/2
__device__ __forceinline__
double kb_window_dev(double dx, double beta, double I0b_inv) {
    double r = dx * (2.0 / NUFFT_W);
    double arg = 1.0 - r*r;
    if (arg <= 0.0) return 0.0;
    return bessel_I0_dev(beta * sqrt(arg)) * I0b_inv;
}

// ─── NUFFT Phase 1: spread a_n onto complex grid ───────────────────────────
//
// For each n=1..N_max:
//   a_n = (1/√n)·e^{-it0·ln n},  xpos = M·h·ln(n)/(2π)
//   g[m] += a_n·φ(m − xpos)  for m in [xpos-W/2, xpos+W/2]
//
// Uses FP64 atomicAdd (hardware-native on sm_120).

__global__ void nufft_spread_kernel(
    double t0, double h, int N_max, int M,
    double beta, double I0b_inv,
    const double* __restrict__ logn_corr,     // [n-1]: logl(n) − log(n)
    cufftDoubleComplex* __restrict__ grid_cx
) {
    int n = (int)(blockIdx.x*(unsigned)blockDim.x + threadIdx.x) + 1;
    if (n > N_max) return;

    double logn  = log((double)n);
    double isqn  = rsqrt((double)n);

    // Pre-phase: a_n = (1/√n)·e^{-it₀·ln n}
    // 2-product for t0*logn, then add t0*corr to reduce log error from ~2e-3 to ~7e-9 rad.
    const double INV_2PI  = 1.5915494309189534561e-1;
    const double TWO_PI_A = 6.2831853071795862320e+0;
    const double TWO_PI_B = 2.4492935982947063780e-16;
    double ph_hi = t0 * logn;
    double ph_lo = fma(t0, logn, -ph_hi);
    double ang   = -ph_hi;
    double alo   = -ph_lo;
    double k2pi  = rint(ang * INV_2PI);
    double r     = fma(-k2pi, TWO_PI_A, ang) - k2pi * TWO_PI_B + alo
                 - t0 * logn_corr[n - 1];      // ln n error ~1e-15 × t0 ~ 3e-3 rad otherwise
    double a_re  = isqn * cos(r);
    double a_im  = isqn * sin(r);

    // Non-uniform grid position in [0, M·u_max]
    double xpos   = M * h * logn * (1.0 / (2.0 * M_PI));
    int    center = (int)xpos;
    double frac   = xpos - center;

    #pragma unroll
    for (int dw = -(NUFFT_W/2); dw <= NUFFT_W/2; dw++) {
        int m = (center + dw) & (M - 1);   // wrap (M is always power of 2)
        double w = kb_window_dev((double)dw - frac, beta, I0b_inv);
        atomicAdd(&grid_cx[m].x, a_re * w);
        atomicAdd(&grid_cx[m].y, a_im * w);
    }
}

// ─── NUFFT Phase 3: deconvolve + compute Z(t_j) ────────────────────────────
//
// G[j]/Φ̂[j] ≈ F(j) = Σ_n n^{-½-it_j}
// Z(t_j) = 2·Re[e^{iθ(t_j)}·F(j)] + C₀(t_j)
//
// Φ̂[j] (KB window DFT) computed on-the-fly — symmetric window → purely real.
// Precomputed phi_vals[0..W] passed as constant array for speed.

// Output j corresponds to frequency k = j − jc, i.e. t_j = tc + k·h.  Centring the
// band (|k| ≤ K/2 ≈ 0.19·M) keeps every output inside the accurate range of the
// KB window; the old uncentred layout (k = 0..K ≈ 0.37·M) lost accuracy toward the
// end of each chunk (|ΔZ| up to ~0.14 at k ≈ K).
//
// θ(t_j) = θ(tc) + θ'(tc)·τ + θ''(tc)·τ²/2,  τ = k·h, with θ(tc) mod 2π from long double
// on the host.  Evaluating θ(t) directly in double at t ~ 3e12 (θ ~ 4e13) costs ~1e-2 rad.
__global__ void nufft_finalize_kernel(
    const cufftDoubleComplex* __restrict__ G,
    const double*             __restrict__ phi_vals,   // [NUFFT_W+1] = φ(-W/2..W/2)
    int K, int M, int jc,
    double tc, double h,
    double th0, double th1, double th2,
    double* __restrict__ Z_out
) {
    int j = (int)(blockIdx.x*(unsigned)blockDim.x + threadIdx.x);
    if (j >= K) return;
    int k  = j - jc;
    int gi = (k < 0) ? k + M : k;

    // Compute Φ̂[k] = Σ_{dw=-W/2}^{W/2} φ(dw)·cos(2π·k·dw/M)
    double phi_hat = phi_vals[NUFFT_W/2];   // dw=0 contribution (cos(0)=1)
    double angle0  = 2.0 * M_PI * k / M;
    #pragma unroll
    for (int dw = 1; dw <= NUFFT_W/2; dw++) {
        double pv = phi_vals[NUFFT_W/2 + dw];  // φ(dw) = φ(-dw) by symmetry
        phi_hat  += 2.0 * pv * cos(angle0 * dw);
    }

    // Deconvolve: F(j) = G[j] / Φ̂[j]
    double Fr = (phi_hat > 1e-30) ? G[gi].x / phi_hat : 0.0;
    double Fi = (phi_hat > 1e-30) ? G[gi].y / phi_hat : 0.0;

    // Z(t_j) = 2·Re[e^{iθ}·F] + C₀
    double tau = k * h;
    double t   = tc + tau;
    double th  = th0 + tau * (th1 + 0.5 * th2 * tau);
    double Z  = 2.0 * (Fr * cos(th) - Fi * sin(th));

    // C₀ correction: (-1)^N * (t/2π)^{-1/4} * cos(2π(p²-p-1/16)) / cos(2πp)
    // where p = frac(√(t/2π)), N = floor(√(t/2π))
    {
        double s2pi = sqrt(t * (1.0 / (2.0 * M_PI)));
        int    Ni   = (int)s2pi;
        double p    = s2pi - Ni;
        double cd   = cos(2.0 * M_PI * p);
        if (fabs(cd) > 1e-6) {
            double cn = cos(2.0 * M_PI * (p*p - p - 1.0/16.0));
            double sg = (Ni & 1) ? 1.0 : -1.0;
            Z += sg * pow(t * (1.0 / (2.0 * M_PI)), -0.25) * cn / cd;
        }
    }

    Z_out[j] = Z;
}

// ─── Direct-sum kernel for diagnostics: F(0) = Σ n^{-½-it0} ──────────────

__global__ void direct_F0_kernel(
    double t0, int N_max,
    double* out_re, double* out_im)   // single-element device outputs
{
    // Each thread handles one n; reduction into shared memory.
    __shared__ double s_re[256], s_im[256];
    int tid = threadIdx.x;
    int n   = (int)(blockIdx.x*(unsigned)blockDim.x + threadIdx.x) + 1;

    double re = 0.0, im = 0.0;
    if (n <= N_max) {
        double logn = log((double)n);
        double isqn = rsqrt((double)n);
        double ph_hi = t0 * logn;
        double ph_lo = fma(t0, logn, -ph_hi);
        double ang = -ph_hi, alo = -ph_lo;
        const double INV_2PI = 1.5915494309189534561e-1;
        const double A = 6.2831853071795862320e+0, B = 2.4492935982947063780e-16;
        double k2pi = rint(ang * INV_2PI);
        double r = fma(-k2pi, A, ang) - k2pi * B + alo;
        re = isqn * cos(r);
        im = isqn * sin(r);
    }
    s_re[tid] = re;  s_im[tid] = im;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) { s_re[tid] += s_re[tid+s]; s_im[tid] += s_im[tid+s]; }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(out_re, s_re[0]);
        atomicAdd(out_im, s_im[0]);
    }
}

// ─── NUFFT GPU state ───────────────────────────────────────────────────────

static cufftHandle          nufft_plan     = 0;
static cufftDoubleComplex*  d_grid_cx      = nullptr;
static double*              d_Z_nufft      = nullptr;
static double*              d_phi_vals     = nullptr;   // [NUFFT_W+1]
static double*              d_logn_corr    = nullptr;   // [N_max]: log_true(n)-log_double(n)
static int                  nufft_M_cur    = 0;
static int                  nufft_K_cur    = 0;
static int                  nufft_N_cur    = 0;

static void ensure_nufft(int K, double h, int N_max) {
    // Need M ≥ 2K (σ≥2) so j/M ≤ 0.5 for all outputs — avoids alias domination.
    int M = 1;
    while (M < 2*K) M <<= 1;

    bool need_realloc = (M != nufft_M_cur);
    bool need_phi     = (K != nufft_K_cur) || need_realloc;

    if (need_realloc) {
        if (d_grid_cx) {
            cudaFree(d_grid_cx); cudaFree(d_Z_nufft);
            d_grid_cx = nullptr; d_Z_nufft = nullptr;
        }
        if (nufft_plan) { cufftDestroy(nufft_plan); nufft_plan = 0; }

        double mem_mb = (M*16. + K*8.) / 1e6;
        fprintf(stderr, "[NUFFT] alloc M=%d K=%d (%.1f MB)\n", M, K, mem_mb);

        cudaError_t ce;
        ce = cudaMalloc(&d_grid_cx, (size_t)M * sizeof(cufftDoubleComplex));
        if (ce != cudaSuccess) { fprintf(stderr, "[NUFFT] OOM d_grid_cx: %s\n", cudaGetErrorString(ce)); return; }
        ce = cudaMalloc(&d_Z_nufft, (size_t)K * sizeof(double));
        if (ce != cudaSuccess) { fprintf(stderr, "[NUFFT] OOM d_Z_nufft: %s\n", cudaGetErrorString(ce)); return; }

        cufftResult r = cufftPlan1d(&nufft_plan, M, CUFFT_Z2Z, 1);
        if (r != CUFFT_SUCCESS) {
            fprintf(stderr, "[NUFFT] cufftPlan1d failed: %d\n", (int)r);
            return;
        }

        nufft_M_cur = M;
    } else if (K > nufft_K_cur) {
        // K grew: realloc Z buffer only
        cudaFree(d_Z_nufft);
        cudaError_t ce = cudaMalloc(&d_Z_nufft, (size_t)K * sizeof(double));
        if (ce != cudaSuccess) { fprintf(stderr, "[NUFFT] OOM d_Z_nufft grow: %s\n", cudaGetErrorString(ce)); return; }
    }

    if (need_phi) {
        // Precompute KB window values φ(-W/2..W/2) on host, upload
        double I0b = bessel_I0_host(NUFFT_BETA);
        std::vector<double> phi(NUFFT_W + 1);
        for (int dw = -(NUFFT_W/2); dw <= NUFFT_W/2; dw++) {
            double r   = dw * (2.0 / NUFFT_W);
            double arg = 1.0 - r*r;
            int    idx = dw + NUFFT_W/2;
            phi[idx]   = (arg > 0) ? bessel_I0_host(NUFFT_BETA * sqrt(arg)) / I0b : 0.0;
        }
        if (!d_phi_vals) cudaMalloc(&d_phi_vals, (NUFFT_W + 1) * sizeof(double));
        cudaMemcpy(d_phi_vals, phi.data(), (NUFFT_W + 1) * sizeof(double),
                   cudaMemcpyHostToDevice);
        nufft_K_cur = K;
    }

    // Precompute log corrections: log_true(n) - log_double(n) using long double.
    // Eliminates ~0.005 Z-error from t*log(n) phase inaccuracy at t~3e12.
    // Allocate for N_LOGCORR_CAP (covers T_end=7.6e12) on first call and never
    // reallocate — repeated cudaFree/cudaMalloc fragments VRAM heap and causes OOM
    // at chunk ~5966 after ~400+ realloc cycles.
    static const int N_LOGCORR_CAP = 1100000;  // sqrt(7.6e12 / 2π) ≈ 1,098,000
    int N_alloc = std::max(N_max, N_LOGCORR_CAP);
    if (N_alloc > nufft_N_cur) {
        std::vector<double> corr(N_alloc);
        for (int n = 1; n <= N_alloc; n++) {
            long double ln_ld = logl((long double)n);
            double      ln_d  = log((double)n);
            corr[n-1] = (double)(ln_ld - (long double)ln_d);
        }
        if (d_logn_corr) cudaFree(d_logn_corr);
        cudaError_t ce = cudaMalloc(&d_logn_corr, (size_t)N_alloc * sizeof(double));
        if (ce != cudaSuccess) {
            fprintf(stderr, "[NUFFT] OOM d_logn_corr N=%d: %s\n", N_alloc, cudaGetErrorString(ce));
            return;
        }
        cudaMemcpy(d_logn_corr, corr.data(), (size_t)N_alloc * sizeof(double),
                   cudaMemcpyHostToDevice);
        nufft_N_cur = N_alloc;
        fprintf(stderr, "[NUFFT] log-corr table N=%d (%.1f MB)\n",
                N_alloc, N_alloc * 8.0 / 1e6);
    }
}

// Compute Z[0..K-1] on a uniform grid t_j = t0 + j*h via NUFFT.  Z stays on the GPU in
// d_Z_nufft; it is copied to Z_host (400 MB at K = 50M) only when Z_host != nullptr.
static void compute_Z_nufft(
    double t0, double h, int K, double* Z_host)
{
    if (K <= 0) return;

    int    N_max   = (int)sqrt(t0 / (2.0 * M_PI));  // RS floor at t_start (matches Python)
    ensure_nufft(K, h, N_max);

    // Centre of the output band.  The grid is t_j = tc + (j − jc)·h; tc is whatever
    // double t0 + jc·h rounds to, and is used consistently below.
    int    jc = K / 2;
    double tc = t0 + jc * h;
    // θ(tc) mod 2π and derivatives, in long double (64-bit mantissa).
    long double T   = tc;
    long double thL = (T/2)*logl(T/(2*M_PIl)) - T/2 - M_PIl/8
                    + 1/(48*T) + 7/(5760*T*T*T);
    long double twopi = 2*M_PIl;
    double th0 = (double)(thL - twopi * floorl(thL / twopi));
    double th1 = (double)(0.5L*logl(T/(2*M_PIl)) - 1/(48*T*T));
    double th2 = (double)(1/(2*T));

    int    M       = nufft_M_cur;
    double I0b_inv = 1.0 / bessel_I0_host(NUFFT_BETA);

    // Zero the complex grid
    cudaMemset(d_grid_cx, 0, (size_t)M * sizeof(cufftDoubleComplex));

    // Phase 1: spread N_max sources directly into complex grid
    {
        int blk = THREADS;
        int grd = (N_max + blk - 1) / blk;
        nufft_spread_kernel<<<grd, blk>>>(
            tc, h, N_max, M, NUFFT_BETA, I0b_inv,
            d_logn_corr, d_grid_cx);
    }

    // Phase 2: FFT
    cufftExecZ2Z(nufft_plan, d_grid_cx, d_grid_cx, CUFFT_FORWARD);

    // Phase 3: deconvolve + compute Z values
    {
        int blk = THREADS, grd = (K + blk - 1) / blk;
        nufft_finalize_kernel<<<grd, blk>>>(
            d_grid_cx, d_phi_vals, K, M, jc, tc, h, th0, th1, th2, d_Z_nufft);
    }

    cudaDeviceSynchronize();
    if (Z_host)
        cudaMemcpy(Z_host, d_Z_nufft, (size_t)K * sizeof(double),
                   cudaMemcpyDeviceToHost);

    // RH_DUMP=<file>: write tc, jc, h and Z at ~200 indices for validation (rh_check.py)
    if (const char* dump = getenv("RH_DUMP")) {
        if (FILE* f = fopen(dump, "w")) {
            fprintf(f, "%.17g %d %.17g\n", tc, jc, h);
            for (int q = 0; q < 200; q++) {
                int j = (int)((long long)q * (K - 1) / 199);
                double zj;
                cudaMemcpy(&zj, d_Z_nufft + j, sizeof(double), cudaMemcpyDeviceToHost);
                fprintf(f, "%d %.17g\n", j, zj);
            }
            fclose(f);
        }
    }
}

// ─── sign changes ──────────────────────────────────────────────────────────

// Raw sign changes: indices i with Z[i]·Z[i+1] < 0 or Z[i] == 0, plus each zero's
// offset from the chunk start by linear interpolation (see rh_stats.h).
//
// GPU path: count → CUB DeviceSelect (ordered stream compaction) → interpolate, then
// copy back only ~K/115 indices and offsets (~5 MB) instead of all of Z (400 MB).

struct SignChangePred {
    const double* Z;
    __device__ bool operator()(int i) const {
        double a = Z[i], b = Z[i + 1];
        return a * b < 0.0 || a == 0.0;
    }
};

__global__ void sign_count_kernel(const double* __restrict__ Z, int n_items,
                                  unsigned long long* __restrict__ count) {
    int i = (int)(blockIdx.x * (unsigned)blockDim.x + threadIdx.x);
    bool f = false;
    if (i < n_items) { double a = Z[i], b = Z[i + 1]; f = (a * b < 0.0 || a == 0.0); }
    int c = __syncthreads_count(f);
    if (threadIdx.x == 0 && c) atomicAdd(count, (unsigned long long)c);
}

__global__ void interp_offset_kernel(const double* __restrict__ Z,
                                     const int* __restrict__ idx, int n, double h,
                                     double* __restrict__ off) {
    int k = (int)(blockIdx.x * (unsigned)blockDim.x + threadIdx.x);
    if (k >= n) return;
    int i = idx[k];
    double a = Z[i], b = Z[i + 1];
    double f = (a == b) ? 0.5 : a / (a - b);
    if (f < 0.0) f = 0.0; else if (f > 1.0) f = 1.0;
    off[k] = (i + f) * h;          // same formula as interp_zero_offset()
}

static int*                d_sc_idx  = nullptr;
static double*             d_sc_off  = nullptr;
static int*                d_sc_nsel = nullptr;
static unsigned long long* d_sc_cnt  = nullptr;
static void*               d_cub_tmp = nullptr;
static size_t              cub_tmp_bytes = 0;
static int                 sc_cap    = 0;

// Returns false (caller falls back to the host path) if the count exceeds the
// preallocated capacity or a CUDA call fails.
static bool gpu_sign_changes(int K, double h, std::vector<int>& raw, std::vector<double>& off) {
    const int n_items = K - 1;
    if (!d_sc_cnt) {
        // ~430k zeros per 50M-point chunk; K/16 leaves >7x headroom for noisy chunks.
        sc_cap = K / 16;
        if (cudaMalloc(&d_sc_idx,  (size_t)sc_cap * sizeof(int))    != cudaSuccess ||
            cudaMalloc(&d_sc_off,  (size_t)sc_cap * sizeof(double)) != cudaSuccess ||
            cudaMalloc(&d_sc_nsel, sizeof(int))                     != cudaSuccess ||
            cudaMalloc(&d_sc_cnt,  sizeof(unsigned long long))      != cudaSuccess) {
            fprintf(stderr, "[signs] OOM allocating compaction buffers\n");
            d_sc_cnt = nullptr; return false;
        }
        cub::DeviceSelect::If(nullptr, cub_tmp_bytes, thrust::counting_iterator<int>(0),
                              d_sc_idx, d_sc_nsel, n_items, SignChangePred{d_Z_nufft});
        if (cudaMalloc(&d_cub_tmp, cub_tmp_bytes) != cudaSuccess) {
            fprintf(stderr, "[signs] OOM allocating CUB temp (%zu B)\n", cub_tmp_bytes);
            return false;
        }
        fprintf(stderr, "[signs] GPU compaction cap=%d tmp=%.1f MB\n", sc_cap, cub_tmp_bytes / 1e6);
    }
    if (n_items <= 0) { raw.clear(); off.clear(); return true; }

    // 1. count, so the compaction can never overrun the buffers
    cudaMemset(d_sc_cnt, 0, sizeof(unsigned long long));
    sign_count_kernel<<<(n_items + THREADS - 1) / THREADS, THREADS>>>(d_Z_nufft, n_items, d_sc_cnt);
    unsigned long long cnt = 0;
    cudaMemcpy(&cnt, d_sc_cnt, sizeof(cnt), cudaMemcpyDeviceToHost);
    if (cnt > (unsigned long long)sc_cap) {
        fprintf(stderr, "[signs] %llu sign changes > cap %d, host fallback\n", cnt, sc_cap);
        return false;
    }

    // 2. ordered compaction of indices, 3. interpolated offsets
    size_t tmp = cub_tmp_bytes;
    cub::DeviceSelect::If(d_cub_tmp, tmp, thrust::counting_iterator<int>(0),
                          d_sc_idx, d_sc_nsel, n_items, SignChangePred{d_Z_nufft});
    int n = (int)cnt;
    if (n > 0)
        interp_offset_kernel<<<(n + THREADS - 1) / THREADS, THREADS>>>(d_Z_nufft, d_sc_idx, n, h, d_sc_off);

    raw.resize(n); off.resize(n);
    if (n > 0) {
        cudaMemcpy(raw.data(), d_sc_idx, (size_t)n * sizeof(int),    cudaMemcpyDeviceToHost);
        cudaMemcpy(off.data(), d_sc_off, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost);
    }
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { fprintf(stderr, "[signs] CUDA error: %s\n", cudaGetErrorString(e)); return false; }
    return true;
}

static void host_sign_changes(const double* Z, int n, double h,
                              std::vector<int>& raw, std::vector<double>& off) {
    raw.clear(); off.clear();
    raw.reserve(n / 64); off.reserve(n / 64);
    for (int i = 0; i < n - 1; i++)
        if (Z[i] * Z[i+1] < 0.0 || Z[i] == 0.0) {
            raw.push_back(i);
            off.push_back(interp_zero_offset(Z, i, h));
        }
}

// Returns positions (into raw) of the sign changes kept after dip filtering.
// dropped_at (optional) receives the raw positions k of each dropped pair (k, k+1).
static std::vector<size_t> filter_dips(const std::vector<int>& raw,
                                       long long* dropped = nullptr,
                                       std::vector<size_t>* dropped_at = nullptr) {
    // Remove narrow sign-change pairs (≤ GAP_MAX grid steps apart).  Since the Nov 2026
    // accuracy fixes these are ~4 per 50M-point chunk and are mostly GENUINE close zero
    // pairs (the best Lehmer candidates) plus occasional near-misses where |Z| < ~2e-3;
    // their positions are recorded as "dip_pairs" so rh_pairs.py can tell which is which.
    // Original rationale (pre-fix, when they were ~5,700 per chunk of numerical noise):
    // A near-miss Z minimum (Z_true ≈ |ε| ≈ 0.005) at t_j can push Z_NUFFT
    // below zero for 1-2 grid steps, creating sign changes at (j-1,j) and
    // (j+k-1, j+k) for k=1 (1-step) or k=2 (2-step dip).  These appear in
    // the raw index array as pairs with gap ≤ GAP_MAX.
    //
    // Genuine zeros are spaced ~233 ms apart (mean) vs h=2 ms, so consecutive
    // or near-consecutive sign changes are overwhelmingly false.  Close genuine
    // pairs (d < GAP_MAX*h ≈ 0.004s) are rare: ~10 per 5000s interval.
    //
    // GAP_MAX=2 catches both 1-point and 2-point false dips while losing ~10
    // genuine close-pair zeros per 5000s interval — a net of ~+600 → ~+10.
    static const int GAP_MAX = 2;

    std::vector<size_t> out;
    out.reserve(raw.size());
    long long nd = 0;
    for (size_t k = 0; k < raw.size(); ) {
        if (k + 1 < raw.size() && raw[k+1] - raw[k] <= GAP_MAX) {
            if (dropped_at) dropped_at->push_back(k);
            k += 2;  // skip pair
            nd++;
        } else {
            out.push_back(k);
            ++k;
        }
    }
    if (dropped) *dropped = nd;
    return out;
}

// ─── timing ────────────────────────────────────────────────────────────────

static double now_s() {
    struct timeval tv; gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec*1e-6;
}

// ─── minimal JSON helpers ──────────────────────────────────────────────────

static std::string jstr(const std::string& j, const std::string& k) {
    size_t p = j.find("\""+k+"\"");
    if (p == std::string::npos) return "";
    p = j.find(':', p)+1;
    while (p < j.size() && (j[p]==' '||j[p]=='\t')) p++;
    if (j[p]=='"') { p++; size_t e=j.find('"',p); return j.substr(p,e-p); }
    size_t e=j.find_first_of(",}\n",p);
    return j.substr(p,e-p);
}
static double jdbl(const std::string& j, const std::string& k) {
    auto s=jstr(j,k); return s.empty()?0.0:std::stod(s);
}

// ─── Thread-safe bounded pipeline queue ───────────────────────────────────

template<typename T>
class Pipeline {
    std::queue<T>           q_;
    std::mutex              m_;
    std::condition_variable cv_;
    bool                    closed_ = false;
    const size_t            cap_;
public:
    explicit Pipeline(size_t cap = ~size_t(0)) : cap_(cap) {}
    void push(T v) {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [&]{ return q_.size() < cap_ || closed_; });
        if (closed_) return;
        q_.push(std::move(v));
        cv_.notify_all();
    }
    // Returns false when closed and empty (no more items ever)
    bool pop(T& v) {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [&]{ return !q_.empty() || closed_; });
        if (q_.empty()) return false;
        v = std::move(q_.front()); q_.pop();
        cv_.notify_all();
        return true;
    }
    void close() { std::unique_lock<std::mutex> lk(m_); closed_ = true; cv_.notify_all(); }
};

// ─── main ──────────────────────────────────────────────────────────────────

struct ChunkResult {
    std::string chunk_id;
    char        host[256];
    double      t_start, t_end, dt;
    long long   zeros_found, backlund_expected;
    bool        backlund_ok;
    double      wall_seconds;
    std::vector<double> zero_positions;  // only populated on Backlund mismatch
    std::string out_dir;                 // where the per-chunk JSON goes
    std::string active_dir;              // where the coordinator's in-flight marker lives
    bool        want_stats = false;
    ZeroStats   stats;
};

// Phase 1: GPU compute + sign counting + Backlund check.  No I/O — pure compute.
static ChunkResult compute_chunk(const std::string& job) {
    std::string chunk_id = jstr(job,"chunk_id");
    double t_start = jdbl(job,"t_start");
    double t_end   = jdbl(job,"t_end");
    double dt_cfg  = jdbl(job,"dt");
    if (dt_cfg <= 0.0) dt_cfg = 0.05;

    char host[256]="unknown"; gethostname(host,sizeof(host));
    fprintf(stderr,"[%s] %s  NUFFT  t=[%.4e,%.4e]  h=%.4f\n",
            host, chunk_id.c_str(), t_start, t_end, NUFFT_H);

    double t_wall = now_s();

    // Build fine t-grid at NUFFT_H spacing
    double h   = NUFFT_H;
    int    K   = (int)((t_end - t_start) / h) + 2;
    // Clamp: 52M points → M=128M, ~4.7GB GPU (RTX 5070 Ti 16GB safe)
    if (K > 52000000) K = 52000000;
    double t0  = t_start;

    // The full Z array is only needed on the host for the RH_DIAG diagnostics.
    const bool diag = getenv("RH_DIAG") != nullptr;
    std::vector<double> Zg;
    if (diag) Zg.resize(K);

    double t0scan = now_s();
    compute_Z_nufft(t0, h, K, diag ? Zg.data() : nullptr);
    double scan_s = now_s() - t0scan;
    fprintf(stderr,"  scan %.3fs  K=%d  pts/s=%.0f\n",
            scan_s, K, K/(scan_s+1e-9));

    // Diagnostic (legacy): compares against a double-precision direct sum indexed from t0.
    // Since the output band is now centred on tc = t0 + (K/2)·h, the F(0) line refers to
    // the uncentred layout and is not meaningful; use RH_DUMP + rh_check.py (mpmath) instead.
    if (getenv("RH_DIAG")) {
        int N_max_d = (int)sqrt(t0 / (2.0 * M_PI)) + 2;
        // Direct sum F(0) on GPU
        double h_re = 0.0, h_im = 0.0;
        double *d_re, *d_im;
        cudaMalloc(&d_re, sizeof(double)); cudaMalloc(&d_im, sizeof(double));
        cudaMemcpy(d_re, &h_re, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_im, &h_im, sizeof(double), cudaMemcpyHostToDevice);
        int blk = 256, grd = (N_max_d + blk - 1) / blk;
        direct_F0_kernel<<<grd, blk>>>(t0, N_max_d, d_re, d_im);
        cudaDeviceSynchronize();
        cudaMemcpy(&h_re, d_re, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_im, d_im, sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(d_re); cudaFree(d_im);
        // Get NUFFT G[0] from FFT output
        cufftDoubleComplex G0;
        cudaMemcpy(&G0, d_grid_cx, sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
        // Phi_hat at j=0
        std::vector<double> phi_h(NUFFT_W + 1);
        cudaMemcpy(phi_h.data(), d_phi_vals, (NUFFT_W+1)*sizeof(double), cudaMemcpyDeviceToHost);
        double phi_hat0 = phi_h[NUFFT_W/2];
        for (int dw=1; dw<=NUFFT_W/2; dw++) phi_hat0 += 2.0*phi_h[NUFFT_W/2+dw];
        double F0_nufft_re = G0.x / phi_hat0, F0_nufft_im = G0.y / phi_hat0;
        fprintf(stderr, "  [DIAG] F(0): direct_gpu=(%.6f,%.6f) nufft=(%.6f,%.6f) phi_hat0=%.6f\n",
                h_re, h_im, F0_nufft_re, F0_nufft_im, phi_hat0);
        fprintf(stderr, "  [DIAG] F(0) error: re_diff=%.6f im_diff=%.6f\n",
                F0_nufft_re - h_re, F0_nufft_im - h_im);

        int N_max_diag = N_max_d;
        fprintf(stderr, "  [DIAG] comparing first 200 pts, N=%d\n", N_max_diag);
        int n_bad = 0;
        for (int j = 0; j < 200 && j < K; j++) {
            double t = t0 + j * h;
            int    N = (int)sqrt(t / (2.0 * M_PI));
            double th = theta_host(t);
            // direct RS
            double sum_re = 0, sum_im = 0;
            for (int n = 1; n <= N; n++) {
                double ln = log((double)n);
                double ph = t * ln;
                double ph_hi = ph, ph_lo = fma(t, ln, -ph_hi);
                double ang = -ph_hi, alo = -ph_lo;
                const double INV_2PI = 1.5915494309189534561e-1;
                const double A = 6.2831853071795862320e+0, B = 2.4492935982947063780e-16;
                double kk = rint(ang * INV_2PI);
                double r2 = fma(-kk, A, ang) - kk * B + alo;
                sum_re += cos(r2) / sqrt((double)n);
                sum_im += sin(r2) / sqrt((double)n);
            }
            double Z_rs = 2.0 * (sum_re * cos(th) - sum_im * sin(th));
            // RS remainder C0
            double s2pi = sqrt(t / (2.0 * M_PI));
            int    Ni   = (int)s2pi;
            double p    = s2pi - Ni;
            double cd   = cos(2.0 * M_PI * p);
            if (fabs(cd) > 1e-6) {
                double cn = cos(2.0 * M_PI * (p*p - p - 1./16.));
                double sg = (Ni & 1) ? 1.0 : -1.0;
                Z_rs += sg * pow(t / (2.0 * M_PI), -0.25) * cn / cd;
            }
            double diff = Zg[j] - Z_rs;
            if (fabs(diff) > 0.1) {
                fprintf(stderr, "  j=%d t=%.6f Z_nufft=%.6f Z_rs=%.6f diff=%.6f\n",
                        j, t, Zg[j], Z_rs, diff);
                n_bad++;
                if (n_bad >= 10) break;
            }
        }
        if (n_bad == 0) fprintf(stderr, "  [DIAG] first 200 pts: max diff < 0.1 ✓\n");

        // Check LAST 200 pts (high j) vs direct RS
        fprintf(stderr, "  [DIAG] comparing last 200 pts\n");
        int n_bad2 = 0;
        double max_diff2 = 0.0;
        for (int jj = K-200; jj < K; jj++) {
            double t = t0 + jj * h;
            int    N = (int)sqrt(t / (2.0 * M_PI));
            double th = theta_host(t);
            double sum_re = 0, sum_im = 0;
            for (int n = 1; n <= N; n++) {
                double ln = log((double)n);
                double ph_hi = t * ln;
                double ph_lo = fma(t, ln, -ph_hi);
                double ang = -ph_hi, alo = -ph_lo;
                const double INV_2PI = 1.5915494309189534561e-1;
                const double A = 6.2831853071795862320e+0, B = 2.4492935982947063780e-16;
                double kk = rint(ang * INV_2PI);
                double r2 = fma(-kk, A, ang) - kk * B + alo;
                sum_re += cos(r2) / sqrt((double)n);
                sum_im += sin(r2) / sqrt((double)n);
            }
            double Z_rs = 2.0 * (sum_re * cos(th) - sum_im * sin(th));
            double s2pi = sqrt(t / (2.0 * M_PI));
            int    Ni   = (int)s2pi;
            double p    = s2pi - Ni;
            double cd   = cos(2.0 * M_PI * p);
            if (fabs(cd) > 1e-6) {
                double cn = cos(2.0 * M_PI * (p*p - p - 1./16.));
                double sg = (Ni & 1) ? 1.0 : -1.0;
                Z_rs += sg * pow(t / (2.0 * M_PI), -0.25) * cn / cd;
            }
            double diff = fabs(Zg[jj] - Z_rs);
            if (diff > max_diff2) max_diff2 = diff;
            if (diff > 0.1) {
                if (n_bad2 < 5) fprintf(stderr, "  j=%d t=%.6f Z_nufft=%.6f Z_rs=%.6f diff=%.6f\n",
                        jj, t, Zg[jj], Z_rs, diff);
                n_bad2++;
            }
        }
        fprintf(stderr, "  [DIAG] last 200 pts: n_bad=%d max_diff=%.6f\n", n_bad2, max_diff2);

        // Count zeros in each half
        int half = K / 2;
        int z0 = 0, z1 = 0;
        for (int i = 0; i < half-1; i++) if (Zg[i]*Zg[i+1] < 0.0) z0++;
        for (int i = half; i < K-1; i++) if (Zg[i]*Zg[i+1] < 0.0) z1++;
        long long ne0 = backlund_count(t0 + half*h) - backlund_count(t0);
        long long ne1 = backlund_count(t0 + (K-1)*h) - backlund_count(t0 + half*h);
        fprintf(stderr, "  [DIAG] half0: found=%d expected=%lld | half1: found=%d expected=%lld\n",
                z0, ne0, z1, ne1);

        // Count zeros in first 500 pts via DIRECT RS (definitive: does RS agree with Backlund?)
        {
            int npts = 500;
            std::vector<double> Z_rs_vec(npts);
            for (int j = 0; j < npts; j++) {
                double t = t0 + j * h;
                int    N = (int)sqrt(t / (2.0 * M_PI));
                double th = theta_host(t);
                double sum_re = 0, sum_im = 0;
                for (int n = 1; n <= N; n++) {
                    double ln = log((double)n);
                    double ph_hi = t * ln;
                    double ph_lo = fma(t, ln, -ph_hi);
                    double ang = -ph_hi, alo = -ph_lo;
                    const double INV_2PI = 1.5915494309189534561e-1;
                    const double A = 6.2831853071795862320e+0, B = 2.4492935982947063780e-16;
                    double kk = rint(ang * INV_2PI);
                    double r2 = fma(-kk, A, ang) - kk * B + alo;
                    sum_re += cos(r2) / sqrt((double)n);
                    sum_im += sin(r2) / sqrt((double)n);
                }
                double Z_rs = 2.0 * (sum_re * cos(th) - sum_im * sin(th));
                double s2pi = sqrt(t / (2.0 * M_PI));
                int    Ni   = (int)s2pi;
                double p    = s2pi - Ni;
                double cd   = cos(2.0 * M_PI * p);
                if (fabs(cd) > 1e-6) {
                    double cn = cos(2.0 * M_PI * (p*p - p - 1./16.));
                    double sg = (Ni & 1) ? 1.0 : -1.0;
                    Z_rs += sg * pow(t / (2.0 * M_PI), -0.25) * cn / cd;
                }
                Z_rs_vec[j] = Z_rs;
            }
            int nufft_sc = 0, rs_sc = 0;
            for (int i = 0; i < npts-1; i++) {
                if (Zg[i]*Zg[i+1] < 0.0) nufft_sc++;
                if (Z_rs_vec[i]*Z_rs_vec[i+1] < 0.0) rs_sc++;
            }
            long long ne_5k = backlund_count(t0 + npts*h) - backlund_count(t0);
            fprintf(stderr, "  [DIAG] first5k: nufft_sc=%d rs_sc=%d backlund=%lld\n",
                    nufft_sc, rs_sc, ne_5k);
        }
    }

    // Count sign changes (positions only needed on Backlund mismatch)
    std::vector<int>    raw;
    std::vector<double> raw_off;
    // RH_HOST_SIGNS=1 forces the host path (for testing the fallback).
    if (getenv("RH_HOST_SIGNS") || !gpu_sign_changes(K, h, raw, raw_off)) {
        if (Zg.empty()) {
            Zg.resize(K);
            cudaMemcpy(Zg.data(), d_Z_nufft, (size_t)K * sizeof(double), cudaMemcpyDeviceToHost);
        }
        host_sign_changes(Zg.data(), K, h, raw, raw_off);
    }
    long long n_dips = 0;
    std::vector<size_t> dip_at;
    std::vector<size_t> keep = filter_dips(raw, &n_dips, &dip_at);
    std::vector<int>    idxs(keep.size());
    std::vector<double> offs(keep.size());
    for (size_t k = 0; k < keep.size(); k++) { idxs[k] = raw[keep[k]]; offs[k] = raw_off[keep[k]]; }
    long long nf = (long long)idxs.size();

    fprintf(stderr,"  zeros=%lld  (pos acc=%.4f)\n", nf, h/2.0);

    // Backlund check
    long long ne  = backlund_count(t_end) - backlund_count(t_start);
    long long tol = std::max(50LL, (long long)(ne * 0.001));
    bool      ok  = (llabs(nf-ne) <= tol);
    fprintf(stderr,"  Backlund: found=%lld expected=%lld  %s\n",
            nf, ne, ok?"ALL_ON_CRITICAL_LINE":"MISMATCH");


    ChunkResult r;
    r.chunk_id          = chunk_id;
    strncpy(r.host, host, sizeof(r.host)-1); r.host[sizeof(r.host)-1] = '\0';
    r.t_start           = t_start;
    r.t_end             = t_end;
    r.dt                = h;
    r.zeros_found       = nf;
    r.backlund_expected = ne;
    r.backlund_ok       = ok;
    r.out_dir           = jstr(job, "out_dir");
    if (r.out_dir.empty()) r.out_dir = "/mnt/math/rh/done";
    r.active_dir        = jstr(job, "active_dir");
    if (r.active_dir.empty()) r.active_dir = "/mnt/math/rh/active";
    r.want_stats        = (jstr(job, "stats") == "true");
    if (r.want_stats) {
        double ct = jdbl(job, "close_thresh");
        if (ct <= 0.0) ct = 0.05;
        r.stats = compute_zero_stats_offsets(offs, K, t0, h, n_dips, ct);
        const double rho = zero_density(t0 + 0.5 * K * h);
        for (size_t k : dip_at)
            r.stats.dip_pairs.push_back({t0 + raw_off[k], t0 + raw_off[k + 1],
                                         (raw_off[k + 1] - raw_off[k]) * rho});
    }
    r.wall_seconds      = now_s() - t_wall;
    if (!ok) {
        r.zero_positions.resize(idxs.size());
        for (size_t i = 0; i < idxs.size(); i++)
            r.zero_positions[i] = t0 + (idxs[i] + 0.5) * h;
    }
    return r;
}

// Phase 2: write result to NAS done/ and remove active/ marker.
// Called after all GPU work in a batch is complete so NAS I/O never stalls the GPU.
static void write_chunk_result(const ChunkResult& r) {
    // Write to <id>.json.tmp, then rename: readers (coordinator, rh_stats.py) only look
    // at *.json, so they never see a half-written file.
    char nas_path[512], tmp_path[520];
    snprintf(nas_path, sizeof(nas_path), "%s/%s.json", r.out_dir.c_str(), r.chunk_id.c_str());
    snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", nas_path);
    FILE* fnas = fopen(tmp_path, "w");
    if (fnas) {
        fprintf(fnas,"{\n");
        fprintf(fnas,"  \"chunk_id\": \"%s\",\n",  r.chunk_id.c_str());
        fprintf(fnas,"  \"host\": \"%s\",\n",       r.host);
        fprintf(fnas,"  \"t_start\": %.15e,\n",     r.t_start);
        fprintf(fnas,"  \"t_end\": %.15e,\n",       r.t_end);
        fprintf(fnas,"  \"dt\": %.15g,\n",          r.dt);
        fprintf(fnas,"  \"zeros_found\": %lld,\n",  r.zeros_found);
        fprintf(fnas,"  \"backlund_expected\": %lld,\n", r.backlund_expected);
        fprintf(fnas,"  \"backlund_ok\": %s,\n",    r.backlund_ok?"true":"false");
        fprintf(fnas,"  \"wall_seconds\": %.3f,\n", r.wall_seconds);
        fprintf(fnas,"  \"gpu_used\": true");
        if (r.want_stats) {
            const ZeroStats& s = r.stats;
            fprintf(fnas,",\n  \"stats\": {\"sp_bin\": %g, \"pc_bin\": %g, "
                         "\"n_spacings\": %lld, \"spacing_overflow\": %lld, "
                         "\"pc_base\": %lld, \"dips_dropped\": %lld,\n",
                    SP_BIN, PC_BIN, s.n_spacings, s.spacing_overflow, s.pc_base, s.dips_dropped);
            fprintf(fnas,"    \"spacing_hist\": [");
            for (int b = 0; b < SP_BINS; b++) fprintf(fnas,"%s%lld", b?",":"", s.spacing_hist[b]);
            fprintf(fnas,"],\n    \"pc_hist\": [");
            for (int b = 0; b < PC_BINS; b++) fprintf(fnas,"%s%lld", b?",":"", s.pc_hist[b]);
            fprintf(fnas,"],\n    \"close_pairs\": [");
            for (size_t i = 0; i < s.close_pairs.size(); i++)
                fprintf(fnas,"%s[%.4f,%.4f,%.6f]", i?",":"",
                        s.close_pairs[i].ta, s.close_pairs[i].tb, s.close_pairs[i].gap);
            fprintf(fnas,"],\n    \"dip_pairs\": [");
            for (size_t i = 0; i < s.dip_pairs.size(); i++)
                fprintf(fnas,"%s[%.4f,%.4f,%.6f]", i?",":"",
                        s.dip_pairs[i].ta, s.dip_pairs[i].tb, s.dip_pairs[i].gap);
            fprintf(fnas,"]}");
        }
        if (!r.backlund_ok) {
            fprintf(fnas,",\n  \"zeros\": [\n");
            for (size_t i = 0; i < r.zero_positions.size(); i++)
                fprintf(fnas,"    %.6f%s\n", r.zero_positions[i],
                        i+1 < r.zero_positions.size() ? "," : "");
            fprintf(fnas,"  ]\n}\n");
        } else {
            fprintf(fnas,"\n}\n");
        }
        bool ok_w = (fclose(fnas) == 0);
        if (!ok_w || rename(tmp_path, nas_path) != 0) {
            fprintf(stderr, "[write] failed to finalise %s\n", nas_path);
            remove(tmp_path);
            return;            // keep the active marker: the chunk gets reclaimed
        }
        char act[512];
        snprintf(act, sizeof(act), "%s/%s.json", r.active_dir.c_str(), r.chunk_id.c_str());
        remove(act);
    }
}

// Build stdout result JSON from a ChunkResult.
static std::string result_to_json(const ChunkResult& r) {
    char buf[512];
    snprintf(buf, sizeof(buf),
             "{\"chunk_id\":\"%s\",\"zeros_found\":%lld,\"backlund_expected\":%lld,"
             "\"backlund_ok\":%s,\"wall_seconds\":%.3f,\"gpu_used\":true,"
             "\"host\":\"%s\",\"t_start\":%.15e,\"t_end\":%.15e,\"dt\":%.15g}",
             r.chunk_id.c_str(), r.zeros_found, r.backlund_expected,
             r.backlund_ok?"true":"false", r.wall_seconds,
             r.host, r.t_start, r.t_end, r.dt);
    return std::string(buf);
}

// Compatibility wrapper for single-chunk path.
static std::string process_chunk(const std::string& job) {
    ChunkResult r = compute_chunk(job);
    write_chunk_result(r);
    return result_to_json(r);
}

// Split a JSON array string "[{...},{...},...]" into individual object strings
static std::vector<std::string> split_json_array(const std::string& arr) {
    std::vector<std::string> out;
    int depth = 0;
    size_t start = std::string::npos;
    for (size_t i = 0; i < arr.size(); i++) {
        if (arr[i] == '{') {
            if (depth == 0) start = i;
            depth++;
        } else if (arr[i] == '}') {
            depth--;
            if (depth == 0 && start != std::string::npos) {
                out.push_back(arr.substr(start, i - start + 1));
                start = std::string::npos;
            }
        }
    }
    return out;
}

static void process_batch(const std::string& line) {
    // Extract the "chunks" array value
    size_t p = line.find("\"chunks\"");
    if (p == std::string::npos) { printf("{\"error\":\"no chunks key\"}\n"); fflush(stdout); return; }
    p = line.find('[', p);
    if (p == std::string::npos) { printf("{\"error\":\"no array\"}\n"); fflush(stdout); return; }
    // Find matching close bracket
    int depth = 0; size_t end = p;
    for (size_t i = p; i < line.size(); i++) {
        if (line[i]=='[') depth++;
        else if (line[i]==']') { depth--; if (depth==0){end=i;break;} }
    }
    auto chunks = split_json_array(line.substr(p, end-p+1));

    // Phase 1: all GPU compute — no I/O between chunks so GPU runs continuously
    std::vector<ChunkResult> results;
    results.reserve(chunks.size());
    for (size_t i = 0; i < chunks.size(); i++)
        results.push_back(compute_chunk(chunks[i]));

    // Phase 2: NAS writes — GPU is now free; other slots are computing during this
    for (size_t i = 0; i < results.size(); i++)
        write_chunk_result(results[i]);

    // Report all results to coordinator
    printf("[");
    for (size_t i = 0; i < results.size(); i++) {
        std::string j = result_to_json(results[i]);
        printf("%s%s", j.c_str(), i+1<results.size()?",":"");
    }
    printf("]\n");
    fflush(stdout);
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--version") {
        printf("rh_worker_cuda (NUFFT, centred band, stats mode)\n");
        return 0;
    }
    // Pre-allocate GPU buffers and warm up cuFFT JIT.
    {
        int K_w = 50000002;
        int N_w = (int)sqrt(3e12 / (2.0 * M_PI));
        ensure_nufft(K_w, NUFFT_H, N_w);
        if (!d_grid_cx) {
            fprintf(stderr, "[startup] VRAM allocation failed — aborting\n");
            return 1;
        }
        cudaMemset(d_grid_cx, 0, (size_t)nufft_M_cur * sizeof(cufftDoubleComplex));
        cufftExecZ2Z(nufft_plan, d_grid_cx, d_grid_cx, CUFFT_FORWARD);
        cudaDeviceSynchronize();
        fprintf(stderr, "[startup] warm-up done M=%d K=%d\n", nufft_M_cur, K_w);
    }

    // 3-thread pipeline: stdin_reader → GPU (main thread) → nas_writer
    //
    // GPU thread never touches disk or blocks on I/O — it pops a chunk,
    // computes, pushes the result, and immediately starts the next chunk.
    // NAS writes and stdout happen in nas_writer concurrently with GPU work.
    // stdin_reader pre-buffers up to 4 chunks so the GPU never waits for the
    // coordinator to send the next one.

    Pipeline<std::string> work_q(4);   // individual chunk JSONs pre-buffered for GPU
    Pipeline<ChunkResult> done_q;      // computed results waiting for NAS write + stdout

    // Thread 1: read stdin → work_q
    // Explodes legacy batch {"chunks":[...]} into individual chunk JSONs.
    std::thread reader([&]() {
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty()) continue;
            if (line.find("\"chunks\"") != std::string::npos) {
                size_t p = line.find('[', line.find("\"chunks\""));
                if (p == std::string::npos) continue;
                int depth = 0; size_t end = p;
                for (size_t i = p; i < line.size(); i++) {
                    if (line[i]=='[') depth++;
                    else if (line[i]==']') { depth--; if (!depth){end=i;break;} }
                }
                for (auto& c : split_json_array(line.substr(p, end-p+1)))
                    work_q.push(c);
            } else {
                work_q.push(line);
            }
        }
        work_q.close();
    });

    // Thread 3: done_q → NAS write + stdout (runs while GPU is already on next chunk)
    std::thread writer([&]() {
        ChunkResult r;
        while (done_q.pop(r)) {
            write_chunk_result(r);
            printf("%s\n", result_to_json(r).c_str());
            fflush(stdout);
        }
    });

    // Main thread (GPU): pop chunk → compute → push result → repeat, never stops
    {
        std::string job;
        while (work_q.pop(job)) {
            ChunkResult r = compute_chunk(job);
            done_q.push(std::move(r));
        }
    }
    done_q.close();

    reader.join();
    writer.join();
    return 0;
}
