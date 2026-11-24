// rh_stats.h — zero statistics computed per chunk from the sampled Z(t) grid.
//
// Pure host C++ (no CUDA) so it can be unit-tested with g++ (see test_stats.cpp).
//
// For each chunk we produce:
//   * spacing histogram  of unfolded nearest-neighbour gaps  δ = (γ₂−γ₁)·ρ(γ̄)
//   * pair-correlation histogram of unfolded differences γ_j − γ_i  (j > i) < PC_MAX
//   * close pairs: consecutive zeros with unfolded gap < close_thresh
//     (Lehmer-pair candidates, to be confirmed with high precision by rh_pairs.py)
//
// ρ(t) = log(t/2π)/(2π) is the mean zero density, so unfolded gaps have mean 1.
// Under the GUE / Montgomery conjecture the spacing density is the Gaudin
// distribution and the pair correlation is 1 − (sin πx / πx)².

#pragma once
#include <cmath>
#include <vector>

#define SP_BIN   0.02     // spacing histogram bin width (unfolded units)
#define SP_BINS  200      // covers [0, 4)
#define PC_BIN   0.02     // pair-correlation bin width
#define PC_BINS  150      // covers [0, 3)
#define PC_MAX   (PC_BIN * PC_BINS)

struct ClosePair { double ta, tb, gap; };

struct ZeroStats {
    std::vector<long long> spacing_hist = std::vector<long long>(SP_BINS, 0);
    long long              spacing_overflow = 0;
    long long              n_spacings = 0;
    std::vector<long long> pc_hist = std::vector<long long>(PC_BINS, 0);
    long long              pc_base = 0;        // zeros used as the left point of pairs
    long long              dips_dropped = 0;   // sign-change pairs removed as numerical dips
    std::vector<ClosePair> close_pairs;
    std::vector<ClosePair> dip_pairs;          // pairs removed by the dip filter (gap ≤ 2h)
};

static inline double zero_density(double t) {
    return std::log(t / (2.0 * M_PI)) / (2.0 * M_PI);
}

// Zero location inside grid interval [i, i+1] by linear interpolation of Z,
// returned as an OFFSET from the chunk start.  Absolute t ~ 3e12 has a double ulp
// of ~5e-4, which would quantise every gap to steps of ~0.002 unfolded units.
static inline double interp_zero_offset(const double* Z, int i, double h) {
    double a = Z[i], b = Z[i + 1];
    double f = (a == b) ? 0.5 : a / (a - b);
    if (f < 0.0) f = 0.0; else if (f > 1.0) f = 1.0;
    return (i + f) * h;
}

// g: zero offsets from the chunk start (sorted).  n: grid length.
static ZeroStats compute_zero_stats_offsets(const std::vector<double>& g, int n,
                                            double t0, double h,
                                            long long dips_dropped, double close_thresh) {
    ZeroStats s;
    s.dips_dropped = dips_dropped;
    const size_t nz = g.size();
    if (nz < 2) return s;

    // Unfold with the density at the chunk midpoint.  Across a 1e5-wide chunk at
    // t ~ 1e12 the density changes by < 1e-8 relative, so one value is exact enough.
    const double rho    = zero_density(t0 + 0.5 * (g.front() + g.back()));
    const double t_last = (n - 1) * h;          // offsets, like g[]

    for (size_t k = 0; k + 1 < nz; k++) {
        double d = (g[k + 1] - g[k]) * rho;
        int b = (int)(d / SP_BIN);
        if (b >= 0 && b < SP_BINS) s.spacing_hist[b]++; else s.spacing_overflow++;
        s.n_spacings++;
        if (d < close_thresh) s.close_pairs.push_back({t0 + g[k], t0 + g[k + 1], d});
    }

    // Pair correlation: only use left points whose full window lies in the chunk,
    // so every base zero sees all of its neighbours up to PC_MAX.
    const double t_win = PC_MAX / rho;
    for (size_t i = 0; i < nz; i++) {
        if (g[i] + t_win > t_last) break;
        s.pc_base++;
        for (size_t j = i + 1; j < nz; j++) {
            double d = (g[j] - g[i]) * rho;
            if (d >= PC_MAX) break;
            s.pc_hist[(int)(d / PC_BIN)]++;
        }
    }
    return s;
}

// Convenience wrapper: idxs are accepted sign-change indices into the Z grid.
static ZeroStats compute_zero_stats(const double* Z, int n, double t0, double h,
                                    const std::vector<int>& idxs,
                                    long long dips_dropped, double close_thresh) {
    std::vector<double> g(idxs.size());
    for (size_t k = 0; k < idxs.size(); k++) g[k] = interp_zero_offset(Z, idxs[k], h);
    return compute_zero_stats_offsets(g, n, t0, h, dips_dropped, close_thresh);
}

