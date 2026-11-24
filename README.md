# riemann-gpu

GPU-accelerated computation of zeros of the Riemann zeta function at large height, spread across a cluster of consumer NVIDIA GPUs. It is an Odlyzko–Schönhage-style NUFFT evaluator of the Hardy Z-function, built to run for days or weeks.

**What it is for now.** A proof of the Riemann Hypothesis was announced in November 2026 (Alpöge, with Claude, formalized in Lean). If it holds up, numerical searches can no longer find a counterexample. So the project's main use has shifted from *testing* RH to *measuring* how the zeros are distributed:

- **Zero statistics.** Nearest-neighbour spacings and pair correlation of hundreds of billions of zeros near t ≈ 10¹², compared with the random-matrix (GUE) predictions of Montgomery, Dyson and Odlyzko. Finite-height deviations from GUE (Bogomolny–Keating) are measurable at this sample size.
- **Lehmer pairs.** Unusually close zero pairs are found on the GPU and then confirmed with arbitrary-precision arithmetic. They are the sharpest probes of zero repulsion and relate to the de Bruijn–Newman constant.
- **A fast Z(t) engine.** About 10⁸ Z-values per second per consumer GPU, usable as a base for other analytic-number-theory computations.

**Earlier run (Oct 2026):** sign changes of Z(t) counted for t ∈ [3×10¹², 4×10¹²], about 3.5×10¹² zeros, chunk by chunk against the smooth zero-count formula. See [Results](#results) and [Caveats](#caveats) for what that run does and doesn't show.

---

## How it works

### The math

The **Riemann Hypothesis** states every non-trivial zero of ζ(s) has real part ½. To test this, we use the **Hardy Z-function**:

```
Z(t) = e^{iθ(t)} ζ(½ + it)
```

Z(t) is real for real t; its sign changes correspond exactly to zeros of ζ on the critical line. Counting those and comparing to **Backlund's formula**:

```
N(T) = ⌊θ(T)/π⌋ + 1
```

gives an approximate per-chunk consistency check. (A rigorous check would need the S(T) term and Turing's method. See Caveats.)

### Computing Z(t) via the Odlyzko-Schönhage algorithm (GPU NUFFT)

The Riemann-Siegel formula expands Z(t) as:

```
Z(t) = 2 · Σ_{n=1}^{N} cos(θ(t) − t·ln n) / √n  +  C₀(t)
       where N = ⌊√(t/2π)⌋
```

For t ~ 4×10¹², N ≈ 800,000 terms evaluated at K = 50,000,000 grid points. Naïve cost: O(N·K) ≈ 4×10¹³ ops per chunk — infeasible even on GPU.

The key insight from **Odlyzko & Schönhage (1988)** is that the main sum is a Fourier-type evaluation: each term n^{-it} = e^{-it ln n} places a source at non-uniform frequency ln(n)/(2π), and evaluating it at K equally-spaced t values is a **Non-Uniform DFT**. This reduces the cost from O(N·K) to O((N+K) log K).

We implement this via the **NUFFT** (Kaiser-Bessel spreading → cuFFT → deconvolution), which is the standard modern realization of the Odlyzko-Schönhage idea. Each 100,000-second chunk (50M Z-values) takes about 0.1 s on a single RTX 5070 Ti.

### Backlund verification (chunk-by-chunk)

The work range is split into chunks of 100,000 t-seconds each. For every chunk `[t_start, t_end]`:

1. Compute Z(t) at 50M points (spacing h = 0.002)
2. Count sign changes (= zeros on critical line)
3. Compare to `N(t_end) − N(t_start)` from Backlund's formula
4. Flag if the difference exceeds 0.1%

Agreement within tolerance means the scan found roughly as many zeros as the smooth count predicts. This is a sanity check on the computation, not a proof that no zeros were missed.

---

## Zero statistics mode

Turn it on in `<mount>/rh/config.json` (see `config.stats.example.json`):

```json
{
  "run":          "stats_3e12",
  "T_start":      3000000000000.0,
  "T_end":        3010000000000.0,
  "chunk_dt":     100000.0,
  "stats":        true,
  "close_thresh": 0.05
}
```

- `run` gives the run its own directory, `<mount>/rh/runs/<run>/{done,active}`, so it never collides with the legacy `rh/done` of the October run. Configs without `run` behave exactly as before.
- `stats: true` makes each worker compute, per chunk:
  - a histogram of **unfolded nearest-neighbour spacings** δ = (γₙ₊₁ − γₙ)·log(γ/2π)/2π (mean 1), bin 0.02, range [0, 4);
  - a histogram of **unfolded pair differences** up to 3, used for the pair correlation R₂(x);
  - every consecutive pair with δ < `close_thresh`, as Lehmer-pair candidates.
  Zero positions come from linear interpolation of Z between grid points. The cost is negligible next to the NUFFT.

A 10¹⁰-wide range is 100,000 chunks, about 3.5×10¹⁰ zeros. That's roughly 2 hours on 4 × RTX 5070 Ti and about 400 MB on the NAS.

**Analyse:**

```bash
python3 rh_stats.py --selftest                       # check GUE reference vs random matrices
python3 rh_stats.py --dir /mnt/math/rh/runs/stats_3e12/done --out results/stats_3e12
python3 rh_pairs.py results/stats_3e12_close_pairs.csv --top 200   # mpmath confirmation
python3 rh_pairs.py --demo                           # reproduces Lehmer's 1956 pair at t≈7005.06
```

`rh_stats.py` writes per-bin CSVs (empirical, GUE/Gaudin or Montgomery reference, Poisson error, z-score), a text report, and the smallest-gap list. The exact GUE spacing density is computed as a Fredholm determinant of the sine kernel (Bornemann's method), not with the Wigner surmise.

`rh_pairs.py` re-evaluates Z with `mpmath.siegelz` (30 digits, about 4 s per evaluation at t ≈ 3×10¹²; about 15 s of wall time per pair on 16 cores). It scans at least ±0.005 around the GPU position, confirms two sign changes, refines both zeros to about 10⁻¹², and reports max |Z| between them. A small max |Z| marks a strong Lehmer pair. `NEAR_MISS` means Z came close to zero without crossing, and the GPU (error up to about 2×10⁻³ in rare spots) mistook that for two zeros.

**Validating the GPU values.** `RH_DUMP=<file>` makes the worker dump Z at 200 points spread evenly across a chunk. `python3 rh_check.py <file>` compares them with `mpmath.siegelz`. At t = 3×10¹² the maximum error is 5×10⁻⁴ across the whole chunk.

Results of the first full run are in [Results](#results).

**Resolution limits.**
- **Pairs closer than about 2h.** With h = 0.002 the grid can't separate zeros closer than about 2h (δ ≈ 0.017 near 3×10¹²). The worker removes such sign-change pairs from the statistics (`dips_dropped`, about 4.4 per chunk) but records their positions as `dip_pairs`, and `rh_stats.py` writes them to `<out>_dip_pairs.csv`. A 16-pair mpmath check found all 16 to be genuine close pairs, so these are the best Lehmer candidates, not noise. The first spacing bin is biased low by these removals.
- **GPU gaps below δ ≈ 0.05.** These are biased low, because linear interpolation fails where Z is nearly parabolic (e.g. 0.0171 reported vs 0.0045 true for the closest pair found). Use `rh_pairs.py` values.

---

## Accuracy fixes (Nov 2026)

Three bugs limited the accuracy of Z in the October run. All three are fixed:

1. **The output band wasn't centred.** Outputs used frequencies 0…K ≈ 0.37·M, past the range the Kaiser-Bessel window (β = 13.8) is accurate for. The error grew from about 2×10⁻³ at the start of each chunk to about 0.14 at the end. The band is now centred on the middle of the chunk (|k| ≤ K/2).
2. **The log-correction table was never applied.** `d_logn_corr` was uploaded to the GPU but never used, leaving a phase error of about 3×10⁻³ rad in each term t·ln n.
3. **θ(t) was evaluated in plain double.** At t ≈ 3×10¹², θ ≈ 4×10¹³, so rounding alone cost about 10⁻² rad. θ is now a Taylor expansion around the chunk centre, with θ(t_c) mod 2π taken in long double.

Effect at t = 3×10¹²: the maximum error against mpmath fell from about 0.14 to 5×10⁻⁴. Numerical dips per chunk fell from about 5,700 to about 4.5, and found minus expected per chunk went from about −240 to about −10.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Coordinator (Python)                    │
│  builds work queue · streams chunks via SSH · collects results│
│  writes NAS · updates live dashboard                        │
└────────────────────┬────────────────────────────────────────┘
                     │ SSH (streaming JSON, one chunk per line)
          ┌──────────┴──────────┐
          │     Per node        │  × N nodes
          │   M worker procs    │
          │   ┌─────────────┐   │
          │   │ stdin reader│   │  pre-buffers 4 chunks
          │   │  GPU thread │   │  NUFFT, never waits for I/O
          │   │  NAS writer │   │  saves results concurrently
          │   └─────────────┘   │
          └─────────────────────┘
```

Workers run a 3-thread pipeline so the GPU computes continuously: while the GPU is computing chunk N+1, the NAS writer is saving chunk N's result and the stdin reader is buffering chunk N+2.

---

## Requirements

- NVIDIA GPU, sm_90+ (Hopper / Ada / Blackwell) — fp64 atomicAdd is hardware-native from sm_60 but fastest on sm_90+
- CUDA Toolkit 12+, with `libcufft`
- Python 3.10+ with `numpy`, `flask`
- **Passwordless SSH** from the coordinator node to every worker node (see setup below)
- Shared storage (NAS/NFS) mounted at the same path on all nodes

Tested on 4× RTX 5070 Ti (Blackwell, sm_120) in WSL2 on Windows 11.

### Setting up passwordless SSH

The coordinator streams work to workers over SSH. Password prompts would block it, so key-based auth is required. Run this **once on the coordinator**:

```bash
ssh-keygen -t ed25519          # accept defaults; skip if you already have a key
ssh-copy-id user@192.168.1.10  # repeat for each worker node IP
ssh-copy-id user@192.168.1.11
```

Test that it works without a password prompt:
```bash
ssh user@192.168.1.10 echo ok
```

---

## Build

```bash
# On the coordinator node (binary is deployed to workers via make install)
make
make install
```

The `Makefile` targets `sm_120` (RTX 50xx Blackwell) by default. For other architectures:

```bash
nvcc -O3 -arch=sm_90 -o rh_worker_cuda rh_cuda.cu -lcufft -lpthread
```

---

## Configuration

**1. Edit `rh_coordinator.py` — top section:**
```python
# Add or remove IPs freely — any number of nodes works (1 to N)
NODES            = ["192.168.1.10", "192.168.1.11", "192.168.1.12"]
WORKERS_PER_NODE = 1      # workers per node; 1 is fastest on RTX 5070 Ti (see "Performance" below)
WORK_DIR         = "/home/user/riemann-gpu"
```

**2. Edit `rh_dashboard.py` — top section:**
```python
NODES = ["192.168.1.10", "192.168.1.11"]  # same as coordinator
NODE_HOST_MAP = {                          # hostname → IP for nvidia-smi
    "MyPC": "192.168.1.10",
}
```

**3. Set the search range in `<mount>/rh/config.json`:**

This file controls what region of the critical line is searched. Create or edit it directly:

```json
{
  "T_start":  14.0,
  "T_end":    1000000.0,
  "chunk_dt": 100000.0
}
```

| Field | Meaning | Notes |
|---|---|---|
| `T_start` | Lower bound of search (imaginary part of s) | 14.134 is the first known zero; start at 14 to cover everything |
| `T_end` | Upper bound | Set to whatever height you want to reach |
| `chunk_dt` | Width of each work unit in t-seconds | 100,000 is a good default (~0.1 s/chunk on RTX 5070 Ti) |

The coordinator splits `[T_start, T_end]` into chunks of `chunk_dt` and distributes them across the cluster. It resumes automatically on restart — already-completed chunks are skipped.

**Where to start?** The first ~10¹³ zeros have been rigorously verified by academic groups. To push into new territory, set `T_start` to match the highest verified value in the literature (Platt & Trudgian 2021 reached t ≈ 3×10¹²; our run extended to 4×10¹²). To re-verify a known range as a sanity check, any interval works.

Or generate a default config automatically:
```bash
python3 -c "import nas_fs as nas; nas.ensure_default_config()"
```

**4. First-time cluster setup:**
```bash
bash setup.sh   # installs dependencies, mounts NAS, copies code to all nodes
```

---

## Run

```bash
# Start the live dashboard (visit http://<coordinator-ip>:8080)
screen -dmS rh_dash bash -c 'python3 rh_dashboard.py'

# Start the coordinator (runs until T_end is reached or Ctrl-C)
screen -dmS rh_coord bash -c 'python3 rh_coordinator.py | tee coordinator.log'
```

The coordinator picks up where it left off on restart — done chunks are skipped automatically.

---

## VRAM usage and tuning

For K = 50M samples per chunk (h = 0.002, chunk_dt = 100,000), the grid size is M = 2²⁷ = 134M complex doubles:

| Buffer | Size | Purpose |
|---|---|---|
| `d_grid_cx` | 2,147 MB | Complex NUFFT grid (spread target + FFT input) |
| `d_Z_nufft` | 400 MB | Z(t) output per chunk |
| `d_logn_corr` | 9 MB | log correction table (precision fix at large t) |
| sign-change buffers | ~38 MB | GPU compaction output (indices + zero offsets) |
| cuFFT workspace | ~2,147 MB | Internal FFT scratch |
| **Total** | **~4.7 GB** | per worker |

The grid is a single `cufftDoubleComplex*` — spreading writes directly into `.x` and `.y` via `atomicAdd`, eliminating the separate real/imaginary buffers that an earlier design used. This saves about 2 GB per worker, which is what lets 3 workers fit on a 16 GB GPU if you ever need them.

## Performance

Sign changes are found on the GPU. A count kernel runs first, then CUB `DeviceSelect` (ordered stream compaction), then a kernel that interpolates each zero's position. Only about 5 MB of indices and offsets come back to the host, not all 400 MB of Z. Before this change, each chunk spent about 0.18 s on the host: allocating a 400 MB array (0.09 s), copying Z back (0.04 s) and scanning for sign changes on one thread (0.05 s). Output is byte-identical to the old host path. `RH_HOST_SIGNS=1` forces the host path, which is also the automatic fallback.

Measured on one RTX 5070 Ti (WSL2) at t ≈ 3.3×10¹², with stats on:

| Binary | Workers/GPU | Chunks/s (steady state) |
|---|---|---|
| host sign detection (Oct 2026) | 3 | 6.55 |
| GPU sign detection | **1** | **9.80** |
| GPU sign detection | 2 | 8.09 |
| GPU sign detection | 3 | 8.18 |

The GPU is now the bottleneck, so more workers only compete for it. Use `WORKERS_PER_NODE = 1`. On 4 nodes that's about 39 chunks/s, roughly 1.4×10⁷ zeros per second.

---

## Results

### Nov 2026: zero statistics, t ∈ [3.00, 3.01]×10¹²

100,000 chunks, **42,801,165,812 zeros**, 0 Backlund failures. About 42 minutes on 4 × RTX 5070 Ti (~39 chunks/s). Reproduce with `config.stats.example.json`, then `rh_stats.py` and `rh_pairs.py`.

**Spacing distribution.** It deviates from the GUE (Gaudin) limit at hundreds of σ, in a stable pattern: too few gaps near s ≈ 0.25–0.5 and s ≈ 2–2.5, too many near s ≈ 1. The distribution is narrower than GUE. The finite-height model of Bogomolny, Bohigas, Leboeuf & Monastra (2006) treats the zeros as eigenvalues of a random unitary matrix of effective size N_eff = log(t/2π)/√(12Λ) = 6.19 (Λ ≈ 1.57314). That model explains most of the deviation:

| | GUE limit | CUE(N_eff = 6.19) |
|---|---|---|
| Kolmogorov distance of spacing CDF | 3.28×10⁻³ | 1.13×10⁻³ |
| χ² over 150 spacing bins (s < 3) | 13.9M | 7.4M |
| Density at s = 0.10 (empirical 0.03827) | 0.03928 (−151σ) | 0.03828 (−1σ) |

The model is near-exact for small gaps but leaves significant residuals in the tail (s ≳ 1.5). Freely fitting N gives N ≈ 7.8 rather than 6.19. With 4×10¹⁰ zeros, small effects show up at high σ: next-order corrections, and measurement effects such as dropped close pairs adding long gaps. So the residuals need modelling before they can be interpreted.

**Pair correlation.** It deviates from Montgomery's 1 − (sin πx/πx)² in the same way at short range. CUE(N_eff) fixes x ≲ 0.5 but overshoots for x ≳ 1.5, where the data stay closer to Montgomery. That's consistent with N_eff being a nearest-neighbour model; long-range correlations carry the arithmetic (Bogomolny–Keating) corrections, which aren't implemented here.

**Lehmer pairs.** 200 of the closest GPU candidates were checked with mpmath. 199 are genuine pairs, and 1 is a near-miss: Z dips to about −1×10⁻³ without crossing. The tightest pairs:

| δ (unfolded) | gap in t | max \|Z\| between | t₁ |
|---|---|---|---|
| 0.00451 | 0.00105 | 1.5×10⁻⁵ | 3000265777242.715400332 |
| 0.00561 | 0.00131 | 2.7×10⁻⁵ | 3000099793213.309308370 |
| 0.00573 | 0.00134 | 3.0×10⁻⁵ | 3001414304717.801272679 |
| 0.00720 | 0.00168 | 6.2×10⁻⁵ | 3003017360194.645113965 |

The `dip_pairs` field was added after this run, so its 439,049 sub-2h pairs aren't recorded. Re-running the range would capture them.

### Oct 2026: sign-change count, t ∈ [3×10¹², 4×10¹²]

| Run | t range | Zeros verified | Chunks | Backlund OK | Duration |
|---|---|---|---|---|---|
| Oct 2026 | [3.0×10¹², 4.0×10¹²] | 3,545,153,568,677 | 10,000,000 | 99.97% | ~7 days |

The 0.03% Backlund mismatch rate was originally put down to closely spaced zero pairs. It was actually caused mostly by the NUFFT accuracy bugs described in [Accuracy fixes](#accuracy-fixes-nov-2026). Before the fix, each chunk dropped about 5,700 spurious sign-change pairs and still came out about 240 zeros short, so the October per-chunk counts are approximate. Only zero counts were stored for matching chunks. Zero positions exist only for the 3,456 mismatch chunks, so this run can't be reused for spacing statistics. Use stats mode.

---

## Caveats

This is a **measurement tool**, not a proof-grade verifier:

- Uses floating-point arithmetic, not exact/interval arithmetic
- Verifies chunks independently via local Backlund counts, not the full cumulative count from t = 0
- The 0.1% tolerance in the Backlund check means very closely-spaced zero pairs can be missed

For a rigorous record-extending verification (like Platt & Trudgian 2021), exact arithmetic and a complete cumulative verification from t = 0 would be required.

- Zero statistics near 10¹² aren't new. Odlyzko studied zeros near 10²⁰ and 10²³. The value here is a large, reproducible, open dataset from consumer hardware, plus the pipeline for confirming close pairs.

---

## License

MIT
