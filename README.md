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

We implement this via the **NUFFT** (Kaiser-Bessel spreading → cuFFT → deconvolution), which is the standard modern realization of the Odlyzko-Schönhage idea. Each 100,000-second chunk runs in ~0.5 seconds on a single RTX 5070 Ti.

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

`rh_pairs.py` re-evaluates Z with `mpmath.siegelz` (30 digits, about 1.8 s per evaluation at t ≈ 3×10¹²). It confirms two sign changes, refines both zeros to about 10⁻¹², and reports max |Z| between them. A small max |Z| marks a strong Lehmer pair.

**Validating the GPU values.** `RH_DUMP=<file>` makes the worker dump Z at 200 points spread evenly across a chunk. `python3 rh_check.py <file>` compares them with `mpmath.siegelz`. At t = 3×10¹² the maximum error is 5×10⁻⁴ across the whole chunk.

**First results (Nov 2026, 20 chunks, 8.56M zeros at t = 3×10¹²):**
- Found minus expected is about −10 zeros per chunk. That matches the roughly 4.5 unresolvable close pairs dropped per chunk (2 zeros each).
- Spacing CDF Kolmogorov distance to GUE is 3.3×10⁻³. Compared with GUE there's a deficit near s ≈ 0.25–0.5 and s ≈ 2 and an excess near s ≈ 1: the distribution is slightly narrower than GUE, the shape the finite-height (Bogomolny–Keating) theory leads you to expect. It hasn't been compared with their formula yet.
- The 8 closest candidates were all confirmed by mpmath as true pairs, with unfolded gaps of 0.016–0.023. The tightest have max |Z| ≈ 3×10⁻⁴ between the zeros.
- Unfolded gaps from the GPU are biased low for very close pairs (e.g. 0.018 reported vs 0.023 true), because linear interpolation fails where Z is nearly parabolic. Use `rh_pairs.py` values for those.

**Resolution limits.** With h = 0.002 the grid can't separate zeros closer than about 2h, which is δ ≈ 0.017 near 3×10¹². The worker removes those as possible numerical dips and only reports how many (`dips_dropped`). The first spacing bin is therefore biased low, and the very closest pairs have to be found another way. For example, the Backlund-mismatch chunks of the October run, which store zero positions, are where missed pairs show up as unusually large gaps.

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
WORKERS_PER_NODE = 3      # parallel workers per node; tune to your GPU's VRAM (see table below)
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
| `chunk_dt` | Width of each work unit in t-seconds | 100,000 is a good default (~0.5 sec/chunk on RTX 5070 Ti) |

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
| cuFFT workspace | ~2,147 MB | Internal FFT scratch |
| **Total** | **~4.7 GB** | per worker |

The grid is a single `cufftDoubleComplex*` — spreading writes directly into `.x` and `.y` via `atomicAdd`, eliminating the separate real/imaginary buffers that an earlier design used. This saves ~2 GB per worker and is the key optimization that allows 3 workers on a 16 GB GPU.

`WORKERS_PER_NODE` controls the tradeoff: more workers = more throughput (up to bandwidth saturation) but more VRAM. 3 workers fit comfortably on RTX 5070 Ti (16 GB), leaving ~750 MB headroom.

---

## Results

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
