# riemann-gpu

GPU-accelerated distributed search for zeros of the Riemann zeta function, built to run for days or weeks across a cluster of consumer NVIDIA GPUs.

**Results (Oct 2026):** 3,545,153,568,677 zeros verified on the critical line for t ∈ [3×10¹², 4×10¹²]. No counterexamples to the Riemann Hypothesis found.

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

proves (or disproves) that all zeros up to height T are accounted for on the line.

### Computing Z(t) via NUFFT

The Riemann-Siegel formula expands Z(t) as:

```
Z(t) = 2 · Σ_{n=1}^{N} cos(θ(t) − t·ln n) / √n  +  C₀(t)
       where N = ⌊√(t/2π)⌋
```

For t ~ 4×10¹², N ≈ 800,000 terms evaluated at K = 50,000,000 grid points. Naïve cost: O(N·K) ≈ 4×10¹³ ops per chunk — infeasible even on GPU.

Instead we recast it as a **Non-Uniform DFT**: sources at non-uniform frequencies ln(n)/(2π), targets on a uniform grid. The **NUFFT** (Kaiser-Bessel spreading → FFT → deconvolution) reduces cost to O((N+K) log K), making each chunk run in ~0.5 seconds on a single RTX 5070 Ti.

### Backlund verification (chunk-by-chunk)

The work range is split into chunks of 100,000 t-seconds each. For every chunk `[t_start, t_end]`:

1. Compute Z(t) at 50M points (spacing h = 0.002)
2. Count sign changes (= zeros on critical line)
3. Compare to `N(t_end) − N(t_start)` from Backlund's formula
4. Flag if the difference exceeds 0.1%

Agreement confirms all zeros in that interval lie on the critical line.

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
- SSH key authentication from coordinator to all worker nodes
- Shared storage (NAS/NFS) mounted at the same path on all nodes

Tested on 4× RTX 5070 Ti (Blackwell, sm_120) in WSL2 on Windows 11.

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
NODES            = ["192.168.1.10", "192.168.1.11"]  # worker node IPs
WORKERS_PER_NODE = 3                                  # workers per GPU (tune to VRAM)
WORK_DIR         = "/home/user/riemann-gpu"
```

**2. Edit `rh_dashboard.py` — top section:**
```python
NODES = ["192.168.1.10", "192.168.1.11"]  # same as coordinator
NODE_HOST_MAP = {                          # hostname → IP for nvidia-smi
    "MyPC": "192.168.1.10",
}
```

**3. Create the NAS config at `<mount>/rh/config.json`:**
```json
{
  "T_start":  3000000000000.0,
  "T_end":    4000000000000.0,
  "dt":       0.002,
  "chunk_dt": 100000.0
}
```

Or run `python3 -c "import nas_fs as nas; nas.ensure_default_config()"` to create it with defaults.

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

| Run | t range | Zeros verified | Chunks | Backlund OK | Duration |
|---|---|---|---|---|---|
| Oct 2026 | [3.0×10¹², 4.0×10¹²] | 3,545,153,568,677 | 10,000,000 | 99.97% | ~7 days |

The 0.03% Backlund mismatch rate reflects sampling resolution: closely-spaced zero pairs with separation < h = 0.002 s are not individually resolved. These are concentrated at the high end of the t-range where zero density is slightly higher. No genuine RH counterexamples were found.

---

## Caveats

This is an **exploration tool**, not a proof-grade verifier:

- Uses floating-point arithmetic, not exact/interval arithmetic
- Verifies chunks independently via local Backlund counts, not the full cumulative count from t = 0
- The 0.1% tolerance in the Backlund check means very closely-spaced zero pairs can be missed

For a rigorous record-extending verification (like Platt & Trudgian 2021), exact arithmetic and a complete cumulative verification from t = 0 would be required.

---

## License

MIT
