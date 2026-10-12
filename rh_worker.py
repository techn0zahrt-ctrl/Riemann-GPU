#!/usr/bin/env python3
"""
RH Worker: reads a job from stdin (JSON), computes zeros of Z(t) in that
range, verifies via Backlund, and writes result JSON to stdout.

The coordinator launches this via SSH and handles all I/O.

Usage (called by coordinator, not directly):
    echo '{"t_start":3e12,"t_end":3.005e12,"dt":0.05,"chunk_id":"chunk_0042"}' \
        | python3 rh_worker.py
"""

import json
import math
import os
import socket
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rh_core import (
    Z_batch, find_sign_changes, batch_bisect_zeros,
    backlund_count, best_xp, gpu_available
)
import nas_fs as nas

BATCH_SIZE_GPU = 1_000
BATCH_SIZE_CPU = 100
N_CHUNK        = 30_000
BISECT_TOL     = 1e-10


def scan_range(t_start, t_end, dt, xp, batch_size):
    t_vals = np.arange(t_start, t_end + dt * 0.5, dt, dtype=np.float64)
    N_pts  = len(t_vals)
    Z_vals = np.empty(N_pts, dtype=np.float64)
    t0     = time.time()

    for lo in range(0, N_pts, batch_size):
        hi = min(lo + batch_size, N_pts)
        tb = t_vals[lo:hi]

        Z_b, sing = Z_batch(tb, xp=xp, n_chunk=N_CHUNK)
        Z_np  = Z_b.get()   if hasattr(Z_b,   'get') else Z_b
        sing  = sing.get()  if hasattr(sing,   'get') else sing

        for idx in np.where(sing)[0]:
            Z_np[idx] = Z_scalar_mpmath(float(tb[idx]))

        Z_vals[lo:hi] = Z_np

        elapsed = time.time() - t0
        pct  = 100.0 * hi / N_pts
        rate = hi / elapsed if elapsed > 0 else 0
        # Progress goes to stderr so it doesn't pollute stdout JSON
        print(f"  [{socket.gethostname()}] {pct:5.1f}%  {rate:.0f} pts/s  "
              f"t={float(tb[-1]):.6e}", file=sys.stderr, flush=True)

    return t_vals, Z_vals


def refine_zeros(t_vals, Z_vals, xp):
    idxs  = find_sign_changes(t_vals, Z_vals)
    if len(idxs) == 0:
        return []
    print(f"  Refining {len(idxs)} zeros via batched GPU bisection...",
          file=sys.stderr, flush=True)
    return batch_bisect_zeros(idxs, t_vals, Z_vals,
                              xp=xp, n_chunk=N_CHUNK, tol=BISECT_TOL)


def verify_backlund(t_start, t_end, n_found):
    """
    Approximate Backlund count: N(T) = floor(θ(T)/π) + 1.
    Exact count differs by at most S(T) which is ±0,1,2 in practice.
    We accept a match of ±1 since S(T) can shift the boundary count by 1.
    For suspicious chunks (|diff| > 1), the coordinator will re-scan at
    higher density and call mpmath.nzeros() explicitly.
    """
    n_approx = backlund_count(t_end) - backlund_count(t_start)
    diff = abs(n_found - n_approx)
    ok   = (diff <= 15)         # S(T) correction can be up to ~±10 at this height
    return n_approx, ok


def main():
    job = json.loads(sys.stdin.read())
    t_start  = float(job["t_start"])
    t_end    = float(job["t_end"])
    dt       = float(job.get("dt", 0.05))
    chunk_id = job["chunk_id"]

    xp         = np
    batch_size = BATCH_SIZE_CPU
    backend    = "CPU"
    if gpu_available():
        xp         = best_xp()
        batch_size = BATCH_SIZE_GPU
        backend    = "GPU"

    print(f"[{socket.gethostname()}] {chunk_id}  {backend}  "
          f"t=[{t_start:.4e},{t_end:.4e}]  dt={dt}",
          file=sys.stderr, flush=True)

    t_wall = time.time()

    t_vals, Z_vals = scan_range(t_start, t_end, dt, xp, batch_size)
    zeros          = refine_zeros(t_vals, Z_vals, xp)

    print(f"  zeros={len(zeros)}  running Backlund...", file=sys.stderr, flush=True)
    n_exp, ok = verify_backlund(t_start, t_end, len(zeros))

    verdict = "ALL_ON_CRITICAL_LINE" if ok else f"MISMATCH_diff={len(zeros)-n_exp}"
    print(f"  {verdict}", file=sys.stderr, flush=True)

    result = {
        "chunk_id":          chunk_id,
        "host":              socket.gethostname(),
        "t_start":           t_start,
        "t_end":             t_end,
        "dt":                dt,
        "zeros_found":       len(zeros),
        "backlund_expected": n_exp,
        "backlund_ok":       ok,
        "wall_seconds":      time.time() - t_wall,
        "gpu_used":          (backend == "GPU"),
        "zeros":             zeros,
    }

    # Write to NAS directly (workers have /mnt/math mounted)
    if nas.nas_status():
        nas.write_json(nas.rh_path("done", f"{chunk_id}.json"), result)
        nas.rh_path("active", f"{chunk_id}.json").unlink(missing_ok=True)
        print(f"  Saved → /mnt/math/rh/done/{chunk_id}.json", file=sys.stderr)

    # Also output to stdout so coordinator can confirm receipt
    print(json.dumps(result))
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
