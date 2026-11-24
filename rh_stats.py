#!/usr/bin/env python3
"""
Aggregate zero statistics from a stats run and compare them with random-matrix (GUE)
predictions.

Each chunk JSON written by rh_worker_cuda in stats mode carries:
  spacing_hist  unfolded nearest-neighbour gap counts      (bin width sp_bin)
  pc_hist       unfolded pair-difference counts            (bin width pc_bin)
  close_pairs   [t_a, t_b, unfolded_gap] with gap < close_thresh

This script sums the histograms over all chunks, computes the exact GUE references

  spacing:          Gaudin distribution  p(s) = d²/ds² det(I − K_sine)|_[0,s]
  pair correlation: Montgomery           R₂(x) = 1 − (sin πx / πx)²

and writes:
  <out>_spacing.csv, <out>_paircorr.csv   per-bin empirical vs GUE
  <out>_close_pairs.csv                   the --top smallest gaps (input for rh_pairs.py)
  <out>_report.txt                        summary

Usage:
  python3 rh_stats.py --dir /mnt/math/rh/runs/stats1/done --out results/stats1
  python3 rh_stats.py --selftest          # check the GUE reference against GUE matrices
"""

import argparse
import heapq
import json
import math
import os
import sys
import time

import numpy as np


# ── GUE references ───────────────────────────────────────────────────────────

def gue_gap_probability(s_vals, m=40):
    """E(0; s) = det(I − K_sine) on [0, s], via Bornemann's Gauss–Legendre method."""
    x, w = np.polynomial.legendre.leggauss(m)
    out = np.empty(len(s_vals))
    for i, s in enumerate(s_vals):
        if s <= 0:
            out[i] = 1.0
            continue
        xs = 0.5 * s * (x + 1.0)
        ws = 0.5 * s * w
        d = xs[:, None] - xs[None, :]
        K = np.sinc(d)                         # numpy sinc(x) = sin(πx)/(πx)
        A = np.eye(m) - np.sqrt(ws)[:, None] * K * np.sqrt(ws)[None, :]
        out[i] = np.linalg.det(A)
    return out


def gue_spacing_bin_probs(edges, eps=1e-4):
    """P(gap in [a, b]) = E'(b) − E'(a), since p = E'' and E'(0) = −1."""
    e = np.asarray(edges, dtype=float)
    dE = (gue_gap_probability(e + eps) - gue_gap_probability(np.maximum(e - eps, 0))) \
         / (e + eps - np.maximum(e - eps, 0))
    dE[0] = -1.0
    return np.diff(dE)


def gue_paircorr_bin_means(edges, sub=20):
    """Bin-averaged Montgomery pair correlation 1 − sinc²."""
    e = np.asarray(edges, dtype=float)
    out = np.empty(len(e) - 1)
    for i in range(len(e) - 1):
        x = np.linspace(e[i], e[i + 1], sub + 1)
        out[i] = np.trapezoid(1.0 - np.sinc(x) ** 2, x) / (e[i + 1] - e[i])
    return out


def wigner_surmise(s):
    return (32.0 / math.pi ** 2) * s * s * np.exp(-4.0 * s * s / math.pi)


# ── Aggregation ──────────────────────────────────────────────────────────────

def aggregate(done_dir, top, max_files=None):
    sp = pc = None
    sp_bin = pc_bin = None
    n_sp = n_over = pc_base = dips = 0
    n_files = n_stats = 0
    zeros = 0
    t_lo, t_hi = math.inf, -math.inf
    heap = []          # max-heap on gap via negation: keeps the `top` smallest
    t0 = time.time()

    with os.scandir(done_dir) as it:
        for ent in it:
            if not ent.name.endswith(".json"):
                continue
            n_files += 1
            try:
                with open(ent.path) as f:
                    c = json.load(f)
            except (OSError, ValueError) as e:
                print(f"  [warn] {ent.name}: {e}", file=sys.stderr)
                continue
            s = c.get("stats")
            if not s:
                continue
            n_stats += 1
            zeros += c.get("zeros_found", 0)
            t_lo = min(t_lo, c["t_start"]); t_hi = max(t_hi, c["t_end"])
            h1 = np.asarray(s["spacing_hist"], dtype=np.int64)
            h2 = np.asarray(s["pc_hist"], dtype=np.int64)
            if sp is None:
                sp, pc = h1.copy(), h2.copy()
                sp_bin, pc_bin = s["sp_bin"], s["pc_bin"]
            else:
                sp += h1; pc += h2
            n_sp += s["n_spacings"]; n_over += s["spacing_overflow"]
            pc_base += s["pc_base"]; dips += s.get("dips_dropped", 0)
            for ta, tb, g in s.get("close_pairs", []):
                item = (-g, ta, tb, c["chunk_id"])
                if len(heap) < top:
                    heapq.heappush(heap, item)
                elif -g > heap[0][0]:
                    heapq.heapreplace(heap, item)
            if n_files % 100_000 == 0:
                print(f"  {n_files:,d} files  {time.time()-t0:.0f}s", file=sys.stderr)
            if max_files and n_files >= max_files:
                break

    pairs = sorted(((-g, ta, tb, cid) for g, ta, tb, cid in heap))
    return dict(sp=sp, pc=pc, sp_bin=sp_bin, pc_bin=pc_bin, n_sp=n_sp, n_over=n_over,
                pc_base=pc_base, dips=dips, n_files=n_files, n_stats=n_stats,
                zeros=zeros, t_lo=t_lo, t_hi=t_hi, pairs=pairs)


def compare(agg):
    sp, pc = agg["sp"], agg["pc"]
    sp_edges = np.arange(len(sp) + 1) * agg["sp_bin"]
    pc_edges = np.arange(len(pc) + 1) * agg["pc_bin"]
    n = agg["n_sp"]

    emp_sp = sp / n / agg["sp_bin"]                         # density
    gue_sp = gue_spacing_bin_probs(sp_edges) / agg["sp_bin"]
    se_sp  = np.sqrt(np.maximum(sp, 1)) / n / agg["sp_bin"]  # Poisson error

    emp_pc = pc / agg["pc_base"] / agg["pc_bin"]
    gue_pc = gue_paircorr_bin_means(pc_edges)
    se_pc  = np.sqrt(np.maximum(pc, 1)) / agg["pc_base"] / agg["pc_bin"]

    # Kolmogorov distance of spacing CDFs (empirical vs GUE)
    cdf_emp = np.cumsum(sp) / n
    cdf_gue = np.cumsum(gue_spacing_bin_probs(sp_edges))
    ks = float(np.max(np.abs(cdf_emp - cdf_gue)))
    return dict(sp_edges=sp_edges, pc_edges=pc_edges, emp_sp=emp_sp, gue_sp=gue_sp,
                se_sp=se_sp, emp_pc=emp_pc, gue_pc=gue_pc, se_pc=se_pc, ks=ks)


def write_outputs(agg, cmp, out):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(f"{out}_spacing.csv", "w") as f:
        f.write("s_lo,s_hi,count,empirical_density,gue_density,std_err,z_score\n")
        for i in range(len(agg["sp"])):
            z = (cmp["emp_sp"][i] - cmp["gue_sp"][i]) / cmp["se_sp"][i]
            f.write(f"{cmp['sp_edges'][i]:.3f},{cmp['sp_edges'][i+1]:.3f},{agg['sp'][i]},"
                    f"{cmp['emp_sp'][i]:.6g},{cmp['gue_sp'][i]:.6g},{cmp['se_sp'][i]:.3g},{z:.2f}\n")
    with open(f"{out}_paircorr.csv", "w") as f:
        f.write("x_lo,x_hi,count,empirical_R2,montgomery_R2,std_err,z_score\n")
        for i in range(len(agg["pc"])):
            z = (cmp["emp_pc"][i] - cmp["gue_pc"][i]) / cmp["se_pc"][i]
            f.write(f"{cmp['pc_edges'][i]:.3f},{cmp['pc_edges'][i+1]:.3f},{agg['pc'][i]},"
                    f"{cmp['emp_pc'][i]:.6g},{cmp['gue_pc'][i]:.6g},{cmp['se_pc'][i]:.3g},{z:.2f}\n")
    with open(f"{out}_close_pairs.csv", "w") as f:
        f.write("rank,unfolded_gap,t_a,t_b,chunk_id\n")
        for r, (g, ta, tb, cid) in enumerate(agg["pairs"], 1):
            f.write(f"{r},{g:.6f},{ta:.4f},{tb:.4f},{cid}\n")

    lines = [
        "Zeta zero statistics vs GUE",
        "=" * 60,
        f"chunks with stats : {agg['n_stats']:,d}  (of {agg['n_files']:,d} files)",
        f"t range           : [{agg['t_lo']:.6e}, {agg['t_hi']:.6e}]",
        f"zeros             : {agg['zeros']:,d}",
        f"spacings          : {agg['n_sp']:,d}  (overflow ≥ {cmp['sp_edges'][-1]:.0f}: {agg['n_over']:,d})",
        f"pair-corr bases   : {agg['pc_base']:,d}",
        f"dips dropped      : {agg['dips']:,d}  (unresolved pairs closer than 2h — see README)",
        "",
        f"Spacing CDF Kolmogorov distance to GUE (Gaudin): {cmp['ks']:.3e}",
        "",
        "  s      empirical   GUE        z",
    ]
    for s in (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5):
        i = int(round(s / agg["sp_bin"]))
        if i < len(agg["sp"]):
            z = (cmp["emp_sp"][i] - cmp["gue_sp"][i]) / cmp["se_sp"][i]
            lines.append(f"  {s:4.2f}   {cmp['emp_sp'][i]:.5f}    {cmp['gue_sp'][i]:.5f}   {z:+6.1f}")
    lines += ["", "  x      R2 emp      Montgomery z"]
    for x in (0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5):
        i = int(round(x / agg["pc_bin"]))
        if i < len(agg["pc"]):
            z = (cmp["emp_pc"][i] - cmp["gue_pc"][i]) / cmp["se_pc"][i]
            lines.append(f"  {x:4.2f}   {cmp['emp_pc'][i]:.5f}    {cmp['gue_pc'][i]:.5f}   {z:+6.1f}")
    lines += ["", f"Smallest gaps (of {len(agg['pairs'])} kept; full list in {out}_close_pairs.csv):"]
    for g, ta, tb, cid in agg["pairs"][:10]:
        lines.append(f"  δ={g:.5f}   t≈{ta:.4f} … {tb:.4f}   {cid}")
    lines += ["",
              "Note: at finite height the zeros deviate slightly from GUE (Bogomolny–Keating",
              "corrections of order 1/log²(t/2π)). Large |z| near the peak is expected,",
              "not an error — measuring that deviation is the point of the run."]
    txt = "\n".join(lines)
    with open(f"{out}_report.txt", "w") as f:
        f.write(txt + "\n")
    print(txt)


# ── Self-test against actual GUE matrices ────────────────────────────────────

def selftest():
    rng = np.random.default_rng(0)
    nm, reps, gaps = 300, 60, []
    for _ in range(reps):
        a = rng.normal(size=(nm, nm)) + 1j * rng.normal(size=(nm, nm))
        ev = np.linalg.eigvalsh((a + a.conj().T) / 2)
        r = 2.0 * math.sqrt(nm)                       # semicircle radius
        x = np.clip(ev / r, -1, 1)
        u = nm * (0.5 + (x * np.sqrt(1 - x * x) + np.arcsin(x)) / math.pi)
        u = u[(u > nm * 0.2) & (u < nm * 0.8)]
        gaps.append(np.diff(u))
    g = np.concatenate(gaps)
    edges = np.linspace(0, 3, 31)
    emp = np.histogram(g, bins=edges)[0] / len(g)
    ref = gue_spacing_bin_probs(edges)
    print(f"sum of GUE bin probs on [0,3): {ref.sum():.6f}  (expect ≈ 0.99999)")
    print(f"mean spacing of sample: {g.mean():.4f}")
    print(" bin       sample    Gaudin   Wigner")
    for i in range(0, 30, 3):
        mid = 0.5 * (edges[i] + edges[i + 1])
        print(f" {edges[i]:.1f}-{edges[i+1]:.1f}   {emp[i]:.4f}   {ref[i]:.4f}   "
              f"{wigner_surmise(mid) * 0.1:.4f}")
    err = np.max(np.abs(emp - ref))
    print(f"max |sample − Gaudin| = {err:.4f}  ({'OK' if err < 0.01 else 'CHECK'})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--dir", help="done/ directory of a stats run")
    ap.add_argument("--out", default="results/zero_stats", help="output file prefix")
    ap.add_argument("--top", type=int, default=1000, help="close pairs to keep")
    ap.add_argument("--max-files", type=int, help="stop after N files (quick look)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.dir:
        ap.error("--dir is required (or use --selftest)")
    agg = aggregate(a.dir, a.top, a.max_files)
    if not agg["n_stats"]:
        sys.exit(f"No chunks with 'stats' in {a.dir}. Was the run started with \"stats\": true?")
    write_outputs(agg, compare(agg), a.out)


if __name__ == "__main__":
    main()
