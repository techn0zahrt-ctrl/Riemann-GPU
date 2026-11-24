#!/usr/bin/env python3
"""
Confirm close zero pairs (Lehmer-pair candidates) with high-precision Z(t).

The GPU scan finds zeros from a grid with h = 0.002 and ~1e-3 absolute accuracy in Z,
so the closest pairs it reports need an independent check. For each candidate this
script uses mpmath.siegelz (arbitrary precision Riemann–Siegel) to:

  1. sample Z on [t_a − Δ, t_b + Δ] and confirm exactly two sign changes,
  2. refine both zeros to ~1e-12,
  3. measure max |Z| between them — a tiny value marks a true Lehmer pair, the
     near-misses of RH that are especially sensitive tests of zero statistics
     (and that bound the de Bruijn–Newman constant Λ from below).

Input is the <out>_close_pairs.csv from rh_stats.py.

Usage:
  python3 rh_pairs.py results/stats1_close_pairs.csv --top 200 --jobs 16
  python3 rh_pairs.py --demo        # Lehmer's classic pair near t = 7005.06
"""

import argparse
import csv
import math
import os
import sys
from multiprocessing import Pool

import mpmath

DPS = 30


def Z(t):
    return mpmath.siegelz(t)


def density(t):
    return math.log(float(t) / (2 * math.pi)) / (2 * math.pi)


def check_pair(args):
    ta, tb, label = args
    mpmath.mp.dps = DPS
    ta, tb = mpmath.mpf(ta), mpmath.mpf(tb)
    # GPU positions of very close pairs can be off by a few 1e-3 (|Z| ~ GPU error there),
    # so search at least ±0.005 around them.
    d = max(tb - ta, mpmath.mpf("0.005"))
    lo, hi = ta - d, tb + d
    # ≥32 samples (step ≲ 5e-4 for the minimum window): the closest confirmed pairs are
    # ~1e-3 apart in t, so a coarse grid can step over both zeros of a tight pair.
    m = 32
    ts = [lo + (hi - lo) * k / m for k in range(m + 1)]
    zs = [Z(t) for t in ts]
    brackets = [(ts[k], ts[k + 1]) for k in range(m) if zs[k] * zs[k + 1] < 0]
    res = {"label": label, "t_a_scan": float(ta), "t_b_scan": float(tb),
           "sign_changes": len(brackets),
           "min_abs_Z_sampled": float(min(abs(z) for z in zs))}
    if len(brackets) != 2:
        # 0 sign changes with tiny |Z|: a near-miss the GPU (error up to ~2e-3) mistook
        # for two zeros.  Not a counterexample: Z just dips close to zero without crossing.
        res["status"] = ("NEAR_MISS" if not brackets and res["min_abs_Z_sampled"] < 5e-3
                         else "NOT_CONFIRMED")
        return res
    # At t ~ 3e12 a 30-digit mpf leaves ~17 digits after the point; ask for 1e-12.
    try:
        roots = [mpmath.findroot(Z, br, solver="illinois", tol=mpmath.mpf(10) ** -24,
                                 verify=False) for br in brackets]
    except Exception as e:                       # never let one pair kill the batch
        res["status"] = f"ERROR: {e}"[:80]
        return res
    if not all(br[0] <= r <= br[1] for r, br in zip(roots, brackets)):
        res["status"] = "ERROR: root left its bracket"
        return res
    r1, r2 = sorted(roots)
    # max |Z| between the zeros: golden-section search on −|Z|
    f = lambda t: -abs(Z(t))
    a, b = r1, r2
    g = (mpmath.sqrt(5) - 1) / 2
    c, e = b - g * (b - a), a + g * (b - a)
    fc, fe = f(c), f(e)
    for _ in range(30):
        if fc < fe:
            b, e, fe = e, c, fc
            c = b - g * (b - a); fc = f(c)
        else:
            a, c, fc = c, e, fe
            e = a + g * (b - a); fe = f(e)
    zmax = -min(fc, fe)
    rho = density(r1)
    res.update(status="CONFIRMED",
               t1=mpmath.nstr(r1, 22), t2=mpmath.nstr(r2, 22),
               gap=float(r2 - r1), unfolded_gap=float(r2 - r1) * rho,
               max_abs_Z_between=float(zmax))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("csv", nargs="?", help="close_pairs CSV from rh_stats.py")
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--out", help="output CSV (default: <input>_confirmed.csv)")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()

    if a.demo:
        # Lehmer (1956): zeros at t ≈ 7005.0629 and 7005.1006
        r = check_pair((7005.06, 7005.10, "lehmer_1956"))
        for k, v in r.items():
            print(f"  {k:18s} {v}")
        return
    if not a.csv:
        ap.error("give a close_pairs CSV, or --demo")

    with open(a.csv) as f:
        rows = list(csv.DictReader(f))[: a.top]
    jobs = [(float(r["t_a"]), float(r["t_b"]), r["chunk_id"]) for r in rows]
    print(f"Checking {len(jobs)} pairs on {a.jobs} processes "
          f"(~4 s per Z evaluation at t≈3e12, ~85 evaluations per pair)…", file=sys.stderr)

    out = a.out or a.csv.replace(".csv", "_confirmed.csv")
    fields = ["label", "status", "sign_changes", "t1", "t2", "gap", "unfolded_gap",
              "max_abs_Z_between", "min_abs_Z_sampled", "t_a_scan", "t_b_scan"]
    n_ok = 0
    with Pool(a.jobs) as pool, open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, r in enumerate(pool.imap(check_pair, jobs), 1):
            w.writerow(r); f.flush()
            n_ok += r["status"] == "CONFIRMED"
            extra = (f"δ={r['unfolded_gap']:.6f} max|Z|={r['max_abs_Z_between']:.3e}"
                     if r["status"] == "CONFIRMED" else f"{r['sign_changes']} sign changes")
            print(f"  [{i}/{len(jobs)}] {r['label']}  {r['status']}  {extra}", file=sys.stderr)
    print(f"{n_ok}/{len(jobs)} confirmed → {out}")


if __name__ == "__main__":
    main()
