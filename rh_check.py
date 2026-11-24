#!/usr/bin/env python3
"""
Validate GPU Z(t) values against mpmath.siegelz (independent, arbitrary precision).

  RH_DUMP=/tmp/zdump.txt ./rh_worker_cuda < job.json
  python3 rh_check.py /tmp/zdump.txt --n 40

The dump holds "tc jc h" then "j Z_gpu" lines; t_j = tc + (j − jc)·h exactly
(tc and h are the doubles the worker used).
"""

import argparse
import os
import sys
from multiprocessing import Pool

import mpmath


def ref(args):
    tc, jc, h, j = args
    mpmath.mp.dps = 30
    t = mpmath.mpf(tc) + (j - jc) * mpmath.mpf(h)
    return float(mpmath.siegelz(t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--n", type=int, default=40, help="points to check (spread evenly)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    a = ap.parse_args()

    lines = open(a.dump).read().split("\n")
    tc_s, jc_s, h_s = lines[0].split()
    tc, jc, h = float(tc_s), int(jc_s), float(h_s)
    pts = [(int(l.split()[0]), float(l.split()[1])) for l in lines[1:] if l.strip()]
    step = max(1, len(pts) // a.n)
    pts = pts[::step][: a.n]

    with Pool(a.jobs) as pool:
        refs = pool.map(ref, [(tc, jc, h, j) for j, _ in pts])

    worst = 0.0
    print(f"{'j':>10} {'position':>9} {'Z_gpu':>12} {'Z_mpmath':>12} {'|diff|':>10}")
    for (j, zg), zr in zip(pts, refs):
        d = abs(zg - zr)
        worst = max(worst, d)
        print(f"{j:>10} {j / (2 * jc):9.3f} {zg:12.6f} {zr:12.6f} {d:10.2e}")
    print(f"\nmax |Z_gpu − Z_mpmath| over {len(pts)} points: {worst:.3e}")
    sys.exit(0 if worst < 1e-3 else 1)


if __name__ == "__main__":
    main()
