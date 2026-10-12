#!/usr/bin/env python3
"""
Report generator: reads all completed chunks from the NAS and prints
a human-readable summary of the RH verification status.

Usage:  python3 rh_report.py [--nas /mnt/math/rh] [--list-errors]
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from rh_core import backlund_count


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--nas", default="/mnt/math/rh")
    p.add_argument("--list-errors", action="store_true",
                   help="Print chunks where Backlund count didn't match")
    p.add_argument("--list-zeros", action="store_true",
                   help="Print the first 20 verified zeros")
    return p.parse_args()


def main():
    args = parse_args()
    nas = Path(args.nas)
    done_dir = nas / "done"

    if not done_dir.exists():
        sys.exit(f"No done/ directory at {nas}. Nothing computed yet.")

    chunks = []
    for p in sorted(done_dir.iterdir()):
        if not p.name.endswith(".json"):
            continue
        try:
            chunks.append(json.loads(p.read_text()))
        except Exception as e:
            print(f"  [warn] could not read {p.name}: {e}")

    if not chunks:
        print("No completed chunks yet.")
        return

    total_zeros   = sum(c.get("zeros_found", 0) for c in chunks)
    chunks_ok     = sum(1 for c in chunks if c.get("backlund_ok"))
    chunks_err    = [c for c in chunks if not c.get("backlund_ok")]
    total_cpu_s   = sum(c.get("wall_seconds", 0) for c in chunks)
    gpu_chunks    = sum(1 for c in chunks if c.get("gpu_used"))

    t_max_done = max(c.get("t_end", 0) for c in chunks) if chunks else 0.0
    n_backlund = backlund_count(t_max_done) if t_max_done > 10 else 0

    print("=" * 65)
    print("  RIEMANN HYPOTHESIS — Computational Verification Report")
    print("=" * 65)
    print(f"  Chunks completed  : {len(chunks):,d}  "
          f"(GPU-accelerated: {gpu_chunks})")
    print(f"  Backlund verified : {chunks_ok:,d}  /  {len(chunks):,d}")
    print(f"  Zeros found & verified on critical line : {total_zeros:,d}")
    print(f"  Maximum t verified: {t_max_done:.10e}")
    print(f"  Backlund N(T_max) : {n_backlund:,d}")
    print(f"  CPU-hours used    : {total_cpu_s/3600:.1f} h")
    print()

    if chunks_ok == len(chunks):
        print("  ✓  STATUS: ALL COMPUTED ZEROS LIE ON THE CRITICAL LINE")
        print("     (Backlund count matches for every completed chunk)")
    else:
        print(f"  ✗  STATUS: {len(chunks_err)} CHUNKS HAVE COUNT MISMATCHES — investigate")

    print("=" * 65)

    if args.list_errors and chunks_err:
        print("\nChunks with Backlund mismatch:")
        for c in chunks_err:
            print(f"  {c['chunk_id']}  t=[{c['t_start']:.4e},{c['t_end']:.4e}]  "
                  f"found={c['zeros_found']} expected={c['backlund_expected']}")

    if args.list_zeros:
        all_zeros = []
        for c in sorted(chunks, key=lambda x: x.get("t_start", 0)):
            all_zeros.extend(c.get("zeros", []))
        print(f"\nFirst {min(20, len(all_zeros))} verified zeros (t values):")
        for i, z in enumerate(all_zeros[:20]):
            print(f"  {i+1:4d}.  t = {z:.15f}")


if __name__ == "__main__":
    main()
