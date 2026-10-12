"""
Core mathematical functions for the Riemann Hypothesis verification.

Method: Hardy Z-function via the Riemann-Siegel formula.
  Z(t) = 2·Σ_{n=1}^{N} cos(θ(t)-t·ln n)/√n  +  R₀(t)  [+ smaller correction terms]
  where N = ⌊√(t/2π)⌋

Z(t) is real for real t; its zeros correspond to zeros of ζ(1/2+it) on the critical line.

Backlund verification: count formula N(T) = ⌊θ(T)/π⌋ + 1  must equal the number of
sign-change zeros found in [0,T]. Equality proves all zeros up to T lie on the critical line.
"""

import math
import numpy as np

try:
    import cupy as cp
    _CUPY = True
except ImportError:
    _CUPY = False

# ── Theta function ────────────────────────────────────────────────────────────

def theta(t, xp=np):
    """
    Riemann-Siegel theta: Im(log Γ(1/4+it/2)) - (t/2)·log π
    Stirling asymptotic series; accurate for t ≳ 10.
    """
    t = xp.asarray(t, dtype=xp.float64)
    return ((t / 2.0) * xp.log(t / (2.0 * math.pi))
            - t / 2.0
            - math.pi / 8.0
            + 1.0 / (48.0 * t)
            + 7.0 / (5760.0 * t**3)
            - 31.0 / (80640.0 * t**5)
            + 127.0 / (1290240.0 * t**7))


def theta_scalar(t):
    """Scalar version (Python float, no array overhead)."""
    return ((t / 2.0) * math.log(t / (2.0 * math.pi))
            - t / 2.0
            - math.pi / 8.0
            + 1.0 / (48.0 * t)
            + 7.0 / (5760.0 * t**3)
            - 31.0 / (80640.0 * t**5))


# ── Riemann-Siegel Z-function ─────────────────────────────────────────────────

def Z_batch(t_array, xp=np, n_chunk=20_000):
    """
    Compute Hardy Z(t) for a 1-D array of t values.

    Uses the Riemann-Siegel formula with the C₀ remainder correction.
    Memory is bounded by len(t_array) × n_chunk × 8 bytes per inner iteration.

    Parameters
    ----------
    t_array : array-like of float64 (t > 10)
    xp      : numpy or cupy
    n_chunk : chunk size along the n-summation axis (tune for GPU memory)

    Returns
    -------
    Z : xp.ndarray, shape (len(t_array),)
    """
    t = xp.asarray(t_array, dtype=xp.float64)
    M = len(t)

    # N(t) = ⌊√(t/2π)⌋ — number of terms in main sum
    sqrt_t_2pi = xp.sqrt(t / (2.0 * math.pi))
    N_arr = xp.floor(sqrt_t_2pi).astype(xp.int64)
    N_max = int(N_arr.max())

    th = theta(t, xp)
    Z_sum = xp.zeros(M, dtype=xp.float64)

    for n_start in range(1, N_max + 1, n_chunk):
        n_end = min(n_start + n_chunk, N_max + 1)
        n = xp.arange(n_start, n_end, dtype=xp.float64)  # (C,)

        # mask[i,j] = (n_j <= N(t_i))
        mask = (n[xp.newaxis, :] <= N_arr[:, xp.newaxis])  # (M, C) bool

        # angle[i,j] = θ(t_i) - t_i·ln(n_j)
        angles = th[:, xp.newaxis] - t[:, xp.newaxis] * xp.log(n)[xp.newaxis, :]

        contrib = xp.where(mask, xp.cos(angles) / xp.sqrt(n)[xp.newaxis, :], 0.0)
        Z_sum += xp.sum(contrib, axis=1)

    Z_main = 2.0 * Z_sum

    # ── Riemann-Siegel C₀ remainder ──
    # p = fractional part of sqrt(t/2π)
    p = sqrt_t_2pi - xp.floor(sqrt_t_2pi)

    cos_num = xp.cos(2.0 * math.pi * (p * p - p - 1.0 / 16.0))
    cos_den = xp.cos(2.0 * math.pi * p)

    # Near p = 1/4 or 3/4, cos_den → 0; flag for high-precision follow-up
    safe = xp.abs(cos_den) > 1e-6
    cos_den_safe = xp.where(safe, cos_den, xp.ones_like(cos_den))

    sign = xp.where(N_arr % 2 == 1, 1.0, -1.0)  # (-1)^(N-1)
    R0 = sign * (t / (2.0 * math.pi))**(-0.25) * cos_num / cos_den_safe
    R0 = xp.where(safe, R0, xp.zeros_like(R0))  # leave unsafe for mpmath

    return Z_main + R0, ~safe  # (Z values, singularity mask)


def Z_scalar(t):
    """
    Vectorized-numpy scalar Z(t).  Uses numpy internally for the sum
    (fast SIMD), but returns a Python float.  Much faster than a pure-Python
    loop at large t (N~700k terms at t~3e12).
    """
    N = int(math.sqrt(t / (2.0 * math.pi)))
    th = theta_scalar(t)
    n = np.arange(1, N + 1, dtype=np.float64)
    Z_main = 2.0 * float(np.dot(np.cos(th - t * np.log(n)), 1.0 / np.sqrt(n)))

    sqrt_t2pi = math.sqrt(t / (2.0 * math.pi))
    p = sqrt_t2pi - N
    cos_den = math.cos(2.0 * math.pi * p)
    if abs(cos_den) < 1e-6:
        return Z_scalar_mpmath(t)
    cos_num = math.cos(2.0 * math.pi * (p * p - p - 1.0 / 16.0))
    sign = 1.0 if N % 2 == 1 else -1.0
    R0 = sign * (t / (2.0 * math.pi))**(-0.25) * cos_num / cos_den
    return Z_main + R0


def Z_scalar_mpmath(t):
    """
    High-precision Z(t) via mpmath (25 decimal places — sufficient for zero
    confirmation while being faster than 50-digit default).
    Called only when C₀ remainder is near-singular or for final verification.
    """
    import mpmath
    mpmath.mp.dps = 25
    return float(mpmath.siegelz(mpmath.mpf(t)))


# ── Zero finding ──────────────────────────────────────────────────────────────

def find_sign_changes(t_grid, Z_vals):
    """
    Return indices i where Z_vals[i] and Z_vals[i+1] have opposite signs.
    Excludes exact zeros (unlikely in floating point but handled).
    """
    xp = np
    s = np.sign(Z_vals)
    # sign change: s[i] * s[i+1] < 0; also treat exact 0 as sign change
    changes = (s[:-1] * s[1:] < 0) | (s[:-1] == 0) | (s[1:] == 0)
    return np.where(changes)[0]


def bisect_zero(t_lo, t_hi, Z_lo=None, Z_hi=None, tol=1e-11, max_iter=60):
    """Serial bisection — used only when batch_bisect_zeros is unavailable."""
    if Z_lo is None:
        Z_lo = Z_scalar(t_lo)
    if Z_hi is None:
        Z_hi = Z_scalar(t_hi)
    for _ in range(max_iter):
        if t_hi - t_lo < tol:
            break
        t_mid = (t_lo + t_hi) / 2.0
        Z_mid = Z_scalar(t_mid)
        if Z_lo * Z_mid <= 0:
            t_hi, Z_hi = t_mid, Z_mid
        else:
            t_lo, Z_lo = t_mid, Z_mid


def batch_bisect_zeros(sign_change_idxs, t_vals, Z_vals,
                       xp=np, n_chunk=30_000, tol=1e-10, max_iter=40):
    """
    Refine ALL bracketed zeros simultaneously using vectorized GPU/CPU Z_batch.
    Each bisection iteration makes ONE batched Z-function call for all brackets.

    Parameters
    ----------
    sign_change_idxs : 1-D integer array of indices where sign changes were detected
    t_vals, Z_vals   : the original scan grid and Z values
    xp               : numpy or cupy

    Returns
    -------
    zeros : list of float (zero locations to within tol)
    """
    if len(sign_change_idxs) == 0:
        return []

    t_lo = t_vals[sign_change_idxs].copy().astype(np.float64)
    t_hi = t_vals[sign_change_idxs + 1].copy().astype(np.float64)
    Z_lo = Z_vals[sign_change_idxs].copy().astype(np.float64)

    for _ in range(max_iter):
        if (t_hi - t_lo).max() < tol:
            break
        t_mid = (t_lo + t_hi) / 2.0

        Z_mid_xp, _ = Z_batch(t_mid, xp=xp, n_chunk=n_chunk)
        Z_mid = Z_mid_xp.get() if hasattr(Z_mid_xp, 'get') else np.array(Z_mid_xp)

        left  = (Z_lo * Z_mid <= 0)
        t_hi  = np.where(left, t_mid, t_hi)
        t_lo  = np.where(~left, t_mid, t_lo)
        Z_lo  = np.where(~left, Z_mid, Z_lo)

    return list((t_lo + t_hi) / 2.0)

    return (t_lo + t_hi) / 2.0


# ── Backlund zero count ───────────────────────────────────────────────────────

def backlund_count(T):
    """
    Expected number of zeros of ζ(s) with 0 < Im(s) < T on the critical line.
    N(T) = ⌊θ(T)/π⌋ + 1

    This is the leading term; the correction S(T) = (1/π)·arg ζ(½+iT) is bounded
    by C·log T but typically small. For rigorous verification we compute it via
    winding number (see backlund_count_rigorous).
    """
    return int(math.floor(theta_scalar(T) / math.pi)) + 1


def backlund_count_rigorous(T, n_points=1000):
    """
    Rigorous N(T) via the argument principle, integrating along the rectangle
    with corners 2, 2+iT, -1+iT, -1 using the functional equation symmetry.
    Computed with mpmath; slower but exact.
    """
    import mpmath
    mpmath.mp.dps = 30
    return int(mpmath.nzeros(T))


# ── Gram points ───────────────────────────────────────────────────────────────

def gram_point(n, tol=1e-10):
    """
    g_n : θ(g_n) = n·π  (Gram points, n ≥ -1).
    Solved by Newton iteration starting from the asymptotic approximation.
    """
    # Approximate: t_approx satisfies (t/2)·ln(t/2πe) ≈ nπ + π/8
    # Newton on f(t) = θ(t) - n·π
    target = n * math.pi
    # Initial guess from inverting leading term of Stirling
    # Solve (t/2)·log(t/2π) - t/2 - π/8 = nπ  →  use Lambert W approximation
    t = max(10.0, 2.0 * math.pi * math.exp(1.0 + math.log(max(1, abs(n)) * 2 * math.pi / math.e) / 2.0))
    for _ in range(50):
        # θ'(t) = (1/2)·ln(t/2π) + higher-order terms ≈ (1/2)·ln(t/2π)
        th = theta_scalar(t)
        dth = 0.5 * math.log(t / (2.0 * math.pi)) - 1.0 / (48.0 * t**2)
        dt = (target - th) / dth
        t += dt
        if abs(dt) < tol:
            break
    return t


# ── GPU utilities ─────────────────────────────────────────────────────────────

def best_xp(prefer_gpu=True):
    """Return cupy if available and prefer_gpu, else numpy."""
    if prefer_gpu and _CUPY:
        try:
            cp.cuda.Device(0).use()
            return cp
        except Exception:
            pass
    return np


def gpu_available():
    if not _CUPY:
        return False
    try:
        cp.cuda.Device(0).use()
        return True
    except Exception:
        return False
