"""Specialized solver for the robust step.

The step solves

    d = argmin_d  g·d + ρ sqrt(dᵀ S d) + ||d||² / (2α)

with S = diag(s), s > 0. The objective is strongly convex, so the minimizer
is unique. It is d = 0 when gᵀ S⁻¹ g ≤ ρ² (including g = 0). Otherwise the
stationarity condition

    g + ρ S d / sqrt(dᵀ S d) + d / α = 0

reduces, for t = ρ / sqrt(dᵀ S d) > 0, to the scalar equation

    sum_j s_j g_j² t² / (1/α + t s_j)² = ρ²,

which is strictly increasing in t from 0 to gᵀ S⁻¹ g. A bisection finds the
unique root. Each trial is O(p), and the number of trials is
O(log(t_max / tol)), so one step costs O(p log(1/tol)).
"""

from __future__ import annotations

import numpy as np


def robust_step(g, s, alpha, rho, tol=1e-12, max_iter=80):
    """Return the robust displacement d. Shape matches g.

    Stopping criterion: the bracket on t has relative width at most `tol`,
    or `max_iter` bisections have been performed. `tol` is relative to
    max(1, t_hi) and is also applied to the absolute bracket width.
    """
    g = np.asarray(g, dtype=float)
    s = np.asarray(s, dtype=float)
    if alpha <= 0:
        raise ValueError("step size alpha must be positive")
    if np.any(s <= 0):
        raise ValueError("diagonal covariance entries must be positive")
    if rho < 0:
        raise ValueError("radius rho must be nonnegative")
    if rho == 0.0:
        return -alpha * g

    quad = float(np.sum((g * g) / s))
    if quad <= rho * rho:
        return np.zeros_like(g)

    inv_alpha = 1.0 / alpha

    def model_quad(t):
        denom = inv_alpha + t * s
        return float(np.sum(s * (g * g) * (t * t) / (denom * denom)))

    t_lo = 0.0
    t_hi = 1.0
    f_hi = model_quad(t_hi)
    doubles = 0
    while f_hi < rho * rho and doubles < 80:
        t_hi *= 2.0
        f_hi = model_quad(t_hi)
        doubles += 1
    if f_hi < rho * rho:
        # Numerically indistinguishable from the zero-step region.
        return np.zeros_like(g)

    target = rho * rho
    for _ in range(max_iter):
        t_mid = 0.5 * (t_lo + t_hi)
        if model_quad(t_mid) < target:
            t_lo = t_mid
        else:
            t_hi = t_mid
        width = t_hi - t_lo
        if width <= tol * max(1.0, t_hi) and width <= 1e-10:
            break

    t = 0.5 * (t_lo + t_hi)
    return -g / (inv_alpha + t * s)


def isotropic_closed_form(g, s_scalar, alpha, rho):
    """Closed form for S = s I, s > 0: d = -α (1 - ρ√s / ||g||)_+ g.

    The zero vector is returned when g = 0, which is the case left open by
    the nonzero-g derivation. Then gᵀ S⁻¹ g = 0 ≤ ρ², so the robust step vanishes.
    """
    g = np.asarray(g, dtype=float)
    nrm = float(np.linalg.norm(g))
    if nrm == 0.0 or rho * np.sqrt(s_scalar) >= nrm:
        return np.zeros_like(g)
    shrink = 1.0 - rho * np.sqrt(s_scalar) / nrm
    return -alpha * shrink * g


def stationarity_residual(d, g, s, alpha, rho):
    """Residual of g + ρ S d / sqrt(dᵀ S d) + d/α for a nonzero step."""
    d = np.asarray(d, dtype=float)
    g = np.asarray(g, dtype=float)
    s = np.asarray(s, dtype=float)
    q = float(np.sqrt(np.sum(s * d * d)))
    if q == 0.0:
        raise ValueError("stationarity residual is for a nonzero step")
    return g + rho * (s * d) / q + d / alpha
