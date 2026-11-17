"""Diagonal covariance of the batch-mean gradient."""

from __future__ import annotations

import numpy as np


def batch_statistics(observations, eps_scale=1e-3, eps_floor=1e-8):
    """Mean g and diagonal covariance S of the batch mean.

    observations has shape (b, p), b ≥ 2, rows are per-example gradient
    observations v_{k,i}, possibly corrupted. With replacement sampling is the
    caller's responsibility.

        bsk_j = 1/(b(b-1)) Σ_i (v_{k,i,j} - g_{k,j})²
        ε = 10^{-3} max{ mean_j bsk_j, 10^{-8} }
        s_j = bsk_j + ε
        S = diag(s)

    Returns g, s, bsk, eps.
    """
    V = np.asarray(observations, dtype=float)
    if V.ndim != 2 or V.shape[0] < 2:
        raise ValueError("need at least two observations")
    b = V.shape[0]
    g = V.mean(axis=0)
    centered = V - g
    bsk = np.sum(centered * centered, axis=0) / (b * (b - 1.0))
    eps = eps_scale * max(float(np.mean(bsk)), eps_floor)
    s = bsk + eps
    return g, s, bsk, eps
