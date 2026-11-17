"""Numerical checks for the robust step, gradients and the covariance estimator.

Run from the repository root: python checks.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from robust_sgd.models import (
    MLP_DIM,
    init_logistic,
    init_mlp,
    logistic_example_gradients,
    logistic_loss,
    mlp_example_gradients,
    mlp_loss,
)
from robust_sgd.solver import isotropic_closed_form, robust_step, stationarity_residual
from robust_sgd.train import train_model
from robust_sgd.uncertainty import batch_statistics

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def _rel(a, b):
    denom = max(1e-12, float(np.linalg.norm(b)))
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / denom)


def finite_difference_gradient(loss_fn, theta, features, labels, eps=1e-6):
    gradient = np.zeros_like(theta)
    for j in range(theta.size):
        step = np.zeros_like(theta)
        step[j] = eps
        gradient[j] = (
            loss_fn(theta + step, features, labels) - loss_fn(theta - step, features, labels)
        ) / (2.0 * eps)
    return gradient


def robust_objective(d, g, s, alpha, rho):
    return float(
        np.dot(g, d) + rho * np.sqrt(np.sum(s * d * d)) + np.dot(d, d) / (2.0 * alpha)
    )


def check_gradients():
    rng = np.random.default_rng(11)
    features = rng.normal(size=(25, 2))
    labels = rng.integers(0, 2, size=25).astype(float)
    theta_log = init_logistic(rng) + rng.normal(scale=0.4, size=3)
    analytic = logistic_example_gradients(theta_log, features, labels).mean(axis=0)
    numeric = finite_difference_gradient(logistic_loss, theta_log, features, labels)
    log_err = _rel(analytic, numeric)

    theta_mlp = init_mlp(rng)
    analytic_mlp = mlp_example_gradients(theta_mlp, features, labels).mean(axis=0)
    numeric_mlp = finite_difference_gradient(mlp_loss, theta_mlp, features, labels)
    mlp_err = _rel(analytic_mlp, numeric_mlp)
    assert theta_mlp.size == MLP_DIM == 33
    assert log_err < 1e-5, log_err
    assert mlp_err < 1e-5, mlp_err
    return {"logistic_relative_error": log_err, "mlp_relative_error": mlp_err}


def check_solver():
    rng = np.random.default_rng(23)
    records = []
    max_form_err = 0.0
    max_zero_radius_err = 0.0
    max_residual = 0.0
    max_scipy_gap = 0.0

    g0 = rng.normal(size=4)
    s0 = np.exp(rng.normal(size=4))
    assert np.all(robust_step(np.zeros(4), s0, alpha=0.3, rho=1.0) == 0.0)
    assert np.all(isotropic_closed_form(np.zeros(4), s_scalar=0.4, alpha=0.3, rho=1.0) == 0.0)

    for _ in range(25):
        p = int(rng.integers(2, 12))
        g = rng.normal(size=p)
        s = np.exp(rng.normal(scale=0.5, size=p))
        alpha = float(np.exp(rng.normal()))
        rho = float(rng.uniform(0.0, 3.0))
        step = robust_step(g, s, alpha, rho)
        sgd = -alpha * g
        radius0 = robust_step(g, s, alpha, 0.0)
        max_zero_radius_err = max(max_zero_radius_err, _rel(radius0, sgd))

        quad = float(np.sum(g * g / s))
        if quad <= rho * rho:
            assert np.all(step == 0.0)
        else:
            residual = stationarity_residual(step, g, s, alpha, rho)
            scale = max(1.0, float(np.linalg.norm(g)))
            max_residual = max(max_residual, float(np.linalg.norm(residual)) / scale)

        # Independent check against a derivative-free-ish smooth solver.
        def objective(d):
            return robust_objective(d, g, s, alpha, rho)

        best = minimize(objective, np.zeros(p), method="L-BFGS-B", options={"ftol": 1e-15})
        gap = objective(step) - float(best.fun)
        max_scipy_gap = max(max_scipy_gap, gap)

        s_iso = float(np.exp(rng.normal()))
        iso = robust_step(g, np.full(p, s_iso), alpha, rho)
        closed = isotropic_closed_form(g, s_iso, alpha, rho)
        max_form_err = max(max_form_err, float(np.linalg.norm(iso - closed)))
        records.append(gap)

    assert max_zero_radius_err == 0.0
    assert max_form_err < 1e-8, max_form_err
    assert max_residual < 1e-6, max_residual
    assert max_scipy_gap < 1e-5, max_scipy_gap
    return {
        "zero_radius_max_relative_error": max_zero_radius_err,
        "isotropic_max_absolute_error": max_form_err,
        "stationarity_max_relative_residual": max_residual,
        "scipy_objective_gap": max_scipy_gap,
    }


def check_rho0_matches_sgd():
    rng = np.random.default_rng(5)
    from robust_sgd.data import train_val_test

    data = train_val_test(
        __import__("robust_sgd.data", fromlist=["make_logistic"]).make_logistic,
        200,
        80,
        80,
        seed=5,
    )
    theta0 = init_logistic(rng)
    sgd = train_model("logistic", data, theta0, "sgd", 0.4, 0.0, 0.2, 30, 16, train_seed=9)
    rsgd = train_model("logistic", data, theta0, "rsgd", 0.4, 0.0, 0.2, 30, 16, train_seed=9)
    err = float(np.max(np.abs(sgd["train_loss"] - rsgd["train_loss"])))
    assert err == 0.0, err
    assert np.array_equal(sgd["theta"], rsgd["theta"])
    return {"max_train_loss_difference": err}


def check_batch_size():
    """Covariance of the batch mean on the noisy quadratic scales as 1/b."""
    rng = np.random.default_rng(7)
    theta = np.array([2.0, 2.0])
    grad = np.array([theta[0], 4.0 * theta[1]])
    scale = np.array([2.0, 0.5])  # columns of A
    sigma = 1.0
    batches = [2, 4, 8, 16, 32, 64, 128, 256]
    repeats = 2000
    mean_bsk = []
    for b in batches:
        estimates = []
        for _ in range(repeats):
            noise = rng.normal(size=(b, 2))
            observations = grad + sigma * noise * scale
            _, _, bsk, _ = batch_statistics(observations)
            estimates.append(bsk)
        mean_bsk.append(np.mean(estimates, axis=0))
    mean_bsk = np.asarray(mean_bsk)
    # E[bsk_j] = σ² A_jj² / b.
    expected = np.array([(sigma * scale[0]) ** 2, (sigma * scale[1]) ** 2])
    slopes = []
    for j in range(2):
        slope = np.polyfit(np.log(batches), np.log(mean_bsk[:, j]), 1)[0]
        slopes.append(float(slope))
        assert abs(slope + 1.0) < 0.05, slope
        ratio = mean_bsk[:, j] * np.asarray(batches) / expected[j]
        # b = 2 has one degree of freedom, so the Monte Carlo mean is noisier.
        assert np.max(np.abs(ratio[3:] - 1.0)) < 0.05, ratio

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.loglog(batches, mean_bsk[:, 0], "o-", label="coordinate 1 (A_11=2)")
    ax.loglog(batches, mean_bsk[:, 1], "s-", label="coordinate 2 (A_22=1/2)")
    ref_b = np.asarray(batches, dtype=float)
    ax.loglog(ref_b, expected[0] / ref_b, "--", color="0.4", label="variance / b")
    ax.loglog(ref_b, expected[1] / ref_b, "--", color="0.4")
    ax.set_xlabel("batch size b")
    ax.set_ylabel("mean diagonal covariance estimate")
    ax.set_title("Batch-mean covariance scales as 1/b")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "batch_size.png", dpi=160)
    plt.close(fig)
    return {"loglog_slopes": slopes, "mean_bsk": mean_bsk.tolist(), "batch_sizes": batches}


def illustrate_quadratic():
    # The quadratic gradient does not depend on a data point. Observations are
    # the true gradient plus anisotropic Gaussian noise, which is the illustration
    # in the assignment. The same solver and covariance estimator are used.
    from robust_sgd.solver import robust_step as solve
    from robust_sgd.uncertainty import batch_statistics as stats

    def run(method, rho, sigma, seed):
        theta = np.array([2.0, 2.0])
        rng = np.random.default_rng(seed)
        losses = []
        zeros = 0
        A = np.diag([2.0, 0.5])
        steps = 200
        batch = 16
        alpha0 = 0.2
        for k in range(steps):
            alpha = alpha0 / np.sqrt(1.0 + k / 50.0)
            grad = np.array([theta[0], 4.0 * theta[1]])
            noise = rng.normal(size=(batch, 2))
            observations = grad + sigma * noise @ A
            g, s, _, _ = stats(observations)
            step = -alpha * g if method == "sgd" else solve(g, s, alpha, rho)
            if np.all(step == 0.0):
                zeros += 1
            theta = theta + step
            losses.append(0.5 * (theta[0] ** 2 + 4.0 * theta[1] ** 2))
        return np.asarray(losses), zeros / steps, theta

    sgd_loss, sgd_zero, sgd_theta = run("sgd", 0.0, 1.0, 0)
    r_loss, r_zero, r_theta = run("rsgd", 1.5, 1.0, 0)
    huge_loss, huge_zero, huge_theta = run("rsgd", 50.0, 1.0, 0)

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.plot(sgd_loss, label="SGD")
    ax.plot(r_loss, label=r"R-SGD, $\rho=1.5$")
    ax.plot(huge_loss, label=r"R-SGD, $\rho=50$")
    ax.set_xlabel("update")
    ax.set_ylabel("objective")
    ax.set_title("Noisy quadratic illustration")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "quadratic.png", dpi=160)
    plt.close(fig)
    assert huge_zero > 0.5
    return {
        "sgd_final": float(sgd_loss[-1]),
        "rsgd_final": float(r_loss[-1]),
        "large_rho_final": float(huge_loss[-1]),
        "sgd_zero_frequency": sgd_zero,
        "rsgd_zero_frequency": r_zero,
        "large_rho_zero_frequency": huge_zero,
        "sgd_theta": sgd_theta.tolist(),
        "rsgd_theta": r_theta.tolist(),
        "large_rho_theta": huge_theta.tolist(),
    }


def main():
    report = {
        "gradients": check_gradients(),
        "solver": check_solver(),
        "rho0_matches_sgd": check_rho0_matches_sgd(),
        "batch_size": check_batch_size(),
        "quadratic": illustrate_quadratic(),
    }
    out = ROOT / "results" / "checks.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
