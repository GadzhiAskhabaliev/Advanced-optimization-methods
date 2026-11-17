"""Matched SGD and R-SGD training loops.

A training seed produces the same index sequence and the same gradient noise
for both methods, because those draws do not depend on the parameters. The
objective, the training labels and the validation/test sets are never corrupted.
Only the observed gradients used in the update contain the additive noise.
"""

from __future__ import annotations

import time

import numpy as np

from robust_sgd.models import MODELS
from robust_sgd.solver import robust_step
from robust_sgd.uncertainty import batch_statistics


def learning_rate(alpha0, k):
    return alpha0 / np.sqrt(1.0 + k / 50.0)


def train_model(
    model_name,
    data,
    theta0,
    method,
    alpha0,
    rho,
    sigma,
    steps,
    batch_size,
    train_seed,
):
    spec = MODELS[model_name]
    loss_fn = spec["loss"]
    acc_fn = spec["accuracy"]
    grad_fn = spec["example_gradients"]
    theta = np.array(theta0, dtype=float, copy=True)
    x_train = data["x_train"]
    y_train = data["y_train"]
    n = x_train.shape[0]
    rng = np.random.default_rng(train_seed)

    train_loss = np.empty(steps)
    val_loss = np.empty(steps)
    shrink = []
    angles = []
    zero_steps = 0
    update_seconds = 0.0

    for k in range(steps):
        alpha = learning_rate(alpha0, k)
        idx = rng.integers(0, n, size=batch_size)
        noise = rng.normal(size=(batch_size, theta.shape[0]))
        t0 = time.perf_counter()
        observations = grad_fn(theta, x_train[idx], y_train[idx])
        if sigma != 0.0:
            observations = observations + sigma * noise
        g, s, _, _ = batch_statistics(observations)
        if method == "sgd":
            step = -alpha * g
        elif method == "rsgd":
            step = robust_step(g, s, alpha, rho)
        else:
            raise ValueError(method)
        theta = theta + step
        update_seconds += time.perf_counter() - t0

        gnorm = float(np.linalg.norm(g))
        dnorm = float(np.linalg.norm(step))
        if dnorm == 0.0:
            zero_steps += 1
        elif gnorm > 0.0:
            shrink.append(dnorm / (alpha * gnorm))
            cosine = float(np.dot(step, -g) / (dnorm * gnorm))
            angles.append(float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))))

        train_loss[k] = loss_fn(theta, x_train, y_train)
        val_loss[k] = loss_fn(theta, data["x_val"], data["y_val"])

    return {
        "theta": theta,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "final_train_loss": float(train_loss[-1]),
        "final_val_loss": float(val_loss[-1]),
        "selection_val_loss": float(np.mean(val_loss[-20:])),
        "test_loss": loss_fn(theta, data["x_test"], data["y_test"]),
        "test_accuracy": acc_fn(theta, data["x_test"], data["y_test"]),
        "train_accuracy": acc_fn(theta, x_train, y_train),
        "zero_step_frequency": zero_steps / steps,
        "mean_shrinkage": float(np.mean(shrink)) if shrink else None,
        "mean_angle_deg": float(np.mean(angles)) if angles else None,
        "update_seconds": update_seconds,
        "gradient_evaluations": steps * batch_size,
    }
