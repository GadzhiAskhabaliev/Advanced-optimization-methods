"""Balanced synthetic datasets. Preprocessing is fit on the training split only."""

from __future__ import annotations

import numpy as np


def _balance_counts(n):
    n0 = n // 2
    return n0, n - n0


def make_logistic(n, rng):
    """a | y ~ N((2y-1)(1, 1), I_2), y in {0, 1}, exactly balanced up to one point."""
    n0, n1 = _balance_counts(n)
    y = np.concatenate([np.zeros(n0), np.ones(n1)])
    means = (2.0 * y - 1.0)[:, None] * np.ones((1, 2))
    features = means + rng.normal(size=(n, 2))
    perm = rng.permutation(n)
    return features[perm], y[perm]


def make_semicircles(n, rng):
    """Noisy semicircles.

    t ~ Uniform[0, π], e ~ N(0, 0.1² I_2),

        y = 0: (cos t, sin t) + e
        y = 1: (1 - cos t, 1/2 - sin t) + e
    """
    n0, n1 = _balance_counts(n)

    def branch(m, label):
        t = rng.uniform(0.0, np.pi, size=m)
        if label == 0:
            features = np.stack([np.cos(t), np.sin(t)], axis=1)
        else:
            features = np.stack([1.0 - np.cos(t), 0.5 - np.sin(t)], axis=1)
        features = features + rng.normal(scale=0.1, size=features.shape)
        return features, np.full(m, float(label))

    a0, y0 = branch(n0, 0)
    a1, y1 = branch(n1, 1)
    features = np.vstack([a0, a1])
    labels = np.concatenate([y0, y1])
    perm = rng.permutation(n)
    return features[perm], labels[perm]


def train_val_test(generator, n_train, n_val, n_test, seed):
    rng = np.random.default_rng(seed)
    splits = []
    for n in (n_train, n_val, n_test):
        features, labels = generator(n, rng)
        splits.append((features, labels))
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = splits
    mu = x_train.mean(axis=0)
    sd = np.maximum(x_train.std(axis=0), 1e-8)

    def apply(features):
        return (features - mu) / sd

    return {
        "x_train": apply(x_train),
        "y_train": y_train,
        "x_val": apply(x_val),
        "y_val": y_val,
        "x_test": apply(x_test),
        "y_test": y_test,
        "x_train_raw": x_train,
        "x_val_raw": x_val,
        "x_test_raw": x_test,
        "mu": mu,
        "sd": sd,
    }


GENERATORS = {
    "logistic": make_logistic,
    "mlp": make_semicircles,
}
