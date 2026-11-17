"""Logistic regression and a 33-parameter two-layer network.

Both models emit a logit t. The loss is binary cross-entropy from logits,

    ℓ(t, y) = log(1 + exp(t)) - y t,

implemented with a stable softplus. Classification uses the threshold t = 0.
"""

from __future__ import annotations

import numpy as np

HIDDEN = 8
LOGISTIC_DIM = 3
MLP_DIM = HIDDEN * 2 + HIDDEN + HIDDEN + 1  # 33


def sigmoid(t):
    t = np.asarray(t, dtype=float)
    out = np.empty_like(t, dtype=float)
    pos = t >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-np.clip(t[pos], 0, None)))
    # t < 0: exp(t) is in (0, 1], so the expression cannot overflow.
    exp_t = np.exp(t[~pos])
    out[~pos] = exp_t / (1.0 + exp_t)
    return out


def softplus(t):
    t = np.asarray(t, dtype=float)
    out = np.empty_like(t, dtype=float)
    pos = t > 0
    out[pos] = t[pos] + np.log1p(np.exp(-np.clip(t[pos], None, 40.0)))
    out[~pos] = np.log1p(np.exp(np.clip(t[~pos], -40.0, None)))
    return out


def bce_logits(t, y):
    y = np.asarray(y, dtype=float)
    return softplus(t) - y * t


def mean_bce(t, y):
    return float(np.mean(bce_logits(t, y)))


def accuracy_from_logits(t, y):
    pred = (np.asarray(t) >= 0.0).astype(float)
    return float(np.mean(pred == np.asarray(y, dtype=float)))


def logistic_logits(theta, A):
    return A @ theta[:2] + theta[2]


def logistic_loss(theta, A, y):
    return mean_bce(logistic_logits(theta, A), y)


def logistic_accuracy(theta, A, y):
    return accuracy_from_logits(logistic_logits(theta, A), y)


def logistic_example_gradients(theta, A, y):
    """Per-example gradients, shape (n, 3), layout [w0, w1, c]."""
    resid = sigmoid(logistic_logits(theta, A)) - np.asarray(y, dtype=float)
    return np.concatenate([resid[:, None] * A, resid[:, None]], axis=1)


def unpack_mlp(theta):
    theta = np.asarray(theta, dtype=float)
    W1 = theta[:16].reshape(HIDDEN, 2)
    b1 = theta[16:24]
    w2 = theta[24:32]
    b2 = float(theta[32])
    return W1, b1, w2, b2


def pack_mlp(W1, b1, w2, b2):
    return np.concatenate(
        [np.asarray(W1, dtype=float).ravel(), np.asarray(b1, dtype=float).ravel(), np.asarray(w2, dtype=float).ravel(), np.atleast_1d(np.asarray(b2, dtype=float)).ravel()]
    )


def mlp_logits(theta, A):
    W1, b1, w2, b2 = unpack_mlp(theta)
    hidden = np.tanh(A @ W1.T + b1)
    return hidden @ w2 + b2


def mlp_loss(theta, A, y):
    return mean_bce(mlp_logits(theta, A), y)


def mlp_accuracy(theta, A, y):
    return accuracy_from_logits(mlp_logits(theta, A), y)


def mlp_example_gradients(theta, A, y):
    """Per-example gradients, shape (n, 33).

    Layout is the concatenation of W1 (row-major), b1, w2, b2.
    """
    W1, b1, w2, b2 = unpack_mlp(theta)
    z = A @ W1.T + b1
    hidden = np.tanh(z)
    resid = sigmoid(hidden @ w2 + b2) - np.asarray(y, dtype=float)
    grad_w2 = resid[:, None] * hidden
    grad_b2 = resid[:, None]
    dhidden = resid[:, None] * w2
    dz = dhidden * (1.0 - hidden * hidden)
    grad_W1 = (dz[:, :, None] * A[:, None, :]).reshape(A.shape[0], 16)
    grad_b1 = dz
    return np.concatenate([grad_W1, grad_b1, grad_w2, grad_b2], axis=1)


def init_logistic(rng):
    return rng.normal(0.0, 0.01, size=LOGISTIC_DIM)


def init_mlp(rng):
    """Glorot normal on both layers, biases at zero."""
    W1 = rng.normal(0.0, np.sqrt(2.0 / (2.0 + HIDDEN)), size=(HIDDEN, 2))
    b1 = np.zeros(HIDDEN)
    w2 = rng.normal(0.0, np.sqrt(2.0 / (HIDDEN + 1.0)), size=HIDDEN)
    b2 = np.zeros(1)
    return pack_mlp(W1, b1, w2, b2)


MODELS = {
    "logistic": {
        "dim": LOGISTIC_DIM,
        "loss": logistic_loss,
        "accuracy": logistic_accuracy,
        "logits": logistic_logits,
        "example_gradients": logistic_example_gradients,
        "init": init_logistic,
    },
    "mlp": {
        "dim": MLP_DIM,
        "loss": mlp_loss,
        "accuracy": mlp_accuracy,
        "logits": mlp_logits,
        "example_gradients": mlp_example_gradients,
        "init": init_mlp,
    },
}
