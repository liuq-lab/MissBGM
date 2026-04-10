"""Synthetic MNAR datasets used by MissBGM."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _calibrate_logit_intercept(scores, target_observed_rate, max_iter=80):
    lower, upper = -20.0, 20.0
    for _ in range(max_iter):
        midpoint = 0.5 * (lower + upper)
        observed_rate = _sigmoid(scores + midpoint).mean()
        if observed_rate > target_observed_rate:
            upper = midpoint
        else:
            lower = midpoint
    return np.float32(0.5 * (lower + upper))


def simulate_mnar_factor_data(
    n_samples=5000,
    x_dim=50,
    z_dim=5,
    missing_rate=0.3,
    seed=123,
):
    """Simulate a nonlinear latent-factor MNAR dataset."""

    if not 0.0 < missing_rate < 1.0:
        raise ValueError("missing_rate must lie strictly between 0 and 1.")

    rng = np.random.default_rng(seed)
    z_true = rng.normal(size=(n_samples, z_dim)).astype(np.float32)

    decoder_w1 = rng.normal(scale=0.7, size=(z_dim, 4 * z_dim)).astype(np.float32)
    decoder_w2 = rng.normal(scale=0.5, size=(4 * z_dim, x_dim)).astype(np.float32)
    decoder_bias = rng.normal(scale=0.15, size=(x_dim,)).astype(np.float32)

    hidden = np.tanh(z_true @ decoder_w1 / np.sqrt(max(z_dim, 1.0))).astype(np.float32)
    mean_x = (hidden @ decoder_w2 / np.sqrt(max(4 * z_dim, 1.0)) + decoder_bias[None, :]).astype(np.float32)

    noise_gate = _sigmoid(z_true @ rng.normal(scale=0.5, size=(z_dim,)).astype(np.float32))
    noise_scale = (0.15 + 0.25 * noise_gate)[:, None].astype(np.float32)
    x_full = mean_x + noise_scale * rng.normal(size=(n_samples, x_dim)).astype(np.float32)

    coupling = rng.normal(scale=0.3, size=(x_dim, x_dim)).astype(np.float32)
    coupling = coupling / np.sqrt(max(x_dim, 1.0))
    shared_signal = np.tanh(x_full @ coupling).astype(np.float32)
    missingness_scores = (0.6 * x_full + 0.4 * shared_signal).astype(np.float32)

    intercept = _calibrate_logit_intercept(missingness_scores, 1.0 - missing_rate)
    observation_prob = _sigmoid(missingness_scores + intercept).astype(np.float32)
    mask = rng.binomial(1, observation_prob).astype(np.float32)

    for row_idx in range(n_samples):
        if mask[row_idx].sum() == 0:
            mask[row_idx, int(rng.integers(0, x_dim))] = 1.0
    for col_idx in range(x_dim):
        if mask[:, col_idx].sum() == 0:
            mask[int(rng.integers(0, n_samples)), col_idx] = 1.0

    x_obs = x_full.copy()
    x_obs[mask == 0.0] = np.nan
    return {
        "x_full": x_full.astype(np.float32),
        "x_obs": x_obs.astype(np.float32),
        "mask": mask.astype(np.float32),
        "z_true": z_true.astype(np.float32),
        "missing_rate": float(1.0 - mask.mean()),
    }


def simulate_mnar_oracle_data(
    n_samples=5000,
    x_dim=50,
    n_anchor=5,
    missing_rate=0.2,
    alpha=0.05,
    seed=123,
):
    """Simulate a self-masked Gaussian benchmark with oracle intervals."""

    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if x_dim <= 1:
        raise ValueError("x_dim must be at least 2.")
    if not (1 <= n_anchor < x_dim):
        raise ValueError("n_anchor must satisfy 1 <= n_anchor < x_dim.")
    if not (0.0 < missing_rate < 1.0):
        raise ValueError("missing_rate must lie strictly between 0 and 1.")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie strictly between 0 and 1.")

    rng = np.random.default_rng(seed)
    n_target = x_dim - n_anchor

    anchors = rng.normal(size=(n_samples, n_anchor)).astype(np.float32)
    beta = rng.normal(scale=0.4, size=(n_anchor, n_target)).astype(np.float32)
    bias = rng.normal(scale=0.3, size=n_target).astype(np.float32)
    sigma = rng.uniform(0.6, 1.2, size=n_target).astype(np.float32)

    mu = (anchors @ beta + bias).astype(np.float32)
    targets = (mu + rng.normal(size=(n_samples, n_target)).astype(np.float32) * sigma).astype(np.float32)

    kappa = norm.ppf(1.0 - missing_rate)
    target_mask = (targets <= mu + sigma * kappa).astype(np.float32)
    mask = np.concatenate([np.ones((n_samples, n_anchor), dtype=np.float32), target_mask], axis=1)

    x_full = np.concatenate([anchors, targets], axis=1).astype(np.float32)
    x_obs = x_full.copy()
    x_obs[mask == 0.0] = np.nan

    q_lo, q_hi = alpha / 2.0, 1.0 - alpha / 2.0
    phi_k = norm.cdf(kappa)
    p_lo = phi_k + q_lo * (1.0 - phi_k)
    p_hi = phi_k + q_hi * (1.0 - phi_k)
    lo = (mu + sigma * norm.ppf(p_lo)).astype(np.float32)
    hi = (mu + sigma * norm.ppf(p_hi)).astype(np.float32)

    x_miss_intervals = np.zeros((n_samples, x_dim, 2), dtype=np.float32)
    x_miss_intervals[:, n_anchor:, 0] = np.where(target_mask == 0.0, lo, 0.0)
    x_miss_intervals[:, n_anchor:, 1] = np.where(target_mask == 0.0, hi, 0.0)

    return {
        "x_full": x_full,
        "x_obs": x_obs.astype(np.float32),
        "mask": mask.astype(np.float32),
        "missing_rate": float(1.0 - mask.mean()),
        "x_miss_intervals": x_miss_intervals.astype(np.float32),
    }
