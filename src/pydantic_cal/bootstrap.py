"""Bootstrap confidence intervals + paired tests for calibration metrics.

Single number is a point estimate; with bootstrap CIs you can claim
"this fix moved ECE from 0.12 [0.10, 0.14] to 0.04 [0.03, 0.05]" and
the non-overlapping intervals carry the argument.

Paired bootstrap is the right tool for "did calibration improve" because
each sample has a before-pair and an after-pair (same input, two
predictions). Naive two-sample bootstrap throws away that pairing and
under-powers the test.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapCI:
    """Bootstrap confidence interval result."""

    point: float
    lower: float
    upper: float
    n_resamples: int
    confidence: float

    def __str__(self) -> str:
        return f"{self.point:.4f} [{self.lower:.4f}, {self.upper:.4f}] (n={self.n_resamples})"


def bootstrap_ci(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    rng: np.random.Generator | None = None,
) -> BootstrapCI:
    """Percentile bootstrap CI for a one-sample statistic.

    `values` is the observation array; `statistic` reduces it to one
    number. Resampling is with replacement, sample size n.
    """
    v = np.asarray(values)
    if v.size == 0:
        raise ValueError("empty input")
    rng = rng or np.random.default_rng()
    n = v.shape[0]
    samples = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        samples[i] = statistic(v[idx])
    alpha = 1.0 - confidence
    lo = float(np.quantile(samples, alpha / 2))
    hi = float(np.quantile(samples, 1 - alpha / 2))
    return BootstrapCI(
        point=float(statistic(v)),
        lower=lo,
        upper=hi,
        n_resamples=n_resamples,
        confidence=confidence,
    )


def paired_bootstrap_diff(
    before: np.ndarray,
    after: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    rng: np.random.Generator | None = None,
) -> BootstrapCI:
    """Paired bootstrap CI on (statistic(after) - statistic(before)).

    Resamples PAIR INDICES, preserving (before_i, after_i) coupling.
    Use this for before/after calibration: same inputs, two predictions.
    """
    b = np.asarray(before)
    a = np.asarray(after)
    if b.shape != a.shape:
        raise ValueError(f"shape mismatch: before {b.shape}, after {a.shape}")
    if b.size == 0:
        raise ValueError("empty input")
    rng = rng or np.random.default_rng()
    n = b.shape[0]
    samples = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        samples[i] = statistic(a[idx]) - statistic(b[idx])
    alpha = 1.0 - confidence
    lo = float(np.quantile(samples, alpha / 2))
    hi = float(np.quantile(samples, 1 - alpha / 2))
    return BootstrapCI(
        point=float(statistic(a) - statistic(b)),
        lower=lo,
        upper=hi,
        n_resamples=n_resamples,
        confidence=confidence,
    )
