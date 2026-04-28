"""Brier score decomposition (Murphy 1973).

The Brier score factors into three components:

  Brier = Reliability - Resolution + Uncertainty

- Reliability   how close the within-bin accuracy is to the within-bin
                forecast confidence. Lower is better. This is the
                calibration component.
- Resolution    how much the bin-wise accuracies vary around the base
                rate. Higher is better. The discrimination component.
- Uncertainty   variance of the base rate itself; depends only on the
                marginal label distribution. Cannot be reduced by the
                forecaster.

Recalibration (e.g. temperature scaling) reduces Reliability without
changing Resolution. Reporting all three separately tells you whether
your model is miscalibrated, undiscriminating, or both.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BrierDecomposition:
    """Murphy 1973 partition of the Brier score."""

    brier: float
    reliability: float
    resolution: float
    uncertainty: float

    def is_consistent(self, tol: float = 1e-6) -> bool:
        """Sanity check: brier ≈ reliability - resolution + uncertainty."""
        recon = self.reliability - self.resolution + self.uncertainty
        return abs(self.brier - recon) < tol


def brier_decomposition(
    confidences: np.ndarray,
    correct: np.ndarray,
    *,
    n_bins: int = 10,
) -> BrierDecomposition:
    """Compute Murphy 1973 decomposition of the Brier score.

    Note: the binned Reliability + Resolution only sum to the true Brier
    when bins are non-empty and confidences are roughly piecewise-constant
    within each bin. For per-sample-exact decomposition, use n_bins = N
    (every sample its own bin) - at that limit Reliability = 0 and
    Resolution = base-rate-variance.
    """
    c = np.asarray(confidences, dtype=np.float64)
    y = np.asarray(correct, dtype=np.float64)
    if c.shape != y.shape or c.ndim != 1:
        raise ValueError("confidences and correct must be 1-D arrays of equal shape")
    if c.size == 0:
        raise ValueError("empty input")

    n = c.size
    base_rate = float(y.mean())
    uncertainty = base_rate * (1.0 - base_rate)
    brier = float(np.mean((c - y) ** 2))

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    reliability = 0.0
    resolution = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (c >= lo) & (c <= hi) if i == n_bins - 1 else (c >= lo) & (c < hi)
        nk = int(in_bin.sum())
        if nk == 0:
            continue
        f_k = float(c[in_bin].mean())  # mean forecast in this bin
        o_k = float(y[in_bin].mean())  # mean outcome in this bin
        reliability += (nk / n) * (f_k - o_k) ** 2
        resolution += (nk / n) * (o_k - base_rate) ** 2

    return BrierDecomposition(
        brier=brier,
        reliability=reliability,
        resolution=resolution,
        uncertainty=uncertainty,
    )
