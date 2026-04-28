"""Calibration metrics: ECE, MCE, ACE, Brier, reliability curve.

All functions take two arrays:
  confidences: shape (n,), float in [0, 1]
  correct:     shape (n,), bool / 0-or-1

ECE/MCE use equal-width bins on [0, 1]. ACE uses equal-mass (adaptive)
bins so each bin holds ~n/k samples - ACE handles the long tails of
LLM confidence distributions better than ECE.

Brier is the mean squared error between confidence and correctness; it
decomposes into reliability (calibration) + resolution + uncertainty
(Murphy 1973), so Brier moves with both calibration AND discrimination.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BinResult:
    """One bin in a reliability curve."""

    confidence_mean: float  # mean predicted confidence in this bin
    accuracy: float  # fraction correct in this bin
    count: int  # number of samples in the bin


@dataclass(frozen=True)
class ReliabilityCurve:
    """The full reliability curve: a list of bins + the per-sample arrays."""

    bins: list[BinResult]
    n_samples: int
    n_bins: int


def _validate(confidences: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c = np.asarray(confidences, dtype=np.float64)
    y = np.asarray(correct, dtype=np.float64)
    if c.shape != y.shape:
        raise ValueError(f"shape mismatch: confidences {c.shape}, correct {y.shape}")
    if c.ndim != 1:
        raise ValueError(f"expected 1-D arrays, got {c.ndim}-D")
    if c.size == 0:
        raise ValueError("empty input")
    if (c < 0).any() or (c > 1).any():
        raise ValueError("confidences must lie in [0, 1]")
    return c, y


def reliability_curve(
    confidences: np.ndarray,
    correct: np.ndarray,
    *,
    n_bins: int = 10,
    adaptive: bool = False,
) -> ReliabilityCurve:
    """Bin samples by confidence and compute per-bin accuracy.

    adaptive=False uses equal-width bins on [0, 1] (ECE/MCE style).
    adaptive=True uses equal-mass bins (ACE style) so every bin holds
    approximately n/n_bins samples.
    """
    c, y = _validate(confidences, correct)
    if adaptive:
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(c, quantiles)
        edges[0] = 0.0
        edges[-1] = 1.0
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[BinResult] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # last bin is right-inclusive so confidence == 1.0 lands somewhere
        in_bin = (c >= lo) & (c <= hi) if i == n_bins - 1 else (c >= lo) & (c < hi)
        count = int(in_bin.sum())
        if count == 0:
            bins.append(BinResult(confidence_mean=(lo + hi) / 2, accuracy=0.0, count=0))
            continue
        bins.append(
            BinResult(
                confidence_mean=float(c[in_bin].mean()),
                accuracy=float(y[in_bin].mean()),
                count=count,
            )
        )
    return ReliabilityCurve(bins=bins, n_samples=int(c.size), n_bins=n_bins)


def ece(
    confidences: np.ndarray,
    correct: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (Naeini et al. 2015).

    ECE = sum over bins of (|bin| / N) * |confidence_mean - accuracy|.
    Equal-width bins. Lower is better; 0.0 is perfectly calibrated.
    """
    curve = reliability_curve(confidences, correct, n_bins=n_bins, adaptive=False)
    n = curve.n_samples
    return sum(
        (b.count / n) * abs(b.confidence_mean - b.accuracy) for b in curve.bins if b.count > 0
    )


def mce(
    confidences: np.ndarray,
    correct: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """Maximum Calibration Error: worst-case |confidence - accuracy| across bins.

    Equal-width bins. Useful for safety-critical settings where the
    worst-bin gap matters more than the average.
    """
    curve = reliability_curve(confidences, correct, n_bins=n_bins, adaptive=False)
    return max(
        (abs(b.confidence_mean - b.accuracy) for b in curve.bins if b.count > 0),
        default=0.0,
    )


def ace(
    confidences: np.ndarray,
    correct: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """Adaptive Calibration Error (Nixon et al. 2019).

    Same shape as ECE but with equal-MASS bins instead of equal-width.
    LLM confidence distributions are heavily peaked near the extremes,
    which makes most equal-width bins empty; ACE fixes that bias.
    """
    curve = reliability_curve(confidences, correct, n_bins=n_bins, adaptive=True)
    n = curve.n_samples
    return sum(
        (b.count / n) * abs(b.confidence_mean - b.accuracy) for b in curve.bins if b.count > 0
    )


def brier(confidences: np.ndarray, correct: np.ndarray) -> float:
    """Brier score: mean squared error between confidence and correctness.

    Brier = mean((confidence - correct)^2). Lower is better; 0.0 is
    perfect. Decomposes into reliability + resolution + uncertainty
    (Murphy 1973), so it captures both calibration and discrimination.
    """
    c, y = _validate(confidences, correct)
    return float(np.mean((c - y) ** 2))
