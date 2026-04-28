"""Smooth ECE (Blasiok et al. 2023) tests."""

from __future__ import annotations

import numpy as np
import pytest

from pydantic_cal import ece, smece


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def test_perfect_calibration_is_small() -> None:
    # When p(y=1 | confidence=c) = c, smECE -> 0 as n -> infinity. At
    # finite n it stays small relative to a worst-case miscalibration.
    rng = _rng()
    n = 4000
    c = rng.uniform(0.0, 1.0, size=n)
    y = rng.binomial(1, c).astype(np.float64)
    assert smece(c, y, sigma=0.1) < 0.05


def test_maximally_miscalibrated_is_large() -> None:
    # Confidence pinned at 0.5, ground truth always 1: the smoothed
    # curve estimates p(y=1 | c~=0.5) ~ 1.0, and |1.0 - 0.5| = 0.5 is
    # what smECE reports.
    n = 1000
    c = np.full(n, 0.5)
    y = np.ones(n)
    val = smece(c, y, sigma=0.1)
    assert abs(val - 0.5) < 1e-6


def test_constant_confidence_zero_with_zero_correct() -> None:
    # Symmetric inverse case: c=0.5 always, y=0 always -> smECE = 0.5.
    n = 1000
    c = np.full(n, 0.5)
    y = np.zeros(n)
    val = smece(c, y, sigma=0.1)
    assert abs(val - 0.5) < 1e-6


def test_smece_independent_of_chunk_size() -> None:
    # Chunked computation must match the dense one bit-for-bit (or near
    # enough): smECE is a deterministic function of (c, y, sigma), and
    # the chunk loop is a memory-tiling refactor, not a math change.
    rng = _rng(seed=42)
    n = 800
    c = rng.uniform(0.0, 1.0, size=n)
    y = rng.binomial(1, c).astype(np.float64)
    val_small = smece(c, y, sigma=0.1, chunk_size=64)
    val_large = smece(c, y, sigma=0.1, chunk_size=4096)
    assert abs(val_small - val_large) < 1e-12


def test_smece_finer_than_ece_on_small_gaps() -> None:
    # ECE smears small gaps inside its bins; smECE should still see a
    # non-zero gap when ground truth deviates from confidence at a sub-
    # bin scale. We construct a dataset where every confidence value is
    # 0.5 +/- 0.005 (well inside one ECE bin) but the ground truth
    # systematically points to ~0.7. ECE registers ~0.2; smECE should
    # also register ~0.2 (the gap is real, not a binning artifact). The
    # value of this test: confirms smECE does NOT zero out small-scale
    # detail the way it would if it were a coarser binning method.
    rng = _rng(seed=7)
    n = 2000
    c = 0.5 + 0.005 * (rng.uniform(-1.0, 1.0, size=n))
    y = rng.binomial(1, 0.7 * np.ones(n)).astype(np.float64)
    sm = smece(c, y, sigma=0.05)
    assert sm > 0.15
    assert sm < 0.25


def test_sigma_validation() -> None:
    c = np.array([0.5, 0.6])
    y = np.array([1.0, 0.0])
    with pytest.raises(ValueError, match="sigma"):
        smece(c, y, sigma=0.0)
    with pytest.raises(ValueError, match="sigma"):
        smece(c, y, sigma=-0.1)
    with pytest.raises(ValueError, match="sigma"):
        smece(c, y, sigma=1.5)


def test_input_validation_inherited() -> None:
    # Reuses _validate from metrics.py - a quick check that empty input
    # and shape mismatches surface the expected errors.
    with pytest.raises(ValueError):
        smece(np.array([]), np.array([]))
    with pytest.raises(ValueError):
        smece(np.array([0.5]), np.array([0.5, 0.6]))
    with pytest.raises(ValueError):
        smece(np.array([1.5]), np.array([1.0]))


def test_smece_matches_ece_order_on_synthetic() -> None:
    # smECE and ECE should agree on direction (which of two distributions
    # is more miscalibrated) on cleanly-separated synthetic cases. They
    # disagree on magnitude by design.
    rng = _rng(seed=1)
    n = 2000
    c_well = rng.uniform(0.0, 1.0, size=n)
    y_well = rng.binomial(1, c_well).astype(np.float64)
    c_bad = c_well
    y_bad = (c_well > 0.3).astype(np.float64)  # miscalibrated step function
    assert smece(c_well, y_well) < smece(c_bad, y_bad)
    assert ece(c_well, y_well) < ece(c_bad, y_bad)
