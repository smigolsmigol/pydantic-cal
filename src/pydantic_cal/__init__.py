"""pydantic-cal: LLM calibration metrics for pydantic-evals.

Accuracy answers "is the model right?". Calibration answers "when the
model says it's 90% confident, is it right 90% of the time?". An LLM
can be 91% accurate and 72% miscalibrated at the same time -
pydantic-cal makes the second number visible.

Modules:
- metrics       ECE, MCE, ACE, Brier, reliability curves
- decomposition Murphy 1973 partition (reliability + resolution +
                uncertainty)
- scalers       post-hoc calibration: temperature, Platt, isotonic
- bootstrap     CI machinery + paired tests for before/after claims
- _geometry     internal: information-geometry primitives on the simplex
                (Fisher-Rao distance, JSD, Bhattacharyya, Hellinger,
                KL, Bregman, alpha-divergence). Public-knowledge math
                only; will extract to standalone f3d1-information-
                geometry on PyPI once a second consumer needs it.
- crazy         paper-locked novel ensemble-structure scaler. Raises
                ImportError until f3d1 paper 1 publishes.
"""
from __future__ import annotations

from pydantic_cal.bootstrap import BootstrapCI, bootstrap_ci, paired_bootstrap_diff
from pydantic_cal.decomposition import BrierDecomposition, brier_decomposition
from pydantic_cal.metrics import (
    BinResult,
    ReliabilityCurve,
    ace,
    brier,
    ece,
    mce,
    reliability_curve,
)
from pydantic_cal.scalers import IsotonicScaler, PlattScaler, TemperatureScaler

__version__ = "0.0.2"

__all__ = [
    # metrics
    "BinResult",
    "ReliabilityCurve",
    "ace",
    "brier",
    "ece",
    "mce",
    "reliability_curve",
    # decomposition
    "BrierDecomposition",
    "brier_decomposition",
    # scalers
    "IsotonicScaler",
    "PlattScaler",
    "TemperatureScaler",
    # bootstrap
    "BootstrapCI",
    "bootstrap_ci",
    "paired_bootstrap_diff",
]
