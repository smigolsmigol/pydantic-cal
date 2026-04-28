"""pydantic-evals adapter: drop calibration into dataset.evaluate().

Usage (requires `pip install pydantic-cal[pydantic-evals]`):

    from pydantic_evals import Dataset, Case
    from pydantic_cal.evaluators import ECE, Brier, ReliabilityDiagram

    dataset = Dataset(cases=[
        Case(name="q1", inputs={...}, expected_output="yes",
             metadata={"confidence": 0.92}),
        ...
    ])
    report = await dataset.evaluate(
        my_task,
        report_evaluators=[ECE(n_bins=10), Brier(), ReliabilityDiagram()],
    )

The evaluators read (confidence, correct) pairs from each ReportCase:
- confidence comes from case.metrics[confidence_field], falling back to
  case.metadata[confidence_field] when metadata is a dict
- correctness comes from case.assertions[correct_field] when present,
  otherwise from a strict equality between case.expected_output and
  case.output

Both fields and the correctness function are constructor-overridable so
this works against any task shape, not just multiple-choice.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from pydantic_cal.metrics import ace, brier, ece, mce, reliability_curve

if TYPE_CHECKING:
    from pydantic_evals.evaluators.report_evaluator import ReportEvaluatorContext
    from pydantic_evals.reporting import EvaluationReport, ReportCase

try:
    from pydantic_evals.evaluators.report_evaluator import ReportEvaluator as _ReportEvaluator
    from pydantic_evals.reporting.analyses import (
        LinePlot as _LinePlot,
        LinePlotCurve as _LinePlotCurve,
        LinePlotPoint as _LinePlotPoint,
        ScalarResult as _ScalarResult,
    )

    _PYDANTIC_EVALS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PYDANTIC_EVALS_AVAILABLE = False
    _ReportEvaluator = object  # type: ignore[assignment,misc]


_DEP_HINT = (
    "pydantic_cal.evaluators requires pydantic-evals. "
    "Install with: pip install pydantic-cal[pydantic-evals]"
)


def _check_dep() -> None:
    if not _PYDANTIC_EVALS_AVAILABLE:
        raise ImportError(_DEP_HINT)


def extract_pairs(
    report: "EvaluationReport",
    *,
    confidence_field: str = "confidence",
    correct_field: str | None = "correct",
    correct_fn: Callable[["ReportCase"], bool] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pull (confidence, correct) arrays out of a pydantic-evals report.

    Confidence is searched in this order: case.metrics[confidence_field],
    then case.metadata[confidence_field] if metadata is a dict-like.
    Cases missing a confidence are skipped.

    Correctness is searched in this order: correct_fn(case) if provided,
    then case.assertions[correct_field].value if present, then
    case.output == case.expected_output. Cases where correctness can't be
    determined are skipped.
    """
    confs: list[float] = []
    corrects: list[float] = []
    for case in report.cases:
        c = _read_confidence(case, confidence_field)
        if c is None:
            continue
        ok = _read_correct(case, correct_field, correct_fn)
        if ok is None:
            continue
        confs.append(float(c))
        corrects.append(1.0 if ok else 0.0)
    if not confs:
        raise ValueError(
            f"no (confidence, correct) pairs extracted from report "
            f"(checked {len(report.cases)} cases). Set confidence_field or "
            f"correct_fn explicitly if your cases store them under different keys."
        )
    return np.asarray(confs, dtype=np.float64), np.asarray(corrects, dtype=np.float64)


def _read_confidence(case: Any, field: str) -> float | None:
    metric = case.metrics.get(field) if isinstance(case.metrics, dict) else None
    if metric is not None:
        return float(metric)
    md = getattr(case, "metadata", None)
    if isinstance(md, dict):
        v = md.get(field)
        if v is not None:
            return float(v)
    return None


def _read_correct(
    case: Any,
    field: str | None,
    fn: Callable[[Any], bool] | None,
) -> bool | None:
    if fn is not None:
        return bool(fn(case))
    if field and isinstance(case.assertions, dict):
        result = case.assertions.get(field)
        if result is not None and hasattr(result, "value"):
            return bool(result.value)
    if case.expected_output is not None:
        return case.output == case.expected_output
    return None


def _pairs_from_ctx(
    ctx: "ReportEvaluatorContext",
    confidence_field: str,
    correct_field: str | None,
    correct_fn: Callable[[Any], bool] | None,
) -> tuple[np.ndarray, np.ndarray]:
    _check_dep()
    return extract_pairs(
        ctx.report,
        confidence_field=confidence_field,
        correct_field=correct_field,
        correct_fn=correct_fn,
    )


@dataclass
class ECE(_ReportEvaluator):  # type: ignore[misc,valid-type]
    """Expected Calibration Error (Naeini et al. 2015).

    ECE = sum over bins of (|bin| / N) * |confidence_mean - accuracy|.
    Equal-width bins. Lower is better; 0.0 is perfectly calibrated.
    """

    n_bins: int = 10
    confidence_field: str = "confidence"
    correct_field: str | None = "correct"
    correct_fn: Callable[[Any], bool] | None = None
    evaluation_name = "ECE"

    def evaluate(self, ctx: "ReportEvaluatorContext"):
        confs, corrects = _pairs_from_ctx(
            ctx, self.confidence_field, self.correct_field, self.correct_fn
        )
        return _ScalarResult(
            title="Expected Calibration Error",
            description="ECE = sum over bins of (|bin| / N) * |confidence_mean - accuracy|. Lower is better.",
            value=float(ece(confs, corrects, n_bins=self.n_bins)),
        )


@dataclass
class MCE(_ReportEvaluator):  # type: ignore[misc,valid-type]
    """Maximum Calibration Error: worst-case |confidence - accuracy| across bins."""

    n_bins: int = 10
    confidence_field: str = "confidence"
    correct_field: str | None = "correct"
    correct_fn: Callable[[Any], bool] | None = None
    evaluation_name = "MCE"

    def evaluate(self, ctx: "ReportEvaluatorContext"):
        confs, corrects = _pairs_from_ctx(
            ctx, self.confidence_field, self.correct_field, self.correct_fn
        )
        return _ScalarResult(
            title="Maximum Calibration Error",
            description="Worst-case |confidence - accuracy| across equal-width bins.",
            value=float(mce(confs, corrects, n_bins=self.n_bins)),
        )


@dataclass
class ACE(_ReportEvaluator):  # type: ignore[misc,valid-type]
    """Adaptive Calibration Error (Nixon et al. 2019).

    ECE-shape with equal-MASS bins. LLM confidence distributions are
    heavily peaked near the extremes which leaves most ECE bins empty;
    ACE fixes that bias.
    """

    n_bins: int = 10
    confidence_field: str = "confidence"
    correct_field: str | None = "correct"
    correct_fn: Callable[[Any], bool] | None = None
    evaluation_name = "ACE"

    def evaluate(self, ctx: "ReportEvaluatorContext"):
        confs, corrects = _pairs_from_ctx(
            ctx, self.confidence_field, self.correct_field, self.correct_fn
        )
        return _ScalarResult(
            title="Adaptive Calibration Error",
            description="ECE shape with equal-mass bins. Handles peaked LLM confidence distributions.",
            value=float(ace(confs, corrects, n_bins=self.n_bins)),
        )


@dataclass
class Brier(_ReportEvaluator):  # type: ignore[misc,valid-type]
    """Brier score: mean squared error between confidence and correctness.

    Decomposes into reliability + resolution + uncertainty (Murphy 1973).
    Lower is better; 0.0 is perfect.
    """

    n_bins: int = 10
    confidence_field: str = "confidence"
    correct_field: str | None = "correct"
    correct_fn: Callable[[Any], bool] | None = None
    evaluation_name = "Brier"

    def evaluate(self, ctx: "ReportEvaluatorContext"):
        confs, corrects = _pairs_from_ctx(
            ctx, self.confidence_field, self.correct_field, self.correct_fn
        )
        return _ScalarResult(
            title="Brier Score",
            description="Mean squared error between confidence and correctness. Decomposes into reliability + resolution + uncertainty.",
            value=float(brier(confs, corrects)),
        )


@dataclass
class ReliabilityDiagram(_ReportEvaluator):  # type: ignore[misc,valid-type]
    """Reliability curve as a LinePlot: per-bin accuracy vs predicted confidence.

    A perfectly calibrated model hugs the y=x diagonal. Bins with no
    samples are dropped from the observed curve.
    """

    n_bins: int = 10
    adaptive: bool = False
    confidence_field: str = "confidence"
    correct_field: str | None = "correct"
    correct_fn: Callable[[Any], bool] | None = None
    evaluation_name = "ReliabilityDiagram"

    def evaluate(self, ctx: "ReportEvaluatorContext"):
        confs, corrects = _pairs_from_ctx(
            ctx, self.confidence_field, self.correct_field, self.correct_fn
        )
        curve = reliability_curve(
            confs, corrects, n_bins=self.n_bins, adaptive=self.adaptive
        )
        observed = _LinePlotCurve(
            name="observed",
            points=[
                _LinePlotPoint(x=b.confidence_mean, y=b.accuracy)
                for b in curve.bins
                if b.count > 0
            ],
            style="solid",
        )
        ideal = _LinePlotCurve(
            name="perfect calibration",
            points=[_LinePlotPoint(x=0.0, y=0.0), _LinePlotPoint(x=1.0, y=1.0)],
            style="dashed",
        )
        return _LinePlot(
            title="Reliability Diagram",
            description=(
                f"Per-bin accuracy vs predicted confidence "
                f"({'equal-mass' if self.adaptive else 'equal-width'} bins, "
                f"n_bins={self.n_bins}). Curve hugging the diagonal = well calibrated."
            ),
            x_label="Predicted confidence",
            y_label="Observed accuracy",
            x_range=(0.0, 1.0),
            y_range=(0.0, 1.0),
            curves=[observed, ideal],
        )


__all__ = ["ECE", "MCE", "ACE", "Brier", "ReliabilityDiagram", "extract_pairs"]
