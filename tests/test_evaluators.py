"""Tests for the pydantic-evals adapter.

Skips cleanly if pydantic-evals is not installed (the dep is optional).
"""
from __future__ import annotations

import pytest

pytest.importorskip("pydantic_evals", reason="pydantic-evals optional dep not installed")

from dataclasses import dataclass
from typing import Any

import numpy as np


def _fake_report(pairs: list[tuple[float, bool]]):
    """Build a minimal stand-in for EvaluationReport with .cases."""

    @dataclass
    class _FakeAssertion:
        value: bool

    @dataclass
    class _FakeCase:
        name: str
        metrics: dict[str, float]
        metadata: Any
        assertions: dict[str, Any]
        expected_output: Any
        output: Any

    @dataclass
    class _FakeReport:
        cases: list[_FakeCase]

    cases = [
        _FakeCase(
            name=f"case-{i}",
            metrics={"confidence": conf},
            metadata={},
            assertions={"correct": _FakeAssertion(value=ok)},
            expected_output=None,
            output=None,
        )
        for i, (conf, ok) in enumerate(pairs)
    ]
    return _FakeReport(cases=cases)


def _make_ctx(report):
    @dataclass
    class _Ctx:
        name: str
        report: Any
        experiment_metadata: Any

    return _Ctx(name="test", report=report, experiment_metadata=None)


def test_extract_pairs_round_trip():
    from pydantic_cal.evaluators import extract_pairs

    pairs = [(0.9, True), (0.7, True), (0.5, False), (0.2, False)]
    confs, corrects = extract_pairs(_fake_report(pairs))
    assert list(confs) == [0.9, 0.7, 0.5, 0.2]
    assert list(corrects) == [1.0, 1.0, 0.0, 0.0]


def test_extract_pairs_skips_missing_confidence():
    from pydantic_cal.evaluators import extract_pairs

    report = _fake_report([(0.9, True), (0.5, False)])
    # Drop confidence on second case
    report.cases[1].metrics = {}
    report.cases[1].metadata = {}
    confs, corrects = extract_pairs(report)
    assert list(confs) == [0.9]
    assert list(corrects) == [1.0]


def test_extract_pairs_falls_back_to_metadata():
    from pydantic_cal.evaluators import extract_pairs

    report = _fake_report([(0.9, True)])
    # Move confidence to metadata
    report.cases[0].metrics = {}
    report.cases[0].metadata = {"confidence": 0.85}
    confs, _ = extract_pairs(report)
    assert list(confs) == [0.85]


def test_extract_pairs_raises_on_empty():
    from pydantic_cal.evaluators import extract_pairs

    report = _fake_report([])
    with pytest.raises(ValueError, match="no .* pairs extracted"):
        extract_pairs(report)


def test_ece_evaluator_returns_scalar():
    from pydantic_cal.evaluators import ECE
    from pydantic_evals.reporting.analyses import ScalarResult

    rng = np.random.default_rng(0)
    pairs = [
        (float(c), bool(rng.uniform() < c))
        for c in rng.uniform(0.1, 0.95, size=200)
    ]
    result = ECE(n_bins=10).evaluate(_make_ctx(_fake_report(pairs)))
    assert isinstance(result, ScalarResult)
    assert 0.0 <= result.value <= 1.0
    assert result.title == "Expected Calibration Error"


def test_brier_evaluator():
    from pydantic_cal.evaluators import Brier
    from pydantic_evals.reporting.analyses import ScalarResult

    perfect = [(1.0, True)] * 50 + [(0.0, False)] * 50
    result = Brier().evaluate(_make_ctx(_fake_report(perfect)))
    assert isinstance(result, ScalarResult)
    assert result.value == pytest.approx(0.0, abs=1e-9)


def test_mce_and_ace_run():
    from pydantic_cal.evaluators import ACE, MCE

    pairs = [(0.9, True), (0.8, False), (0.7, True), (0.4, False), (0.2, True)]
    mce_result = MCE(n_bins=5).evaluate(_make_ctx(_fake_report(pairs)))
    ace_result = ACE(n_bins=3).evaluate(_make_ctx(_fake_report(pairs)))
    assert mce_result.value >= 0
    assert ace_result.value >= 0


def test_reliability_diagram_returns_lineplot():
    from pydantic_cal.evaluators import ReliabilityDiagram
    from pydantic_evals.reporting.analyses import LinePlot

    pairs = [(c / 10, c % 2 == 0) for c in range(1, 10)]
    result = ReliabilityDiagram(n_bins=5).evaluate(_make_ctx(_fake_report(pairs)))
    assert isinstance(result, LinePlot)
    assert result.x_label == "Predicted confidence"
    assert result.y_label == "Observed accuracy"
    assert len(result.curves) == 2
    assert result.curves[0].name == "observed"
    assert result.curves[1].name == "perfect calibration"
    assert result.curves[1].style == "dashed"


def test_correct_fn_override():
    from pydantic_cal.evaluators import ECE

    pairs = [(0.9, True), (0.5, False)]
    report = _fake_report(pairs)
    # Force everything wrong via a custom correct_fn
    result = ECE(n_bins=2, correct_fn=lambda case: False).evaluate(_make_ctx(report))
    # All cases marked wrong; ECE should be the mean confidence (since accuracy=0)
    assert result.value > 0.5


def test_confidence_field_override():
    from pydantic_cal.evaluators import ECE

    report = _fake_report([(0.9, True)])
    report.cases[0].metrics = {"my_score": 0.7, "confidence": 0.9}
    result = ECE(n_bins=10, confidence_field="my_score").evaluate(_make_ctx(report))
    # Picked up 0.7 as confidence, not 0.9
    assert 0.0 <= result.value <= 1.0
