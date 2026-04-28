# pydantic-cal

Accuracy answers *"is the model right?"*. Calibration answers *"when the model says it's 90% confident, is it right 90% of the time?"*. An LLM can be 91% accurate and 72% miscalibrated at the same time. Every eval library on PyPI today reports the first number; none report the second. `pydantic-cal` is the calibration layer for [pydantic-evals](https://ai.pydantic.dev/evals/).

```bash
pip install pydantic-cal
```

```python
import numpy as np
from pydantic_cal import ece, mce, ace, brier, brier_decomposition, TemperatureScaler

# Pull (confidence, correct) pairs from any eval run
confidences = np.array([0.95, 0.80, 0.70, 0.60, 0.45, 0.30])
correct     = np.array([1, 1, 0, 1, 0, 0])

print(f"ECE  = {ece(confidences, correct):.4f}")     # Expected calibration error
print(f"ACE  = {ace(confidences, correct):.4f}")     # Adaptive (equal-mass bins)
print(f"MCE  = {mce(confidences, correct):.4f}")     # Worst-case bin gap
print(f"Brier= {brier(confidences, correct):.4f}")   # MSE confidence vs outcome

decomp = brier_decomposition(confidences, correct, n_bins=10)
print(f"Reliability  {decomp.reliability:.4f}  (the calibration component)")
print(f"Resolution   {decomp.resolution:.4f}   (the discrimination component)")
print(f"Uncertainty  {decomp.uncertainty:.4f}  (intrinsic, can't be reduced)")
```

Fit a temperature scaler on a calibration split and recheck:

```python
scaler = TemperatureScaler().fit(cal_confidences, cal_labels)
ece_after = ece(scaler.transform(test_confidences), test_labels)
```

## Why this exists

Every other eval framework reports accuracy + LLM-judge variants. Surveyed nine of them - pydantic-evals, openai-evals, lm-eval-harness, deepeval, promptfoo, ragas, langfuse, langchain.evaluation, helicone - and not one ships ECE, Brier, or a reliability diagram. The gap is not subtle.

Calibration matters most when downstream code routes on confidence: human-in-the-loop fallback, conformal prediction, eval-test-suite pruning, RAG re-ranking, gating UX disclaimers. A miscalibrated 0.85 confidence routes the same as a calibrated 0.85 confidence and the system breaks silently.

## Modules

| Module | What it does |
|---|---|
| `pydantic_cal.metrics` | ECE / MCE / ACE / Brier / reliability curves |
| `pydantic_cal.decomposition` | Murphy 1973 partition: reliability + resolution + uncertainty |
| `pydantic_cal.scalers` | Post-hoc calibration: TemperatureScaler / PlattScaler / IsotonicScaler |
| `pydantic_cal.bootstrap` | Bootstrap CIs + paired before/after tests |
| `pydantic_cal._geometry` | Internal: Fisher-Rao distance, Jensen-Shannon, Bhattacharyya, Hellinger, KL, Bregman, alpha-divergence |

The `_geometry` module ships textbook information-geometry primitives (Amari 1985, Chentsov 1972, Nielsen 2018, Lin 1991) on the probability simplex. It is internal because no second consumer needs it yet; it will lift to a standalone `f3d1-information-geometry` package on PyPI once that changes.

## Pydantic-evals integration

```python
# (lands in v0.0.2 - adapter via the Evaluator + ReportEvaluator hooks)
from pydantic_cal.evaluators import ECE, Brier, ReliabilityDiagram

dataset = Dataset(cases=[...])
report = await dataset.evaluate(
    task_fn,
    report_evaluators=[ECE(n_bins=10), Brier(), ReliabilityDiagram()],
)
print(report.report_metrics["ece"])
```

## What's behind `pydantic_cal.crazy`

A paper-locked novel ensemble-structure scaler. Imports raise `ImportError` until f3d1 paper 1 publishes. The public stack above is unconditionally available and does not depend on the gated method. See `src/pydantic_cal/crazy/__init__.py` for the unlock conditions.

## Layout

```
pydantic-cal/
  src/pydantic_cal/
    __init__.py        public API
    metrics.py         ECE / MCE / ACE / Brier / reliability curves
    decomposition.py   Murphy 1973 partition
    scalers.py         temperature / Platt / isotonic
    bootstrap.py       CI machinery
    _geometry.py       internal info-geometry kernel
    crazy/             paper-locked gate
  examples/smoke.py    end-to-end sanity check
  tests/               pytest suite
```

## License

MIT.

## References

- Amari, S. (1985). *Differential-Geometrical Methods in Statistics*. Springer.
- Chentsov, N. N. (1972). *Statistical Decision Rules and Optimal Inference*.
- Murphy, A. H. (1973). *A New Vector Partition of the Probability Score*.
- Lin, J. (1991). *Divergence Measures Based on the Shannon Entropy*. IEEE TIT.
- Naeini, M. P., Cooper, G. F., & Hauskrecht, M. (2015). *Obtaining Well Calibrated Probabilities Using Bayesian Binning*. AAAI.
- Nixon, J., et al. (2019). *Measuring Calibration in Deep Learning*. CVPR Workshops.
- Nielsen, F. (2018). *On Information Projections, Bregman Divergences and the Centroids of Continuous Probability Distributions*. ICASSP.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On Calibration of Modern Neural Networks*. ICML.
