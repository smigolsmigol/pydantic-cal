"""Post-hoc calibration scalers: temperature, Platt, isotonic.

Fit on a held-out calibration split. At inference, transform raw model
confidence -> calibrated confidence. Common pattern:

    scaler = TemperatureScaler().fit(cal_logits, cal_labels)
    p_cal = scaler.transform(prod_logits)
    ece_after = ece(p_cal, prod_labels)

Temperature scaling has one parameter (T); Platt has two (a, b);
isotonic is non-parametric. Pick by sample budget: T < 100 examples,
Platt 100-1000, isotonic > 1000.
"""
from __future__ import annotations

import numpy as np


class TemperatureScaler:
    """Single-parameter scaling: confidence_calibrated = sigmoid(logit / T).

    Operates on confidences in [0, 1] (not logits) for ergonomic LLM
    use where the consumer rarely has raw logits. Internally converts
    confidence -> logit -> divide by T -> sigmoid.

    Optimizes T by minimizing NLL via grid + golden-section refine -
    avoids a scipy dependency for a one-parameter optimization.
    """

    def __init__(self) -> None:
        self.temperature: float | None = None

    def fit(self, confidences: np.ndarray, correct: np.ndarray) -> "TemperatureScaler":
        c = np.clip(np.asarray(confidences, dtype=np.float64), 1e-6, 1 - 1e-6)
        y = np.asarray(correct, dtype=np.float64)
        logits = np.log(c / (1 - c))

        def nll(t: float) -> float:
            scaled = logits / t
            p = 1.0 / (1.0 + np.exp(-scaled))
            p = np.clip(p, 1e-12, 1 - 1e-12)
            return -float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

        # coarse grid then golden-section refine
        grid = np.linspace(0.1, 10.0, 100)
        best = float(grid[int(np.argmin([nll(float(t)) for t in grid]))])
        # refine
        lo, hi = max(0.05, best - 0.5), best + 0.5
        phi = (5**0.5 - 1) / 2
        for _ in range(40):
            a = hi - phi * (hi - lo)
            b = lo + phi * (hi - lo)
            if nll(a) < nll(b):
                hi = b
            else:
                lo = a
        self.temperature = (lo + hi) / 2
        return self

    def transform(self, confidences: np.ndarray) -> np.ndarray:
        if self.temperature is None:
            raise RuntimeError("TemperatureScaler.fit must be called before transform")
        c = np.clip(np.asarray(confidences, dtype=np.float64), 1e-6, 1 - 1e-6)
        logits = np.log(c / (1 - c))
        scaled = logits / self.temperature
        return 1.0 / (1.0 + np.exp(-scaled))


class PlattScaler:
    """Two-parameter scaling: p_cal = sigmoid(a * logit + b).

    Platt 1999. More flexible than TemperatureScaler, handles bias as
    well as sharpness. Same fit interface; same optimizer.
    """

    def __init__(self) -> None:
        self.a: float | None = None
        self.b: float | None = None

    def fit(self, confidences: np.ndarray, correct: np.ndarray) -> "PlattScaler":
        c = np.clip(np.asarray(confidences, dtype=np.float64), 1e-6, 1 - 1e-6)
        y = np.asarray(correct, dtype=np.float64)
        logits = np.log(c / (1 - c))

        def nll(params: tuple[float, float]) -> float:
            a, b = params
            scaled = a * logits + b
            p = 1.0 / (1.0 + np.exp(-scaled))
            p = np.clip(p, 1e-12, 1 - 1e-12)
            return -float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

        # coarse 2-D grid
        grid_a = np.linspace(0.1, 5.0, 25)
        grid_b = np.linspace(-3.0, 3.0, 25)
        best = (1.0, 0.0)
        best_nll = float("inf")
        for ai in grid_a:
            for bi in grid_b:
                v = nll((float(ai), float(bi)))
                if v < best_nll:
                    best_nll = v
                    best = (float(ai), float(bi))
        # local refine via Nelder-Mead-style coordinate descent
        a, b = best
        step_a, step_b = 0.5, 0.5
        for _ in range(60):
            improved = False
            for da in (-step_a, 0.0, step_a):
                for db in (-step_b, 0.0, step_b):
                    if da == 0.0 and db == 0.0:
                        continue
                    cand = (max(0.05, a + da), b + db)
                    v = nll(cand)
                    if v < best_nll:
                        best_nll = v
                        a, b = cand
                        improved = True
            if not improved:
                step_a *= 0.5
                step_b *= 0.5
                if step_a < 1e-4:
                    break
        self.a, self.b = a, b
        return self

    def transform(self, confidences: np.ndarray) -> np.ndarray:
        if self.a is None or self.b is None:
            raise RuntimeError("PlattScaler.fit must be called before transform")
        c = np.clip(np.asarray(confidences, dtype=np.float64), 1e-6, 1 - 1e-6)
        logits = np.log(c / (1 - c))
        scaled = self.a * logits + self.b
        return 1.0 / (1.0 + np.exp(-scaled))


class IsotonicScaler:
    """Non-parametric isotonic regression scaling.

    Pool-adjacent-violators (PAV) algorithm. No assumptions about the
    shape of the miscalibration curve. Needs more samples than Platt /
    Temperature to avoid overfitting; rule of thumb >= 1000.
    """

    def __init__(self) -> None:
        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None

    def fit(self, confidences: np.ndarray, correct: np.ndarray) -> "IsotonicScaler":
        c = np.asarray(confidences, dtype=np.float64)
        y = np.asarray(correct, dtype=np.float64)
        order = np.argsort(c)
        x_sorted = c[order]
        y_sorted = y[order]
        # Pool-adjacent-violators
        n = len(x_sorted)
        weights = np.ones(n)
        means = y_sorted.copy()
        i = 0
        while i < n - 1:
            if means[i] > means[i + 1]:
                # pool
                w_sum = weights[i] + weights[i + 1]
                pooled = (means[i] * weights[i] + means[i + 1] * weights[i + 1]) / w_sum
                means[i] = pooled
                weights[i] = w_sum
                # remove i+1
                means = np.delete(means, i + 1)
                weights = np.delete(weights, i + 1)
                x_sorted = np.delete(x_sorted, i + 1)
                # back up to check the new pair
                if i > 0:
                    i -= 1
            else:
                i += 1
        self._x = x_sorted
        self._y = means
        return self

    def transform(self, confidences: np.ndarray) -> np.ndarray:
        if self._x is None or self._y is None:
            raise RuntimeError("IsotonicScaler.fit must be called before transform")
        c = np.asarray(confidences, dtype=np.float64)
        return np.interp(c, self._x, self._y)
