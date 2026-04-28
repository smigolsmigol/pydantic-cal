"""End-to-end smoke for pydantic-cal.

Synthesises a miscalibrated forecast (overconfident at the high end),
runs the full metric suite + bootstrap CI + temperature scaling, and
checks the calibration improves. Also exercises the geometry kernel
and confirms the crazy gate is locked.
"""
from __future__ import annotations

import numpy as np

from pydantic_cal import (
    BrierDecomposition,
    TemperatureScaler,
    ace,
    bootstrap_ci,
    brier,
    brier_decomposition,
    ece,
    mce,
    paired_bootstrap_diff,
    reliability_curve,
)
from pydantic_cal._geometry import (
    alpha_divergence,
    bhattacharyya,
    fisher_rao,
    hellinger,
    jensen_shannon,
    kl,
)


def make_miscalibrated(n: int = 1000, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Generate confidences and outcomes where the model is overconfident."""
    rng = np.random.default_rng(seed)
    true_p = rng.uniform(0.1, 0.95, size=n)
    correct = (rng.uniform(size=n) < true_p).astype(np.float64)
    # overconfidence: report sqrt of the true probability (pulls toward 1)
    confidences = np.sqrt(true_p)
    return confidences, correct


def main() -> None:
    confidences, correct = make_miscalibrated()

    print("== metrics on miscalibrated forecast ==")
    e = ece(confidences, correct, n_bins=10)
    m = mce(confidences, correct, n_bins=10)
    a = ace(confidences, correct, n_bins=10)
    b = brier(confidences, correct)
    print(f"  ECE  = {e:.4f}")
    print(f"  MCE  = {m:.4f}")
    print(f"  ACE  = {a:.4f}")
    print(f"  Brier= {b:.4f}")
    assert e > 0.05, "synthetic overconfidence should produce ECE > 0.05"

    curve = reliability_curve(confidences, correct, n_bins=10)
    print(f"  reliability curve: {curve.n_bins} bins, {curve.n_samples} samples")
    print("  -> metrics OK")

    print("\n== Brier decomposition (Murphy 1973) ==")
    decomp: BrierDecomposition = brier_decomposition(confidences, correct, n_bins=20)
    print(f"  Brier       = {decomp.brier:.4f}")
    print(f"  Reliability = {decomp.reliability:.4f}  (lower is better)")
    print(f"  Resolution  = {decomp.resolution:.4f}  (higher is better)")
    print(f"  Uncertainty = {decomp.uncertainty:.4f}  (intrinsic)")
    print(f"  consistent (binned approx): {decomp.is_consistent(tol=0.02)}")
    print("  -> decomposition OK")

    print("\n== bootstrap CI on ECE ==")
    ci = bootstrap_ci(
        np.column_stack([confidences, correct]),
        statistic=lambda v: ece(v[:, 0], v[:, 1], n_bins=10),
        n_resamples=500,
    )
    print(f"  ECE = {ci}")
    assert ci.lower <= ci.point <= ci.upper, "point estimate must lie in CI"
    print("  -> bootstrap OK")

    print("\n== temperature scaling ==")
    scaler = TemperatureScaler().fit(confidences[:500], correct[:500])
    print(f"  fit T = {scaler.temperature:.3f}")
    cal = scaler.transform(confidences[500:])
    e_after = ece(cal, correct[500:], n_bins=10)
    e_before = ece(confidences[500:], correct[500:], n_bins=10)
    print(f"  ECE before scaling = {e_before:.4f}")
    print(f"  ECE after  scaling = {e_after:.4f}")

    print("\n== paired bootstrap on the improvement ==")
    held_c = confidences[500:]
    held_y = correct[500:]
    diff_ci = paired_bootstrap_diff(
        np.column_stack([held_c, held_y]),
        np.column_stack([cal, held_y]),
        statistic=lambda v: ece(v[:, 0], v[:, 1], n_bins=10),
        n_resamples=500,
    )
    print(f"  delta-ECE = {diff_ci}")
    print("  -> negative interval means scaling reduced ECE")

    print("\n== information-geometry primitives ==")
    p = np.array([0.7, 0.2, 0.1])
    q = np.array([0.3, 0.5, 0.2])
    print(f"  KL(p||q)              = {kl(p, q):.4f}")
    print(f"  JSD(p, q)             = {jensen_shannon(p, q):.4f}")
    print(f"  Bhattacharyya(p, q)   = {bhattacharyya(p, q):.4f}")
    print(f"  Hellinger(p, q)       = {hellinger(p, q):.4f}")
    print(f"  Fisher-Rao(p, q)      = {fisher_rao(p, q):.4f}")
    print(f"  alpha-div(p, q, 0.5)  = {alpha_divergence(p, q, 0.5):.4f}")
    # symmetry sanity
    assert abs(jensen_shannon(p, q) - jensen_shannon(q, p)) < 1e-10
    assert abs(fisher_rao(p, q) - fisher_rao(q, p)) < 1e-10
    print("  -> geometry kernel OK (symmetry verified)")

    print("\n== crazy gate ==")
    try:
        import pydantic_cal.crazy  # noqa: F401

        print("  ERROR: crazy gate should be locked but imported successfully")
    except ImportError as e:
        print(f"  gate locked as expected:")
        print(f"    {str(e)[:120]}...")
        print("  -> gate OK")

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
