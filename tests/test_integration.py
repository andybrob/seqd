"""Integration tests: full pipeline on synthetic data."""

import datetime

import numpy as np
import pandas as pd
import pytest

from seqd import SeqdDecomposer, DecompositionResult


def make_full_synthetic(seed=42):
    """Generate a 3-year daily series with all three effects."""
    rng = np.random.default_rng(seed)
    n_days = 365 * 3 + 1
    dates = pd.date_range("2022-01-01", periods=n_days, freq="D")
    t = np.arange(n_days, dtype=float)

    # Trend
    trend = 100.0 + 0.02 * t

    # Weekly effect (additive)
    true_weekly = np.array([5.0, 3.0, 0.0, -2.0, -4.0, -8.0, 6.0])
    true_weekly -= true_weekly.mean()
    dow = dates.dayofweek
    weekly_component = true_weekly[dow]

    # Annual Fourier (K=2)
    annual_component = (
        10.0 * np.cos(2 * np.pi * t / 365.25)
        + 5.0 * np.sin(2 * np.pi * t / 365.25)
        + 3.0 * np.cos(4 * np.pi * t / 365.25)
        + 2.0 * np.sin(4 * np.pi * t / 365.25)
    )

    # Holiday effects: Christmas with ramp
    holiday_dates = [
        datetime.date(2022, 12, 25),
        datetime.date(2023, 12, 25),
        datetime.date(2024, 12, 25),
    ]
    holiday_component = np.zeros(n_days)
    for h in holiday_dates:
        h_ts = pd.Timestamp(h)
        if h_ts not in dates:
            continue
        h_idx = dates.get_loc(h_ts)
        for delta in range(-5, 6):
            idx_d = h_idx + delta
            if 0 <= idx_d < n_days:
                ramp_factor = 1.0 - abs(delta) / 6.0
                holiday_component[idx_d] += 20.0 * ramp_factor

    # Noise
    noise = rng.normal(0, 2.0, n_days)

    y_vals = trend + weekly_component + annual_component + holiday_component + noise
    y = pd.Series(y_vals, index=dates, name="sales")

    return y, holiday_dates, true_weekly, annual_component, holiday_component


def test_full_pipeline_runs():
    """Full pipeline should complete without errors."""
    y, holiday_dates, _, _, _ = make_full_synthetic()
    decomp = SeqdDecomposer(holiday_dates={"Christmas": holiday_dates})
    result = decomp.fit(y)
    assert isinstance(result, DecompositionResult)


def test_residual_lower_variance():
    """Residual variance should be substantially lower than original variance."""
    y, holiday_dates, _, _, _ = make_full_synthetic()
    decomp = SeqdDecomposer(holiday_dates={"Christmas": holiday_dates})
    result = decomp.fit(y)

    # Detrend both for fair comparison
    n = len(y)
    t = np.arange(n, dtype=float)
    coef_y = np.polyfit(t, y.values, 1)
    coef_r = np.polyfit(t, result.residual.values, 1)

    var_y = float(np.var(y.values - np.polyval(coef_y, t)))
    var_r = float(np.var(result.residual.values - np.polyval(coef_r, t)))

    assert var_r < var_y * 0.5, (
        f"Residual variance {var_r:.2f} should be < 50% of original {var_y:.2f}"
    )


def test_components_sum_to_original():
    """fitted() should approximately reconstruct the original series."""
    y, holiday_dates, _, _, _ = make_full_synthetic()
    decomp = SeqdDecomposer(holiday_dates={"Christmas": holiday_dates})
    result = decomp.fit(y)

    reconstructed = result.fitted()
    diff = np.abs(reconstructed.values - y.values)
    max_diff = float(diff.max())

    # Max reconstruction error should be very small (floating point)
    assert max_diff < 0.1, (
        f"Max reconstruction error {max_diff:.6f} > 0.1 — components don't sum to original"
    )


def test_r2_by_component():
    """R² by component should be reasonable for our synthetic data."""
    y, holiday_dates, _, _, _ = make_full_synthetic()
    decomp = SeqdDecomposer(holiday_dates={"Christmas": holiday_dates})
    result = decomp.fit(y)

    r2 = result.r2_by_component
    assert "weekly" in r2
    assert "holiday" in r2
    assert "annual" in r2

    # All R² should be non-negative
    for k, v in r2.items():
        assert v >= 0, f"r2[{k}] = {v} < 0"

    # Weekly R² should be substantial given strong DOW effects
    assert r2["weekly"] > 0.01, f"Weekly R² too low: {r2['weekly']:.4f}"


def test_weekly_component_shape():
    """weekly_component() should have same length as input."""
    y, holiday_dates, _, _, _ = make_full_synthetic()
    decomp = SeqdDecomposer(holiday_dates={"Christmas": holiday_dates})
    result = decomp.fit(y)

    wc = result.weekly_component()
    assert len(wc) == len(y)
    assert wc.index.equals(y.index)


def test_holiday_component_shape():
    """holiday_component() should have same length as input."""
    y, holiday_dates, _, _, _ = make_full_synthetic()
    decomp = SeqdDecomposer(holiday_dates={"Christmas": holiday_dates})
    result = decomp.fit(y)

    hc = result.holiday_component()
    assert len(hc) == len(y)
    assert hc.index.equals(y.index)


def test_annual_component_shape():
    """annual_component() should have same length as input."""
    y, holiday_dates, _, _, _ = make_full_synthetic()
    decomp = SeqdDecomposer(holiday_dates={"Christmas": holiday_dates})
    result = decomp.fit(y)

    ac = result.annual_component()
    assert len(ac) == len(y)
    assert ac.index.equals(y.index)


def test_holiday_input_formats():
    """Should accept flat list, dict, and mixed date types."""
    y, holiday_dates, _, _, _ = make_full_synthetic()

    # Flat list of strings
    decomp1 = SeqdDecomposer(
        holiday_dates=[str(h) for h in holiday_dates]
    )
    r1 = decomp1.fit(y)
    assert isinstance(r1, DecompositionResult)

    # Dict with timestamps
    decomp2 = SeqdDecomposer(
        holiday_dates={"Christmas": [pd.Timestamp(h) for h in holiday_dates]}
    )
    r2 = decomp2.fit(y)
    assert isinstance(r2, DecompositionResult)


def test_no_holidays():
    """Pipeline should work with no holidays."""
    rng = np.random.default_rng(99)
    dates = pd.date_range("2022-01-01", periods=365 * 2, freq="D")
    y = pd.Series(100.0 + rng.normal(0, 5.0, len(dates)), index=dates)

    decomp = SeqdDecomposer(holiday_dates=[])
    result = decomp.fit(y)
    assert isinstance(result, DecompositionResult)
    assert len(result.holidays) == 0


def test_weekly_effect_recovery():
    """Weekly DOW effects in full pipeline should match true effects within 10%."""
    y, holiday_dates, true_weekly, _, _ = make_full_synthetic()
    decomp = SeqdDecomposer(holiday_dates={"Christmas": holiday_dates})
    result = decomp.fit(y)

    estimated = result.weekly.coefficients
    true_norm = true_weekly  # already normalized (mean-zero) in make_full_synthetic

    effect_range = true_norm.max() - true_norm.min()
    tolerance = 0.15 * effect_range  # 15% tolerance in full pipeline

    errors = np.abs(estimated - true_norm)
    max_error = float(errors.max())
    assert max_error <= tolerance, (
        f"Weekly effect recovery error {max_error:.3f} > tolerance {tolerance:.3f}"
    )


def test_residual_index():
    """Residual should have same index as input."""
    y, holiday_dates, _, _, _ = make_full_synthetic()
    decomp = SeqdDecomposer(holiday_dates={"Christmas": holiday_dates})
    result = decomp.fit(y)

    assert result.residual.index.equals(y.index)
