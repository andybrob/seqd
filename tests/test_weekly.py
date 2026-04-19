"""Tests for Stage 1: Weekly effect estimation."""

import datetime

import numpy as np
import pandas as pd
import pytest

from seqd._weekly import fit_weekly


def make_weekly_series(n_days=365 * 2, seed=42):
    """Generate synthetic series with known DOW effects (additive)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n_days, freq="D")

    # Linear trend
    trend = np.linspace(100.0, 150.0, n_days)

    # Known DOW effects: Mon=+5, Tue=+3, Wed=0, Thu=-2, Fri=-4, Sat=-8, Sun=+6
    true_effects = np.array([5.0, 3.0, 0.0, -2.0, -4.0, -8.0, 6.0])

    dow = dates.dayofweek
    weekly_component = true_effects[dow]

    # Small noise
    noise = rng.normal(0, 1.0, n_days)

    y = pd.Series(trend + weekly_component + noise, index=dates)
    return y, true_effects


def test_weekly_recovery_additive():
    """Weekly DOW effects should be recovered within 5% of true values."""
    y, true_effects = make_weekly_series()
    holiday_dates = []  # no holidays

    weekly_effect, y_w, is_mult = fit_weekly(y, holiday_dates)

    assert not is_mult, "Should detect additive mode for low-variance series"

    # Normalize true effects to have mean zero (as our estimator does)
    true_norm = true_effects - true_effects.mean()

    estimated = weekly_effect.coefficients  # shape (7,)

    # Allow 5% of the range of effects
    effect_range = true_norm.max() - true_norm.min()
    tolerance = 0.05 * effect_range

    errors = np.abs(estimated - true_norm)
    max_error = float(errors.max())
    assert max_error <= tolerance, (
        f"Max DOW estimation error {max_error:.3f} exceeds tolerance {tolerance:.3f}\n"
        f"True (norm): {true_norm}\nEstimated: {estimated}"
    )


def test_weekly_removes_effect():
    """After removing weekly effect, variance should decrease."""
    y, true_effects = make_weekly_series()
    _, y_w, _ = fit_weekly(y, [])

    # Variance of y_w should be lower than y (DOW variation removed)
    # Remove trend first for fair comparison
    n = len(y)
    t = np.arange(n, dtype=float)
    # Detrend y and y_w
    coef_y = np.polyfit(t, y.values, 1)
    coef_yw = np.polyfit(t, y_w.values, 1)
    y_detrended = y.values - np.polyval(coef_y, t)
    yw_detrended = y_w.values - np.polyval(coef_yw, t)

    var_y = float(np.var(y_detrended))
    var_yw = float(np.var(yw_detrended))

    assert var_yw < var_y, (
        f"Variance should decrease after removing weekly effect: "
        f"var(y)={var_y:.3f}, var(y_w)={var_yw:.3f}"
    )


def test_weekly_recency_structure():
    """Recency dict should have correct structure."""
    y, _ = make_weekly_series()
    weekly_effect, _, _ = fit_weekly(y, [])

    assert isinstance(weekly_effect.recency, dict)
    for window in [60, 90, 365]:
        assert window in weekly_effect.recency, f"Window {window} missing from recency"
        df = weekly_effect.recency[window]
        assert "date" in df.columns
        for d in range(7):
            assert f"dow_{d}" in df.columns, f"dow_{d} missing from recency df"


def test_weekly_drift_classification():
    """Drift dict should have correct structure."""
    y, _ = make_weekly_series()
    weekly_effect, _, _ = fit_weekly(y, [])

    for d in range(7):
        assert d in weekly_effect.drift
        assert "slope" in weekly_effect.drift[d]
        assert "classification" in weekly_effect.drift[d]
        assert weekly_effect.drift[d]["classification"] in ("stable", "drifting")


def test_weekly_holiday_masking():
    """Holiday masking should still produce valid estimates."""
    y, true_effects = make_weekly_series()
    # Add holidays on Christmas dates
    holidays = [
        datetime.date(2022, 12, 25),
        datetime.date(2023, 12, 25),
    ]
    weekly_effect, y_w, is_mult = fit_weekly(y, holidays)
    assert weekly_effect.coefficients.shape == (7,)
    # With holidays masked, should still recover within 10%
    true_norm = true_effects - true_effects.mean()
    effect_range = true_norm.max() - true_norm.min()
    tolerance = 0.10 * effect_range
    errors = np.abs(weekly_effect.coefficients - true_norm)
    assert float(errors.max()) <= tolerance


def test_weekly_multiplicative_detection():
    """High CV series should trigger multiplicative mode."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2022-01-01", periods=365, freq="D")
    # Create series with large relative variation (CV > 0.15)
    trend = np.linspace(10.0, 100.0, 365)
    dow = dates.dayofweek
    mult_effects = np.array([1.2, 1.1, 1.0, 0.95, 0.9, 0.8, 1.1])
    y_vals = trend * mult_effects[dow] + rng.normal(0, 0.5, 365)
    y = pd.Series(y_vals, index=dates)

    _, _, is_mult = fit_weekly(y, [])
    assert is_mult, "High CV series should be detected as multiplicative"
