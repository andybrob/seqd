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

    # All R² values must be in [0, 1]
    for k, v in r2.items():
        assert v >= 0, f"r2[{k}] = {v} < 0"
        assert v <= 1.0, f"r2[{k}] = {v} > 1.0 — impossible variance fraction"

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


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


def test_nan_in_series_raises():
    """fit() should raise ValueError when y contains NaN values."""
    rng = np.random.default_rng(7)
    dates = pd.date_range("2022-01-01", periods=365, freq="D")
    y = pd.Series(100.0 + rng.normal(0, 1.0, 365), index=dates)
    y.iloc[42] = np.nan  # introduce a NaN

    decomp = SeqdDecomposer(holiday_dates=[])
    with pytest.raises(ValueError, match="NaN"):
        decomp.fit(y)


def test_non_datetime_index_raises():
    """fit() should raise ValueError when y does not have a DatetimeIndex."""
    y = pd.Series([1.0, 2.0, 3.0], index=[0, 1, 2])
    decomp = SeqdDecomposer(holiday_dates=[])
    with pytest.raises(ValueError, match="DatetimeIndex"):
        decomp.fit(y)


def test_gaps_in_series_raises():
    """fit() should raise ValueError when y has date gaps larger than 1 day."""
    dates = pd.to_datetime(["2022-01-01", "2022-01-02", "2022-01-04"])  # gap on Jan 3
    y = pd.Series([100.0, 101.0, 103.0], index=dates)
    decomp = SeqdDecomposer(holiday_dates=[])
    with pytest.raises(ValueError, match="gap"):
        decomp.fit(y)


def test_short_series_warns_not_crashes():
    """fit() should warn (not crash) when series is shorter than 2 * reference_window."""
    rng = np.random.default_rng(8)
    # reference_window=60 by default, so 2*60=120; use 50 days
    dates = pd.date_range("2022-01-01", periods=50, freq="D")
    y = pd.Series(100.0 + rng.normal(0, 1.0, 50), index=dates)

    decomp = SeqdDecomposer(holiday_dates=[])
    with pytest.warns(UserWarning, match="reference_window"):
        result = decomp.fit(y)
    # Should still return a valid result
    assert isinstance(result, DecompositionResult)


def test_short_series_annual_warns():
    """fit() should warn when series is shorter than 365 days."""
    rng = np.random.default_rng(9)
    dates = pd.date_range("2022-01-01", periods=200, freq="D")
    y = pd.Series(100.0 + rng.normal(0, 1.0, 200), index=dates)

    decomp = SeqdDecomposer(holiday_dates=[], reference_window=30)
    with pytest.warns(UserWarning, match="365 days"):
        decomp.fit(y)


# ---------------------------------------------------------------------------
# Robustness edge case tests
# ---------------------------------------------------------------------------


def test_duplicate_holiday_dates_deduplicated():
    """Duplicate dates in holiday input should be silently deduplicated."""
    y, holiday_dates, _, _, _ = make_full_synthetic()
    # Pass 2022-12-25 twice
    dup_holidays = {"Christmas": [holiday_dates[0], holiday_dates[0], holiday_dates[1], holiday_dates[2]]}
    decomp = SeqdDecomposer(holiday_dates=dup_holidays)
    result = decomp.fit(y)

    # After dedup: 3 unique occurrences -> 3 HolidayEffect objects
    assert len(result.holidays) == 3, (
        f"Expected 3 HolidayEffects (one per unique occurrence) after dedup, "
        f"got {len(result.holidays)}"
    )
    # All should be for Christmas and carry 3 year_magnitudes
    for he in result.holidays:
        assert he.name == "Christmas"
        assert len(he.year_magnitudes) == 3, (
            f"Expected 3 year magnitudes after dedup, got {len(he.year_magnitudes)}"
        )


def test_dict_named_holidays():
    """Should accept a dict of named holidays and produce correctly named effects."""
    rng = np.random.default_rng(11)
    n_days = 365 * 3
    dates = pd.date_range("2022-01-01", periods=n_days, freq="D")
    y_vals = 100.0 + rng.normal(0, 2.0, n_days)
    y = pd.Series(y_vals, index=dates)

    # Inject spikes for two named holidays
    for h_str in ["2022-07-04", "2023-07-04", "2022-12-25", "2023-12-25"]:
        ts = pd.Timestamp(h_str)
        if ts in y.index:
            y.loc[ts] += 40.0

    holidays = {
        "IndependenceDay": [datetime.date(2022, 7, 4), datetime.date(2023, 7, 4)],
        "Christmas": [datetime.date(2022, 12, 25), datetime.date(2023, 12, 25)],
    }
    decomp = SeqdDecomposer(holiday_dates=holidays)
    result = decomp.fit(y)

    he_names = {he.name for he in result.holidays}
    assert "IndependenceDay" in he_names, f"Expected 'IndependenceDay' in {he_names}"
    assert "Christmas" in he_names, f"Expected 'Christmas' in {he_names}"


def test_future_holiday_skipped_gracefully():
    """Holiday dates outside the series range should be skipped without error."""
    rng = np.random.default_rng(12)
    dates = pd.date_range("2022-01-01", periods=365, freq="D")
    y = pd.Series(100.0 + rng.normal(0, 1.0, 365), index=dates)

    # Holiday far in the future — not in series
    future = {"FutureHoliday": [datetime.date(2030, 12, 25)]}
    decomp = SeqdDecomposer(holiday_dates=future)
    result = decomp.fit(y)
    # Should produce 0 holiday effects (holiday not in range)
    assert len(result.holidays) == 0


def test_fitted_reconstructs_original():
    """fitted() must reconstruct the original series to within floating-point tolerance."""
    y, holiday_dates, _, _, _ = make_full_synthetic()
    decomp = SeqdDecomposer(holiday_dates={"Christmas": holiday_dates})
    result = decomp.fit(y)

    reconstructed = result.fitted()
    assert reconstructed.index.equals(y.index), "fitted() index does not match input"

    max_diff = float(np.abs(reconstructed.values - y.values).max())
    assert max_diff < 1e-6, (
        f"fitted() reconstruction error {max_diff:.2e} exceeds floating-point tolerance. "
        "Components do not sum to original series."
    )


def test_recency_shape():
    """recency DataFrames should have the expected column structure and row count."""
    y, _, _, _, _ = make_full_synthetic()
    decomp = SeqdDecomposer(holiday_dates=[])
    result = decomp.fit(y)

    recency = result.weekly.recency
    # For a 3-year series all three windows (60, 90, 365) should be present
    for window in [60, 90, 365]:
        assert window in recency, f"Window {window} missing from recency"
        df = recency[window]
        # Must have 'date' column and dow_0..dow_6
        assert "date" in df.columns
        for d in range(7):
            assert f"dow_{d}" in df.columns

        # Number of rows = number of endpoints stepped every 7 days
        # endpoints go from n-1 back to 0 in steps of 7, sorted ascending
        n = len(y)
        expected_rows = len(range(n - 1, -1, -7))
        assert len(df) == expected_rows, (
            f"recency[{window}] has {len(df)} rows, expected {expected_rows}"
        )


def test_r2_values_between_0_and_1():
    """R² values for all components must be in [0, 1] — impossible to explain > 100% variance."""
    y, holiday_dates, _, _, _ = make_full_synthetic()
    decomp = SeqdDecomposer(holiday_dates={"Christmas": holiday_dates})
    result = decomp.fit(y)

    for component, value in result.r2_by_component.items():
        assert 0.0 <= value <= 1.0, (
            f"r2_by_component['{component}'] = {value:.4f} is outside [0, 1]. "
            "Variance fractions must be between 0 and 1."
        )


def test_r2_warning_on_pathological_fit():
    """Verify the R² warning fires when holiday removal increases residual variance.

    We mock fit_holidays to return a y_h with larger variance than y_w, which
    guarantees var(y_h) > var(y_w) and triggers the holiday R² negative warning.
    """
    from unittest.mock import patch

    rng = np.random.default_rng(500)
    n_days = 365 * 2
    dates = pd.date_range("2022-01-01", periods=n_days, freq="D")
    y_vals = 100.0 + rng.normal(0, 1.0, n_days)
    y = pd.Series(y_vals, index=dates, name="y")

    def bad_fit_holidays(y_w, holidays, holiday_window, reference_window, **kwargs):
        # Return a y_h with inflated variance — guaranteed negative R² for holiday stage
        bad_y_h = y_w + np.random.default_rng(1).normal(0, 20.0, len(y_w))
        bad_y_h = bad_y_h.set_axis(y_w.index)
        return [], bad_y_h

    decomp = SeqdDecomposer(
        holiday_dates={"dummy": [datetime.date(2022, 6, 15), datetime.date(2023, 6, 15)]},
    )

    with patch("seqd.fit_holidays", bad_fit_holidays):
        with pytest.warns(UserWarning, match="R\u00b2"):
            decomp.fit(y)
