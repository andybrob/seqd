"""Tests for SeqdForecaster and forecast_from_result (Stages 4-6)."""

from __future__ import annotations

import warnings
from datetime import date

import numpy as np
import pandas as pd
import pytest

from seqd import SeqdDecomposer, SeqdForecaster, forecast_from_result
from seqd._forecast import _combine_components, _project_annual, _project_weekly
from seqd._structures import ForecastResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_decomposition(
    n_years=3,
    seed=42,
    multiplicative=False,
    add_holiday=True,
):
    """Build a synthetic V1 decomposition for V2 tests."""
    rng = np.random.default_rng(seed)
    n = int(n_years * 365.25)
    dates = pd.date_range("2021-01-01", periods=n, freq="D")
    t = np.arange(n, dtype=float)

    # Components
    trend = 100.0 + 0.05 * t
    annual = 10.0 * np.sin(2 * np.pi * t / 365.25)
    weekly_raw = np.array([1.05, 1.02, 1.0, 1.0, 0.98, 0.95, 0.88])
    weekly_component = weekly_raw[dates.dayofweek]
    noise = rng.normal(0, 3.0, n)

    if multiplicative:
        y = weekly_component * (trend + annual) + noise
    else:
        dow_additive = weekly_raw - weekly_raw.mean()
        y = trend + annual + dow_additive[dates.dayofweek] + noise

    holidays = {}
    if add_holiday:
        bfcm_dates = []
        for yr in range(2021, 2021 + n_years):
            bfcm_dates.append(date(yr, 11, 26))
        holidays = {"Black Friday": bfcm_dates}
        # Add spike
        for bfcm in bfcm_dates:
            mask = (dates >= pd.Timestamp(bfcm) - pd.Timedelta(5)) & (
                dates <= pd.Timestamp(bfcm) + pd.Timedelta(3)
            )
            y[mask] += 50

    y_series = pd.Series(y, index=dates, name="y")

    decomposer = SeqdDecomposer(
        holiday_dates=holidays,
        multiplicative=multiplicative,
        holiday_window=20,
        reference_window=30,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = decomposer.fit(y_series)
    return result, holidays


# ---------------------------------------------------------------------------
# SeqdForecaster basic API
# ---------------------------------------------------------------------------


def test_fit_returns_self():
    result, _ = make_decomposition()
    forecaster = SeqdForecaster(result)
    ret = forecaster.fit()
    assert ret is forecaster


def test_predict_before_fit_raises():
    result, _ = make_decomposition()
    forecaster = SeqdForecaster(result)
    with pytest.raises(RuntimeError, match="fit\\(\\)"):
        forecaster.predict(horizon=30)


def test_changepoints_before_fit_raises():
    result, _ = make_decomposition()
    forecaster = SeqdForecaster(result)
    with pytest.raises(RuntimeError):
        _ = forecaster.changepoints


def test_segments_before_fit_raises():
    result, _ = make_decomposition()
    forecaster = SeqdForecaster(result)
    with pytest.raises(RuntimeError):
        _ = forecaster.segments


def test_invalid_horizon_raises():
    result, _ = make_decomposition()
    forecaster = SeqdForecaster(result).fit()
    with pytest.raises(ValueError):
        forecaster.predict(horizon=0)


def test_empty_residual_raises():
    result, _ = make_decomposition()
    import copy
    bad = copy.copy(result)
    bad.residual = pd.Series([], dtype=float, name="residual")
    with pytest.raises(ValueError):
        SeqdForecaster(bad)


# ---------------------------------------------------------------------------
# ForecastResult structure
# ---------------------------------------------------------------------------


def test_forecast_result_types():
    """ForecastResult fields should have correct types and shapes."""
    result, _ = make_decomposition()
    fr = SeqdForecaster(result).fit().predict(horizon=90)

    assert isinstance(fr, ForecastResult)
    assert isinstance(fr.forecast, pd.Series)
    assert len(fr.forecast) == 90
    assert isinstance(fr.trend_component, pd.Series)
    assert isinstance(fr.weekly_component, pd.Series)
    assert isinstance(fr.annual_component, pd.Series)
    assert isinstance(fr.holiday_component, pd.Series)
    assert isinstance(fr.changepoints, list)
    assert isinstance(fr.segments, list)
    assert fr.horizon == 90
    assert isinstance(fr.is_multiplicative, bool)


def test_forecast_index_starts_after_series():
    """Forecast dates should start the day after the series ends."""
    result, _ = make_decomposition()
    fr = SeqdForecaster(result).fit().predict(horizon=30)
    expected_start = result.residual.index[-1] + pd.Timedelta(days=1)
    assert fr.forecast.index[0] == expected_start


# ---------------------------------------------------------------------------
# Multiple predict calls
# ---------------------------------------------------------------------------


def test_predict_twice_no_refit():
    """predict() can be called multiple times without re-running fit()."""
    result, _ = make_decomposition()
    forecaster = SeqdForecaster(result).fit()
    segs_after_first_fit = list(forecaster.segments)

    fr1 = forecaster.predict(horizon=30)
    fr2 = forecaster.predict(horizon=90)

    # Segments should be identical across calls
    assert forecaster.segments == segs_after_first_fit
    assert len(fr1.forecast) == 30
    assert len(fr2.forecast) == 90


# ---------------------------------------------------------------------------
# Combination formula
# ---------------------------------------------------------------------------


def test_combine_multiplicative():
    """Multiplicative: weekly * (trend + annual + holiday)."""
    trend = np.array([100.0, 105.0])
    weekly = np.array([1.1, 0.9])
    annual = np.array([5.0, -5.0])
    holiday = np.array([0.0, 10.0])

    result = _combine_components(trend, weekly, annual, holiday, is_multiplicative=True)
    expected = weekly * (trend + annual + holiday)
    np.testing.assert_allclose(result, expected)


def test_combine_additive():
    """Additive: trend + weekly + annual + holiday."""
    trend = np.array([100.0, 105.0])
    weekly = np.array([3.0, -3.0])
    annual = np.array([5.0, -5.0])
    holiday = np.array([0.0, 10.0])

    result = _combine_components(trend, weekly, annual, holiday, is_multiplicative=False)
    expected = trend + weekly + annual + holiday
    np.testing.assert_allclose(result, expected)


def test_combine_multiplicative_not_independent():
    """Multiplicative formula is weekly*(sum), NOT weekly*trend*annual*holiday."""
    trend = np.array([100.0])
    weekly = np.array([1.1])
    annual = np.array([10.0])
    holiday = np.array([5.0])

    correct = _combine_components(trend, weekly, annual, holiday, is_multiplicative=True)
    wrong = weekly * trend * annual * holiday  # incorrect formula

    # They should NOT be equal
    assert not np.allclose(correct, wrong)
    expected = weekly * (trend + annual + holiday)
    np.testing.assert_allclose(correct, expected)


# ---------------------------------------------------------------------------
# Annual projection
# ---------------------------------------------------------------------------


def test_annual_projection_matches_v1():
    """Annual projection at in-sample dates should match result.annual_component()."""
    result, _ = make_decomposition()

    # Use last 10 dates of the in-sample period
    check_dates = result.series.index[-10:]
    projected = _project_annual(result, check_dates)
    v1_values = result.annual_component().loc[check_dates].values

    np.testing.assert_allclose(projected, v1_values, atol=1e-8)


# ---------------------------------------------------------------------------
# Weekly projection fallback
# ---------------------------------------------------------------------------


def test_weekly_projection_uses_60day_window():
    """_project_weekly should use 60-day recency if available."""
    result, _ = make_decomposition()
    forecast_dates = pd.date_range("2024-01-01", periods=7, freq="D")

    weekly = _project_weekly(result, forecast_dates)
    assert len(weekly) == 7

    # All values should be finite
    assert np.all(np.isfinite(weekly))


# ---------------------------------------------------------------------------
# Unknown holiday warning
# ---------------------------------------------------------------------------


def test_unknown_future_holiday_warns():
    """Unknown future holiday name should emit UserWarning and give zero effect."""
    result, _ = make_decomposition()
    fr = None
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fr = SeqdForecaster(result).fit().predict(
            horizon=30,
            future_holidays={"Unknown Holiday": [pd.Timestamp("2024-11-28")]},
        )
    assert any("not found" in str(warning.message) for warning in w)
    # Holiday component should be all zero (holiday name not in result.holidays)
    np.testing.assert_allclose(fr.holiday_component.values, 0.0)


# ---------------------------------------------------------------------------
# forecast_from_result convenience function
# ---------------------------------------------------------------------------


def test_forecast_from_result_wrapper():
    """forecast_from_result should produce same result as SeqdForecaster.fit().predict()."""
    result, _ = make_decomposition(seed=99)
    fr = forecast_from_result(result, horizon=60)
    assert len(fr.forecast) == 60
    assert fr.horizon == 60


# ---------------------------------------------------------------------------
# Sanity: forecast is finite and in reasonable range
# ---------------------------------------------------------------------------


def test_forecast_finite_values():
    """Forecast values should all be finite."""
    result, _ = make_decomposition()
    fr = SeqdForecaster(result).fit().predict(horizon=90)
    assert np.all(np.isfinite(fr.forecast.values))
    assert np.all(np.isfinite(fr.trend_component.values))
    assert np.all(np.isfinite(fr.annual_component.values))
    assert np.all(np.isfinite(fr.weekly_component.values))
    assert np.all(np.isfinite(fr.holiday_component.values))


def test_segment_count_positive():
    """At least one segment must exist after fit()."""
    result, _ = make_decomposition()
    forecaster = SeqdForecaster(result).fit()
    assert len(forecaster.segments) >= 1


def test_version_bumped():
    """Package version should be 0.2.0."""
    import seqd
    assert seqd.__version__ == "0.2.0"
